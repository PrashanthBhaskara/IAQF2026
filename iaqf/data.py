from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from io import BytesIO
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from iaqf.config import STUDY_END, STUDY_START, RepoPaths

STUDY_INDEX = pd.date_range(STUDY_START, STUDY_END, freq="1min", tz="UTC")
STUDY_INDEX.freq = None

RAW_MARKETS = (
    "binance_btcusdt",
    "binance_btcusdc",
    "binance_usdcusdt",
    "coinbase_btcusd",
    "coinbase_btcusdt",
    "coinbase_usdtusd",
    "kraken_btcusd",
    "kraken_btcusdt",
    "kraken_btcusdc",
    "kraken_usdcusd",
    "kraken_usdtusd",
)

PRICE_COLUMNS = RAW_MARKETS + (
    "implied_usdt_usd_coinbase",
    "implied_usdt_usd_kraken",
    "implied_usdc_usd_kraken",
)

BASIS_COLUMNS = (
    "dispersion_usdc_kraken",
    "dispersion_usdt_kraken",
    "basis_usdc_kraken",
    "basis_usdt_kraken",
    "dispersion_usdt_coinbase",
    "basis_usdt_coinbase",
    "basis_usdc_usdt_binance",
    "xbasis_btcusdt_binance_kraken",
    "xbasis_btcusdt_coinbase_kraken",
    "xbasis_btcusd_coinbase_kraken",
    "usdt_peg_dev_kraken",
    "usdt_peg_dev_coinbase",
    "usdc_peg_dev_kraken",
)

RAW_CANDLE_COLUMNS = ("open", "high", "low", "close", "volume")

PROCESSED_FILES = {
    "prices": "prices.parquet",
    "price_ffill_flags": "price_ffill_flags.parquet",
    "ranges": "intraminute_ranges.parquet",
    "volumes": "volumes.parquet",
    "basis": "basis.parquet",
    "basis_ffill_flags": "basis_ffill_flags.parquet",
}

PROCESSED_COLUMNS = {
    "prices": PRICE_COLUMNS,
    "price_ffill_flags": PRICE_COLUMNS,
    "ranges": RAW_MARKETS,
    "volumes": RAW_MARKETS,
    "basis": BASIS_COLUMNS,
    "basis_ffill_flags": BASIS_COLUMNS,
}


@dataclass(frozen=True)
class AnalysisData:
    prices: pd.DataFrame
    price_ffill_flags: pd.DataFrame
    ranges: pd.DataFrame
    volumes: pd.DataFrame
    basis: pd.DataFrame
    basis_ffill_flags: pd.DataFrame


def _validate_raw_frames(raw: Mapping[str, pd.DataFrame]) -> None:
    missing = [market for market in RAW_MARKETS if market not in raw]
    unexpected = [market for market in raw if market not in RAW_MARKETS]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing markets: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected markets: {', '.join(unexpected)}")
        raise ValueError("; ".join(details))

    for market in RAW_MARKETS:
        frame = raw[market]
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{market}: expected a non-empty candle DataFrame")
        missing_columns = [
            column for column in RAW_CANDLE_COLUMNS if column not in frame.columns
        ]
        if missing_columns:
            raise ValueError(
                f"{market}: missing candle columns {', '.join(missing_columns)}"
            )
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"{market}: candle index must be a DatetimeIndex")
        if frame.index.tz is None or str(frame.index.tz) != "UTC":
            raise ValueError(f"{market}: candle index must use UTC")
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"{market}: candle index must be sorted")
        if frame.index.has_duplicates:
            raise ValueError(f"{market}: candle index must be unique")
        for column in RAW_CANDLE_COLUMNS:
            if not is_numeric_dtype(frame[column]):
                raise ValueError(f"{market}: {column} must be numeric")


def _read_raw_files(paths: RepoPaths) -> dict[str, pd.DataFrame]:
    missing = [
        market
        for market in RAW_MARKETS
        if not (paths.raw / f"{market}.parquet").is_file()
    ]
    if missing:
        raise ValueError(f"missing raw markets: {', '.join(missing)}")

    raw = {}
    for market in RAW_MARKETS:
        path = paths.raw / f"{market}.parquet"
        try:
            raw[market] = pd.read_parquet(path)
        except Exception as error:
            raise ValueError(f"{market}: could not read {path}") from error
    return raw


def validate_raw(paths: RepoPaths) -> None:
    _validate_raw_frames(_read_raw_files(paths))


def load_raw(paths: RepoPaths) -> dict[str, pd.DataFrame]:
    raw = _read_raw_files(paths)
    _validate_raw_frames(raw)
    return raw


def _extract(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].reindex(STUDY_INDEX)


def build_master(raw: Mapping[str, pd.DataFrame]) -> AnalysisData:
    _validate_raw_frames(raw)

    raw_prices = {market: _extract(raw[market], "close") for market in RAW_MARKETS}
    prices = pd.DataFrame(index=STUDY_INDEX)
    price_ffill_flags = pd.DataFrame(index=STUDY_INDEX)
    for market, raw_price in raw_prices.items():
        filled = raw_price.ffill(limit=5)
        prices[market] = filled
        price_ffill_flags[market] = raw_price.isna() & filled.notna()

    ranges = pd.DataFrame(
        {
            market: (
                (_extract(raw[market], "high") - _extract(raw[market], "low"))
                / _extract(raw[market], "close")
            ).ffill(limit=5)
            for market in RAW_MARKETS
        },
        index=STUDY_INDEX,
    )
    volumes = pd.DataFrame(
        {market: _extract(raw[market], "volume") for market in RAW_MARKETS},
        index=STUDY_INDEX,
    )

    basis = pd.DataFrame(index=STUDY_INDEX)
    basis_ffill_flags = pd.DataFrame(index=STUDY_INDEX)

    basis["dispersion_usdc_kraken"] = (
        np.log(prices["kraken_btcusdc"]) - np.log(prices["kraken_btcusd"])
    ) * 10_000
    basis_ffill_flags["dispersion_usdc_kraken"] = price_ffill_flags[
        ["kraken_btcusdc", "kraken_btcusd"]
    ].any(axis=1)

    basis["dispersion_usdt_kraken"] = (
        np.log(prices["kraken_btcusdt"]) - np.log(prices["kraken_btcusd"])
    ) * 10_000
    basis_ffill_flags["dispersion_usdt_kraken"] = price_ffill_flags[
        ["kraken_btcusdt", "kraken_btcusd"]
    ].any(axis=1)

    basis["basis_usdc_kraken"] = (
        np.log(prices["kraken_btcusdc"] * prices["kraken_usdcusd"])
        - np.log(prices["kraken_btcusd"])
    ) * 10_000
    basis_ffill_flags["basis_usdc_kraken"] = price_ffill_flags[
        ["kraken_btcusdc", "kraken_usdcusd", "kraken_btcusd"]
    ].any(axis=1)

    basis["basis_usdt_kraken"] = (
        np.log(prices["kraken_btcusdt"] * prices["kraken_usdtusd"])
        - np.log(prices["kraken_btcusd"])
    ) * 10_000
    basis_ffill_flags["basis_usdt_kraken"] = price_ffill_flags[
        ["kraken_btcusdt", "kraken_usdtusd", "kraken_btcusd"]
    ].any(axis=1)

    basis["dispersion_usdt_coinbase"] = (
        np.log(prices["coinbase_btcusdt"]) - np.log(prices["coinbase_btcusd"])
    ) * 10_000
    basis_ffill_flags["dispersion_usdt_coinbase"] = price_ffill_flags[
        ["coinbase_btcusdt", "coinbase_btcusd"]
    ].any(axis=1)

    basis["basis_usdt_coinbase"] = (
        np.log(prices["coinbase_btcusdt"] * prices["coinbase_usdtusd"])
        - np.log(prices["coinbase_btcusd"])
    ) * 10_000
    basis_ffill_flags["basis_usdt_coinbase"] = price_ffill_flags[
        ["coinbase_btcusdt", "coinbase_usdtusd", "coinbase_btcusd"]
    ].any(axis=1)

    basis["basis_usdc_usdt_binance"] = (
        np.log(prices["binance_btcusdc"] * prices["binance_usdcusdt"])
        - np.log(prices["binance_btcusdt"])
    ) * 10_000
    basis_ffill_flags["basis_usdc_usdt_binance"] = price_ffill_flags[
        ["binance_btcusdc", "binance_usdcusdt", "binance_btcusdt"]
    ].any(axis=1)

    basis["xbasis_btcusdt_binance_kraken"] = (
        np.log(prices["binance_btcusdt"]) - np.log(prices["kraken_btcusdt"])
    ) * 10_000
    basis_ffill_flags["xbasis_btcusdt_binance_kraken"] = price_ffill_flags[
        ["binance_btcusdt", "kraken_btcusdt"]
    ].any(axis=1)

    basis["xbasis_btcusdt_coinbase_kraken"] = (
        np.log(prices["coinbase_btcusdt"]) - np.log(prices["kraken_btcusdt"])
    ) * 10_000
    basis_ffill_flags["xbasis_btcusdt_coinbase_kraken"] = price_ffill_flags[
        ["coinbase_btcusdt", "kraken_btcusdt"]
    ].any(axis=1)

    basis["xbasis_btcusd_coinbase_kraken"] = (
        np.log(prices["coinbase_btcusd"]) - np.log(prices["kraken_btcusd"])
    ) * 10_000
    basis_ffill_flags["xbasis_btcusd_coinbase_kraken"] = price_ffill_flags[
        ["coinbase_btcusd", "kraken_btcusd"]
    ].any(axis=1)

    prices["implied_usdt_usd_coinbase"] = (
        prices["coinbase_btcusd"] / prices["coinbase_btcusdt"]
    )
    prices["implied_usdt_usd_kraken"] = (
        prices["kraken_btcusd"] / prices["kraken_btcusdt"]
    )
    prices["implied_usdc_usd_kraken"] = (
        prices["kraken_btcusd"] / prices["kraken_btcusdc"]
    )
    price_ffill_flags["implied_usdt_usd_coinbase"] = False
    price_ffill_flags["implied_usdt_usd_kraken"] = False
    price_ffill_flags["implied_usdc_usd_kraken"] = False

    basis["usdt_peg_dev_kraken"] = (prices["kraken_usdtusd"] - 1.0) * 10_000
    basis["usdt_peg_dev_coinbase"] = (prices["coinbase_usdtusd"] - 1.0) * 10_000
    basis["usdc_peg_dev_kraken"] = (prices["kraken_usdcusd"] - 1.0) * 10_000
    basis_ffill_flags["usdt_peg_dev_kraken"] = price_ffill_flags["kraken_usdtusd"]
    basis_ffill_flags["usdt_peg_dev_coinbase"] = price_ffill_flags["coinbase_usdtusd"]
    basis_ffill_flags["usdc_peg_dev_kraken"] = price_ffill_flags["kraken_usdcusd"]

    price_ffill_flags = price_ffill_flags.reindex(columns=PRICE_COLUMNS).astype(bool)
    basis_ffill_flags = basis_ffill_flags.reindex(columns=BASIS_COLUMNS).astype(bool)

    return AnalysisData(
        prices=prices,
        price_ffill_flags=price_ffill_flags,
        ranges=ranges,
        volumes=volumes,
        basis=basis,
        basis_ffill_flags=basis_ffill_flags,
    )


def _validate_processed(data: AnalysisData) -> None:
    if not isinstance(data, AnalysisData):
        raise TypeError("processed data must be an AnalysisData instance")
    for attribute, expected_columns in PROCESSED_COLUMNS.items():
        frame = getattr(data, attribute)
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{attribute}: expected a DataFrame")
        if not frame.index.equals(STUDY_INDEX):
            raise ValueError(f"{attribute}: index does not match the study grid")
        if tuple(frame.columns) != expected_columns:
            raise ValueError(f"{attribute}: columns do not match the study schema")
        if attribute.endswith("_flags"):
            if not all(is_bool_dtype(dtype) for dtype in frame.dtypes):
                raise ValueError(f"{attribute}: flag columns must be boolean")
        elif not all(is_numeric_dtype(dtype) for dtype in frame.dtypes):
            raise ValueError(f"{attribute}: columns must be numeric")


def load_processed(paths: RepoPaths) -> AnalysisData:
    frames = {}
    for attribute, filename in PROCESSED_FILES.items():
        path = paths.processed / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing processed artifact: {path}")
        try:
            frames[attribute] = pd.read_parquet(path)
        except Exception as error:
            raise ValueError(f"could not read processed artifact: {path}") from error
    data = AnalysisData(**frames)
    _validate_processed(data)
    return data


def write_processed(data: AnalysisData, paths: RepoPaths) -> None:
    _validate_processed(data)
    paths.processed.mkdir(parents=True, exist_ok=True)
    for attribute, filename in PROCESSED_FILES.items():
        getattr(data, attribute).to_parquet(paths.processed / filename)


def _fetch_binance_vision(
    session: Any,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    frames = []
    months = pd.date_range(
        pd.Timestamp(start_ms, unit="ms", tz="UTC").replace(day=1),
        pd.Timestamp(end_ms, unit="ms", tz="UTC").replace(day=1),
        freq="MS",
    )
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    for month in months:
        year_month = month.strftime("%Y-%m")
        url = (
            "https://data.binance.vision/data/spot/monthly/klines/"
            f"{symbol}/1m/{symbol}-1m-{year_month}.zip"
        )
        response = session.get(url, timeout=60)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as archive:
            frames.append(
                pd.read_csv(
                    archive.open(archive.namelist()[0]),
                    header=None,
                    names=columns,
                )
            )
    if not frames:
        return pd.DataFrame()
    return _prepare_binance(pd.concat(frames, ignore_index=True), start_ms, end_ms)


def _prepare_binance(
    frame: pd.DataFrame,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume", "quote_volume"):
        frame[column] = frame[column].astype(float)
    frame["trades"] = frame["trades"].astype(int)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    return frame[(frame.index >= start) & (frame.index <= end)]


def _fetch_binance(session: Any, symbol: str) -> pd.DataFrame:
    start_ms = STUDY_START.value // 1_000_000
    end_ms = STUDY_END.value // 1_000_000
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    url = "https://api.binance.com/api/v3/klines"
    used_fallback = False
    rows = []
    current_start = start_ms
    while current_start < end_ms:
        response = session.get(
            url,
            params={
                "symbol": symbol,
                "interval": "1m",
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1_000,
            },
            timeout=30,
        )
        if response.status_code == 451 and not used_fallback:
            url = "https://api.binance.us/api/v3/klines"
            used_fallback = True
            continue
        if response.status_code == 429:
            time.sleep(60)
            continue
        if response.status_code == 400 and used_fallback:
            return _fetch_binance_vision(session, symbol, start_ms, end_ms)
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        rows.extend(page)
        current_start = int(page[-1][6]) + 1
    if not rows:
        return pd.DataFrame()
    return _prepare_binance(pd.DataFrame(rows, columns=columns), start_ms, end_ms)


def _fetch_coinbase(session: Any, product: str) -> pd.DataFrame:
    url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{product}/candles"
    rows = []
    current_start = STUDY_START
    chunk = timedelta(seconds=300 * 60)
    while current_start < STUDY_END:
        current_end = min(current_start + chunk, STUDY_END)
        response = session.get(
            url,
            params={
                "granularity": "ONE_MINUTE",
                "start": int(current_start.timestamp()),
                "end": int(current_end.timestamp()),
            },
            timeout=30,
        )
        if response.status_code == 429:
            time.sleep(5)
            continue
        if response.status_code == 404:
            return pd.DataFrame()
        response.raise_for_status()
        rows.extend(response.json().get("candles", []))
        current_start = current_end
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["start"].astype(int), unit="s", utc=True)
    for column in RAW_CANDLE_COLUMNS:
        frame[column] = frame[column].astype(float)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    return frame[(frame.index >= STUDY_START) & (frame.index <= STUDY_END)]


def _fetch_kraken(session: Any, pair: str) -> pd.DataFrame:
    url = "https://api.kraken.com/0/public/Trades"
    since_ns = STUDY_START.value
    rows = []
    while True:
        response = session.get(
            url,
            params={"pair": pair, "since": str(since_ns)},
            timeout=30,
        )
        if response.status_code == 429:
            time.sleep(10)
            continue
        response.raise_for_status()
        payload = response.json()
        errors = payload.get("error", [])
        if errors:
            if any("Rate limit" in error for error in errors):
                time.sleep(10)
                continue
            raise RuntimeError(f"Kraken error for {pair}: {errors[0]}")
        result = payload["result"]
        new_since = int(result["last"])
        trade_key = next(key for key in result if key != "last")
        page = result[trade_key]
        if not page or new_since == since_ns:
            break
        rows.extend(page)
        since_ns = new_since
        if float(page[-1][2]) >= STUDY_END.timestamp():
            break
    if not rows:
        return pd.DataFrame()
    trades = pd.DataFrame(
        rows,
        columns=["price", "volume", "time", "side", "type", "misc", "trade_id"],
    )
    trades["price"] = trades["price"].astype(float)
    trades["volume"] = trades["volume"].astype(float)
    trades["timestamp"] = pd.to_datetime(
        trades["time"].astype(float), unit="s", utc=True
    )
    trades = trades[
        (trades["timestamp"] >= STUDY_START) & (trades["timestamp"] <= STUDY_END)
    ].set_index("timestamp")
    candles = trades["price"].resample("1min").ohlc()
    candles["volume"] = trades["volume"].resample("1min").sum()
    candles["trades"] = trades["price"].resample("1min").count()
    return candles.dropna(subset=["close"])


def _fetch_all_raw(session: Any) -> dict[str, pd.DataFrame]:
    raw = {
        "binance_btcusdt": _fetch_binance(session, "BTCUSDT"),
        "binance_btcusdc": _fetch_binance(session, "BTCUSDC"),
        "binance_usdcusdt": _fetch_binance(session, "USDCUSDT"),
        "coinbase_btcusd": _fetch_coinbase(session, "BTC-USD"),
        "coinbase_btcusdt": _fetch_coinbase(session, "BTC-USDT"),
        "coinbase_usdtusd": _fetch_coinbase(session, "USDT-USD"),
        "kraken_btcusd": _fetch_kraken(session, "XXBTZUSD"),
        "kraken_btcusdt": _fetch_kraken(session, "XBTUSDT"),
        "kraken_btcusdc": _fetch_kraken(session, "XBTUSDC"),
        "kraken_usdcusd": _fetch_kraken(session, "USDCUSD"),
        "kraken_usdtusd": _fetch_kraken(session, "USDTZUSD"),
    }
    _validate_raw_frames(raw)
    return raw


def refresh_raw(paths: RepoPaths, session: Any | None = None) -> None:
    owns_session = session is None
    client = requests.Session() if owns_session else session
    try:
        raw = _fetch_all_raw(client)
    finally:
        if owns_session:
            client.close()

    paths.raw.mkdir(parents=True, exist_ok=True)
    for market in RAW_MARKETS:
        raw[market].to_parquet(paths.raw / f"{market}.parquet")
    validate_raw(paths)
