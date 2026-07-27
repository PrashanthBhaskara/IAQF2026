from __future__ import annotations

import contextlib
from dataclasses import FrozenInstanceError
from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest


RAW_MARKETS = [
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
]

PROCESSED_ATTRIBUTES = [
    "prices",
    "price_ffill_flags",
    "ranges",
    "volumes",
    "basis",
    "basis_ffill_flags",
]

STUDY_START_MS = 1_677_628_800_000
STUDY_START_SECONDS = STUDY_START_MS // 1_000
STUDY_START_NS = STUDY_START_MS * 1_000_000


def data_api():
    try:
        return import_module("iaqf.config"), import_module("iaqf.data")
    except ModuleNotFoundError as error:
        raise AssertionError("missing planned iaqf data I/O API") from error


def one_candle_raw() -> dict[str, pd.DataFrame]:
    index = pd.DatetimeIndex(
        [pd.Timestamp("2023-03-01 00:00", tz="UTC")],
        name="timestamp",
    )
    candle = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [2.0],
        },
        index=index,
    )
    return {market: candle.copy() for market in RAW_MARKETS}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.content = b""

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        return None


class StudyDataSession:
    def get(self, url, params=None, timeout=None):
        del timeout
        params = params or {}
        if "binance.com/api" in url or "binance.us/api" in url:
            if params["startTime"] != STUDY_START_MS:
                return FakeResponse([])
            return FakeResponse(
                [
                    [
                        STUDY_START_MS,
                        "100",
                        "101",
                        "99",
                        "100",
                        "2",
                        STUDY_START_MS + 59_999,
                        "200",
                        3,
                        "1",
                        "100",
                        "0",
                    ]
                ]
            )
        if "coinbase.com" in url:
            candles = []
            if params["start"] == STUDY_START_SECONDS:
                candles = [
                    {
                        "start": str(STUDY_START_SECONDS),
                        "low": "99",
                        "high": "101",
                        "open": "100",
                        "close": "100",
                        "volume": "2",
                    }
                ]
            return FakeResponse({"candles": candles})
        if "kraken.com" in url:
            pair = params["pair"]
            trades = []
            last = int(params["since"])
            if last == STUDY_START_NS:
                trades = [
                    [
                        "100",
                        "2",
                        str(STUDY_START_SECONDS + 10),
                        "b",
                        "m",
                        "",
                        1,
                    ]
                ]
                last += 1
            return FakeResponse(
                {"error": [], "result": {pair: trades, "last": str(last)}}
            )
        raise AssertionError(f"unexpected provider URL: {url}")


def test_repo_paths_are_frozen_and_root_every_output_at_the_checkout(
    tmp_path: Path,
) -> None:
    config, _ = data_api()
    root = tmp_path / "checkout"

    paths = config.RepoPaths.from_root(root)

    assert paths.root == root
    assert paths.raw == root / "data_raw"
    assert paths.processed == root / "data_processed"
    assert paths.tables == root / "tables"
    assert paths.figures == root / "figures_col"
    assert paths.paper_tex == root / "IAQF_column_Final.tex"
    assert paths.paper_pdf == root / "IAQF_column_Final.pdf"
    with pytest.raises(FrozenInstanceError):
        paths.raw = tmp_path / "elsewhere"


def test_processed_data_round_trips_through_the_explicit_repo_paths(
    tmp_path: Path,
) -> None:
    config, data = data_api()
    paths = config.RepoPaths.from_root(tmp_path / "checkout")
    expected = data.build_master(one_candle_raw())

    data.write_processed(expected, paths)
    actual = data.load_processed(paths)

    for attribute in PROCESSED_ATTRIBUTES:
        pd.testing.assert_frame_equal(
            getattr(actual, attribute),
            getattr(expected, attribute),
            check_exact=True,
            check_freq=False,
        )


def test_load_processed_fails_loudly_when_an_artifact_is_missing(
    tmp_path: Path,
) -> None:
    config, data = data_api()
    paths = config.RepoPaths.from_root(tmp_path)
    paths.processed.mkdir()

    with pytest.raises(FileNotFoundError, match="prices.parquet"):
        data.load_processed(paths)


def test_refresh_raw_writes_only_valid_raw_data_below_explicit_root(
    tmp_path: Path,
) -> None:
    config, data = data_api()
    checkout = tmp_path / "checkout"
    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    paths = config.RepoPaths.from_root(checkout)

    with contextlib.chdir(unrelated_cwd):
        data.refresh_raw(paths, session=StudyDataSession())

    assert sorted(path.stem for path in paths.raw.glob("*.parquet")) == sorted(
        RAW_MARKETS
    )
    assert not paths.processed.exists()
    assert not (unrelated_cwd / "data_raw").exists()
    assert not (unrelated_cwd / "data_processed").exists()
