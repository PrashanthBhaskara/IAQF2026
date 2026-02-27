# DEPRECATED SUPPORT ARTIFACT
# This file is a legacy notebook extract retained for archival reference only.
# It is NOT part of the canonical reproducible pipeline for the submission.
# Use run_all.py and scripts in src/ for all manuscript tables/figures.
#
# ============================================================
# Cell 1: Imports & Configuration
# ============================================================
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime, timedelta, timezone
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (14, 6),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# Time window
START_DT = datetime(2023, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
END_DT   = datetime(2023, 3, 21, 23, 59, 0, tzinfo=timezone.utc)

START_MS = int(START_DT.timestamp() * 1000)
END_MS   = int(END_DT.timestamp() * 1000)

DATA_DIR = 'data_sparsh'
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Study window: {START_DT.isoformat()} to {END_DT.isoformat()}")
print(f"Expected 1-min candles per pair: {21 * 24 * 60:,}")
# ============================================================
# Cell 2: Binance Data Fetcher
# ============================================================

from io import BytesIO
from zipfile import ZipFile

def fetch_binance_vision(symbol: str, interval: str = '1m',
                         start_ms: int = START_MS, end_ms: int = END_MS) -> pd.DataFrame:
    """
    Fetch historical klines from Binance public data (data.binance.vision).
    No geo-restrictions. Downloads monthly CSV zips.
    """
    start_dt_local = pd.Timestamp(start_ms, unit='ms', tz='UTC')
    end_dt_local = pd.Timestamp(end_ms, unit='ms', tz='UTC')

    all_dfs = []
    current = start_dt_local.to_period('M')
    end_period = end_dt_local.to_period('M')

    while current <= end_period:
        year_month = current.strftime('%Y-%m')
        url = (f"https://data.binance.vision/data/spot/monthly/klines/"
               f"{symbol}/{interval}/{symbol}-{interval}-{year_month}.zip")
        print(f"    Downloading {url}")
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            print(f"    WARNING: No data at {url}")
            current += 1
            continue
        resp.raise_for_status()

        with ZipFile(BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            df_chunk = pd.read_csv(
                zf.open(csv_name), header=None,
                names=['open_time', 'open', 'high', 'low', 'close', 'volume',
                       'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                       'taker_buy_quote', 'ignore']
            )
            all_dfs.append(df_chunk)
        current += 1

    if not all_dfs:
        print(f"  WARNING: No data from binance.vision for {symbol}")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = df[col].astype(float)
    df['trades'] = df['trades'].astype(int)
    df = df.set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    # Filter to requested window
    df = df[(df.index >= pd.Timestamp(start_ms, unit='ms', tz='UTC')) &
            (df.index <= pd.Timestamp(end_ms, unit='ms', tz='UTC'))]
    print(f"    Got {len(df):,} rows from binance.vision")
    return df

def fetch_binance_klines(symbol: str, interval: str = '1m',
                         start_ms: int = START_MS, end_ms: int = END_MS,
                         limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical klines (candlestick) data from Binance.
    Paginates automatically in chunks of `limit` candles.

    Note: Binance.com returns HTTP 451 from restricted locations (e.g., US).
    In that case we automatically fall back to Binance US.
    """
    primary_url = 'https://api.binance.com/api/v3/klines'
    fallback_url = 'https://api.binance.us/api/v3/klines'
    url = primary_url
    used_fallback = False

    all_data = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': limit
        }
        resp = requests.get(url, params=params, timeout=30)

        if resp.status_code == 451 and not used_fallback:
            print("  Binance.com blocked (451). Switching to binance.us for data.")
            url = fallback_url
            used_fallback = True
            continue
        if resp.status_code == 429:
            print(f"  Rate limited on {symbol}, sleeping 60s...")
            time.sleep(60)
            continue
        if resp.status_code == 400 and used_fallback:
            print(f"  Binance.us returned 400 for {symbol}. Falling back to data.binance.vision ...")
            return fetch_binance_vision(symbol, interval, start_ms, end_ms)

        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_data.extend(data)
        # Next start = close_time of last candle + 1ms
        current_start = data[-1][6] + 1
        time.sleep(0.15)  # respect rate limits

    if not all_data:
        print(f"  WARNING: No data returned for Binance {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = df[col].astype(float)
    df['trades'] = df['trades'].astype(int)
    df = df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='first')]
    return df

print("Binance fetcher ready.")

# ============================================================
# Cell 3: Coinbase Data Fetcher
# ============================================================

def fetch_coinbase_candles(product_id: str, granularity: int = 60,
                           start_dt: datetime = START_DT,
                           end_dt: datetime = END_DT) -> pd.DataFrame:
    """
    Fetch historical candles from Coinbase Advanced Trade Market Data API.
    Max ~350 candles per request; paginates automatically.
    granularity is in seconds (60 = 1 minute).
    """
    url = f'https://api.coinbase.com/api/v3/brokerage/market/products/{product_id}/candles'

    gran_map = {
        60: 'ONE_MINUTE',
        300: 'FIVE_MINUTE',
        900: 'FIFTEEN_MINUTE',
        3600: 'ONE_HOUR',
        21600: 'SIX_HOUR',
        86400: 'ONE_DAY',
    }
    if granularity not in gran_map:
        raise ValueError("Unsupported granularity. Use 60, 300, 900, 3600, 21600, or 86400 seconds.")
    granularity_str = gran_map[granularity]

    all_data = []
    # Coinbase limit: < 350 candles per request. Use 300 to be safe.
    chunk_seconds = 300 * granularity
    current_start = start_dt

    while current_start < end_dt:
        current_end = min(current_start + timedelta(seconds=chunk_seconds), end_dt)
        params = {
            'granularity': granularity_str,
            'start': int(current_start.timestamp()),
            'end': int(current_end.timestamp())
        }
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            print(f"  Rate limited on {product_id}, sleeping 5s...")
            time.sleep(5)
            continue
        if resp.status_code == 404:
            print(f"  WARNING: Product {product_id} not found on Coinbase Advanced Trade")
            return pd.DataFrame()
        resp.raise_for_status()
        data = resp.json().get('candles', [])
        if data:
            all_data.extend(data)
        current_start = current_end
        time.sleep(0.2)  # respect rate limits

    if not all_data:
        print(f"  WARNING: No data returned for Coinbase {product_id}")
        return pd.DataFrame()

    # Coinbase returns: start, low, high, open, close, volume (as strings)
    df = pd.DataFrame(all_data)
    if df.empty:
        return df

    df['timestamp'] = pd.to_datetime(df['start'].astype(int), unit='s', utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df = df.set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df

print("Coinbase fetcher ready.")

# ============================================================
# Cell 4: Kraken Data Fetcher (from raw trades -> 1-min OHLCV)
# ============================================================

def fetch_kraken_ohlcv(pair: str,
                       start_dt: datetime = START_DT,
                       end_dt: datetime = END_DT) -> pd.DataFrame:
    """
    Fetch historical 1-min OHLCV from Kraken by paginating raw trades
    and aggregating into 1-minute candles.

    Kraken's OHLC endpoint only returns the last 720 candles, so
    we use the Trades endpoint which supports full history via the
    'since' parameter (nanosecond timestamp).

    pair: Kraken pair name, e.g. 'XXBTZEUR', 'XXBTZUSD', 'XBTUSDT', 'XBTUSDC'
    """
    url = 'https://api.kraken.com/0/public/Trades'
    since_ns = int(start_dt.timestamp() * 1e9)
    end_ts = end_dt.timestamp()

    all_trades = []
    call_count = 0

    while True:
        params = {'pair': pair, 'since': str(since_ns)}
        resp = requests.get(url, params=params, timeout=30)

        if resp.status_code == 429:
            print(f"  Rate limited on {pair}, sleeping 10s...")
            time.sleep(10)
            continue

        resp.raise_for_status()
        data = resp.json()

        if data.get('error'):
            print(f"  Kraken API error for {pair}: {data['error']}")
            break

        result = data['result']
        new_since = int(result['last'])
        trade_key = [k for k in result if k != 'last'][0]
        trades = result[trade_key]

        if not trades or new_since == since_ns:
            break

        all_trades.extend(trades)
        since_ns = new_since
        call_count += 1

        last_trade_ts = float(trades[-1][2])
        if last_trade_ts >= end_ts:
            break

        if call_count % 100 == 0:
            pct = (last_trade_ts - start_dt.timestamp()) / (end_dt.timestamp() - start_dt.timestamp()) * 100
            print(f"    {pair}: {call_count} calls, {pct:.0f}% done...")

        # Kraken public API: counter limit=15, decay=1/sec. Burst then pace.
        if call_count % 14 == 0:
            time.sleep(3)
        else:
            time.sleep(0.1)

    if not all_trades:
        print(f"  WARNING: No trades returned for Kraken {pair}")
        return pd.DataFrame()

    print(f"  Fetched {len(all_trades):,} trades in {call_count} API calls for {pair}")

    # Build DataFrame from trades: [price, volume, time, buy/sell, market/limit, misc, trade_id]
    df_trades = pd.DataFrame(all_trades, columns=['price', 'volume', 'time', 'side', 'type', 'misc', 'trade_id'])
    df_trades['price'] = df_trades['price'].astype(float)
    df_trades['volume'] = df_trades['volume'].astype(float)
    df_trades['timestamp'] = pd.to_datetime(df_trades['time'].astype(float), unit='s', utc=True)

    # Filter to study window
    df_trades = df_trades[(df_trades['timestamp'] >= start_dt) & (df_trades['timestamp'] <= end_dt)]

    # Resample to 1-minute OHLCV
    df_trades = df_trades.set_index('timestamp')
    ohlcv = df_trades['price'].resample('1min').ohlc()
    ohlcv['volume'] = df_trades['volume'].resample('1min').sum()
    ohlcv['trades'] = df_trades['price'].resample('1min').count()
    ohlcv = ohlcv.dropna(subset=['close'])

    print(f"  Built {len(ohlcv):,} 1-min candles for {pair}")
    return ohlcv

print("Kraken fetcher ready.")

# ============================================================
# Cell 5: Fetch ALL data (with caching)
# ============================================================

def cache_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.parquet")

def load_or_fetch(name: str, fetch_fn, *args, **kwargs) -> pd.DataFrame:
    """Load from cache if available, otherwise fetch and cache."""
    path = cache_path(name)
    if os.path.exists(path):
        print(f"  Loading cached: {name}")
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
        return df
    print(f"  Fetching: {name} ...")
    df = fetch_fn(*args, **kwargs)
    if not df.empty:
        df.to_parquet(path)
        print(f"  Saved {len(df):,} rows -> {path}")
    return df

# ----- Binance Pairs -----
print("=" * 60)
print("BINANCE")
print("=" * 60)

binance_btcusdt = load_or_fetch(
    'binance_btcusdt',
    fetch_binance_klines, 'BTCUSDT'
)

binance_btcusdc = load_or_fetch(
    'binance_btcusdc',
    fetch_binance_klines, 'BTCUSDC'
)

binance_btceur = load_or_fetch(
    'binance_btceur',
    fetch_binance_klines, 'BTCEUR'
)

# ----- Coinbase Pairs -----
print("\n" + "=" * 60)
print("COINBASE")
print("=" * 60)

coinbase_btcusd = load_or_fetch(
    'coinbase_btcusd',
    fetch_coinbase_candles, 'BTC-USD'
)

coinbase_btcusdt = load_or_fetch(
    'coinbase_btcusdt',
    fetch_coinbase_candles, 'BTC-USDT'
)

coinbase_btceur = load_or_fetch(
    'coinbase_btceur',
    fetch_coinbase_candles, 'BTC-EUR'
)

# NOTE: Coinbase BTC-USDC is an alias for BTC-USD (same order book), not fetched separately

# ----- Kraken Pairs -----
print("\n" + "=" * 60)
print("KRAKEN")
print("=" * 60)

kraken_btcusd = load_or_fetch(
    'kraken_btcusd',
    fetch_kraken_ohlcv, 'XXBTZUSD'
)

kraken_btcusdt = load_or_fetch(
    'kraken_btcusdt',
    fetch_kraken_ohlcv, 'XBTUSDT'
)

kraken_btcusdc = load_or_fetch(
    'kraken_btcusdc',
    fetch_kraken_ohlcv, 'XBTUSDC'
)

kraken_btceur = load_or_fetch(
    'kraken_btceur',
    fetch_kraken_ohlcv, 'XXBTZEUR'
)

# ----- Summary -----
print("\n" + "=" * 60)
print("DATA SUMMARY")
print("=" * 60)
datasets = {
    'Binance BTC/USDT': binance_btcusdt,
    'Binance BTC/USDC': binance_btcusdc,
    'Binance BTC/EUR': binance_btceur,
    'Coinbase BTC/USD': coinbase_btcusd,
    'Coinbase BTC/USDT': coinbase_btcusdt,
    'Coinbase BTC/EUR': coinbase_btceur,
    'Kraken BTC/USD': kraken_btcusd,
    'Kraken BTC/USDT': kraken_btcusdt,
    'Kraken BTC/USDC': kraken_btcusdc,
    'Kraken BTC/EUR': kraken_btceur,
}
for name, df in datasets.items():
    if df.empty:
        print(f"  {name:30s} -> NO DATA")
    else:
        print(f"  {name:30s} -> {len(df):>8,} rows  |  {df.index.min()} to {df.index.max()}")



# ============================================================
# Cell 6: Build a unified close-price panel
# ============================================================

# Create a common minute-level index
full_index = pd.date_range(START_DT, END_DT, freq='1min', tz='UTC')

def extract_close(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    """Extract close prices reindexed to common grid."""
    if df.empty:
        return pd.Series(np.nan, index=idx, dtype=float)
    s = df['close'].reindex(idx)
    return s

def extract_volume(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    if df.empty:
        return pd.Series(0.0, index=idx, dtype=float)
    s = df['volume'].reindex(idx, fill_value=0.0)
    return s

def extract_trades(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    if df.empty or 'trades' not in df.columns:
        return pd.Series(0, index=idx, dtype=int)
    s = df['trades'].reindex(idx, fill_value=0)
    return s

def extract_spread_proxy(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    """Use (high - low) as an intra-candle range proxy."""
    if df.empty:
        return pd.Series(np.nan, index=idx, dtype=float)
    s = (df['high'] - df['low']).reindex(idx)
    return s

# Close prices panel
closes = pd.DataFrame({
    'binance_btcusdt': extract_close(binance_btcusdt, full_index),
    'binance_btcusdc': extract_close(binance_btcusdc, full_index),
    'binance_btceur': extract_close(binance_btceur, full_index),
    'coinbase_btcusd': extract_close(coinbase_btcusd, full_index),
    'coinbase_btcusdt': extract_close(coinbase_btcusdt, full_index),
    'coinbase_btceur': extract_close(coinbase_btceur, full_index),
    'kraken_btcusd': extract_close(kraken_btcusd, full_index),
    'kraken_btcusdt': extract_close(kraken_btcusdt, full_index),
    'kraken_btcusdc': extract_close(kraken_btcusdc, full_index),
    'kraken_btceur': extract_close(kraken_btceur, full_index),
}, index=full_index)

# Volume panel
volumes = pd.DataFrame({
    'binance_btcusdt': extract_volume(binance_btcusdt, full_index),
    'binance_btcusdc': extract_volume(binance_btcusdc, full_index),
    'binance_btceur': extract_volume(binance_btceur, full_index),
    'coinbase_btcusd': extract_volume(coinbase_btcusd, full_index),
    'coinbase_btcusdt': extract_volume(coinbase_btcusdt, full_index),
    'coinbase_btceur': extract_volume(coinbase_btceur, full_index),
    'kraken_btcusd': extract_volume(kraken_btcusd, full_index),
    'kraken_btcusdt': extract_volume(kraken_btcusdt, full_index),
    'kraken_btcusdc': extract_volume(kraken_btcusdc, full_index),
    'kraken_btceur': extract_volume(kraken_btceur, full_index),
}, index=full_index)

# Range proxy panel
spreads = pd.DataFrame({
    'binance_btcusdt': extract_spread_proxy(binance_btcusdt, full_index),
    'binance_btcusdc': extract_spread_proxy(binance_btcusdc, full_index),
    'binance_btceur': extract_spread_proxy(binance_btceur, full_index),
    'coinbase_btcusd': extract_spread_proxy(coinbase_btcusd, full_index),
    'coinbase_btcusdt': extract_spread_proxy(coinbase_btcusdt, full_index),
    'coinbase_btceur': extract_spread_proxy(coinbase_btceur, full_index),
    'kraken_btcusd': extract_spread_proxy(kraken_btcusd, full_index),
    'kraken_btcusdt': extract_spread_proxy(kraken_btcusdt, full_index),
    'kraken_btcusdc': extract_spread_proxy(kraken_btcusdc, full_index),
    'kraken_btceur': extract_spread_proxy(kraken_btceur, full_index),
}, index=full_index)

# Trade count panel (Binance and Kraken provide trade counts)
trade_counts = pd.DataFrame({
    'binance_btcusdt': extract_trades(binance_btcusdt, full_index),
    'binance_btcusdc': extract_trades(binance_btcusdc, full_index),
    'binance_btceur': extract_trades(binance_btceur, full_index),
    'kraken_btcusd': extract_trades(kraken_btcusd, full_index),
    'kraken_btcusdt': extract_trades(kraken_btcusdt, full_index),
    'kraken_btcusdc': extract_trades(kraken_btcusdc, full_index),
    'kraken_btceur': extract_trades(kraken_btceur, full_index),
}, index=full_index)

# Forward-fill small gaps (max 5 minutes) for close prices
closes = closes.ffill(limit=5)

# Data coverage report
print("Close price coverage (% non-NaN):")
print((closes.notna().mean() * 100).round(2).to_string())
print(f"\nTotal minutes in window: {len(full_index):,}")

# ============================================================
# Cell 7: Compute implied stablecoin exchange rates from BTC triangulation
# ============================================================

def aligned_ratio(numer: pd.Series, denom: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Compute numer/denom on overlapping timestamps only, then reindex."""
    pair = pd.concat([numer, denom], axis=1).dropna()
    if pair.empty:
        return pd.Series(np.nan, index=idx, dtype=float)
    ratio = pair.iloc[:, 0] / pair.iloc[:, 1]
    return ratio.reindex(idx)

# --- Coinbase (has btcusd, btcusdt, btceur — no btcusdc) ---

# Implied USDT/USD = BTC_USD / BTC_USDT (from Coinbase)
closes['implied_usdt_usd_coinbase'] = aligned_ratio(
    closes['coinbase_btcusd'],
    closes['coinbase_btcusdt'],
    closes.index
)

# Implied EUR/USD = BTC_USD / BTC_EUR (from Coinbase)
closes['implied_eur_usd_coinbase'] = aligned_ratio(
    closes['coinbase_btcusd'],
    closes['coinbase_btceur'],
    closes.index
)

# --- Binance (has btcusdt, btcusdc, btceur — no btcusd) ---

# Implied USDT/USDC = BTC_USDC / BTC_USDT (from Binance)
closes['implied_usdt_usdc_binance'] = aligned_ratio(
    closes['binance_btcusdc'],
    closes['binance_btcusdt'],
    closes.index
)

# Implied EUR/USDT = BTC_USDT / BTC_EUR (from Binance)
closes['implied_eur_usdt_binance'] = aligned_ratio(
    closes['binance_btcusdt'],
    closes['binance_btceur'],
    closes.index
)

# --- Kraken (has btcusd, btcusdt, btcusdc, btceur) ---

# Implied USDT/USD = BTC_USD / BTC_USDT (from Kraken)
closes['implied_usdt_usd_kraken'] = aligned_ratio(
    closes['kraken_btcusd'],
    closes['kraken_btcusdt'],
    closes.index
)

# Implied USDC/USD = BTC_USD / BTC_USDC (from Kraken)
closes['implied_usdc_usd_kraken'] = aligned_ratio(
    closes['kraken_btcusd'],
    closes['kraken_btcusdc'],
    closes.index
)

# Implied USDT/USDC = BTC_USDC / BTC_USDT (from Kraken)
closes['implied_usdt_usdc_kraken'] = aligned_ratio(
    closes['kraken_btcusdc'],
    closes['kraken_btcusdt'],
    closes.index
)

# Implied EUR/USD = BTC_USD / BTC_EUR (from Kraken)
closes['implied_eur_usd_kraken'] = aligned_ratio(
    closes['kraken_btcusd'],
    closes['kraken_btceur'],
    closes.index
)

# Quick sanity check
print("Implied stablecoin rates (mean over full window):")
for col in [
    'implied_usdt_usd_coinbase',
    'implied_eur_usd_coinbase',
    'implied_usdt_usdc_binance',
    'implied_eur_usdt_binance',
    'implied_usdt_usd_kraken',
    'implied_usdc_usd_kraken',
    'implied_usdt_usdc_kraken',
    'implied_eur_usd_kraken',
]:
    if closes[col].notna().any():
        print(f"  {col}: {closes[col].mean():.6f}")

# ============================================================
# Cell 8: Summary statistics table
# ============================================================

btc_cols = [
    'binance_btcusdt', 'binance_btcusdc', 'binance_btceur',
    'coinbase_btcusd', 'coinbase_btcusdt', 'coinbase_btceur',
    'kraken_btcusd', 'kraken_btcusdt', 'kraken_btcusdc', 'kraken_btceur',
]
nice_names = [
    'Binance BTC/USDT', 'Binance BTC/USDC', 'Binance BTC/EUR',
    'Coinbase BTC/USD', 'Coinbase BTC/USDT', 'Coinbase BTC/EUR',
    'Kraken BTC/USD', 'Kraken BTC/USDT', 'Kraken BTC/USDC', 'Kraken BTC/EUR',
]

# 1-min returns
returns = closes[btc_cols].pct_change()

stats = pd.DataFrame({
    'Mean Price': closes[btc_cols].mean(),
    'Std Price': closes[btc_cols].std(),
    'Min Price': closes[btc_cols].min(),
    'Max Price': closes[btc_cols].max(),
    'Mean 1m Return (bps)': returns[btc_cols].mean() * 1e4,
    'Std 1m Return (bps)': returns[btc_cols].std() * 1e4,
    'Mean Volume (BTC)': volumes[btc_cols].mean(),
    'Total Volume (BTC)': volumes[btc_cols].sum(),
    'Mean Spread Proxy ($)': spreads[btc_cols].mean(),
    'Coverage (%)': closes[btc_cols].notna().mean() * 100,
})
stats.index = nice_names
stats.round(4)



# ============================================================
# Cell 9: Cross-currency basis â€” same exchange (Coinbase)
# ============================================================

# Basis: BTC/USD vs BTC/USDT on Coinbase
basis_usdt_coinbase = (closes['implied_usdt_usd_coinbase'] - 1) * 1e4
basis_usdt_coinbase.name = 'USDT basis (bps) [Coinbase]'

# Basis: BTC/USD vs BTC/USDC on Coinbase
basis_usdc_kraken = (closes['implied_usdc_usd_kraken'] - 1) * 1e4
basis_usdc_kraken.name = 'USDC basis (bps) [Kraken]'

# Basis: BTC/USDT vs BTC/USDC on Binance
basis_usdt_usdc_binance = (closes['implied_usdt_usdc_binance'] - 1) * 1e4
basis_usdt_usdc_binance.name = 'USDT/USDC basis (bps) [Binance]'

# Basis: BTC/USDT vs BTC/USDC on Coinbase
basis_usdt_usdc_kraken = (closes['implied_usdt_usdc_kraken'] - 1) * 1e4
basis_usdt_usdc_kraken.name = 'USDT/USDC basis (bps) [Kraken]'

# Cross-exchange basis: Binance BTC/USDT vs Coinbase BTC/USDT
basis_cross_exchange = (closes['binance_btcusdt'] / closes['coinbase_btcusdt'] - 1) * 1e4
basis_cross_exchange.name = 'Binance vs Kraken BTC/USDT (bps)'

fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)

# SVB event shading
svb_start = pd.Timestamp('2023-03-10', tz='UTC')
svb_end = pd.Timestamp('2023-03-13', tz='UTC')

for ax in axes:
    ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis Window')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

axes[0].plot(basis_usdt_coinbase.index, basis_usdt_coinbase, linewidth=0.4, alpha=0.8, color='blue')
axes[0].set_title('BTC/USD vs BTC/USDT Basis â€” Coinbase (same exchange)')
axes[0].set_ylabel('Basis (bps)')

axes[1].plot(basis_usdc_kraken.index, basis_usdc_kraken, linewidth=0.4, alpha=0.8, color='green')
axes[1].set_title('BTC/USD vs BTC/USDC Basis â€” Coinbase (same exchange)')
axes[1].set_ylabel('Basis (bps)')

axes[2].plot(basis_usdt_usdc_binance.index, basis_usdt_usdc_binance, linewidth=0.4, alpha=0.8, color='purple', label='Binance')
axes[2].plot(basis_usdt_usdc_kraken.index, basis_usdt_usdc_kraken, linewidth=0.4, alpha=0.8, color='teal', label='Kraken')
axes[2].set_title('BTC/USDC vs BTC/USDT Basis â€” Binance vs Kraken')
axes[2].set_ylabel('Basis (bps)')

axes[3].plot(basis_cross_exchange.index, basis_cross_exchange, linewidth=0.4, alpha=0.8, color='orange')
axes[3].set_title('BTC/USDT: Binance vs Kraken â€” Cross-Exchange Basis')
axes[3].set_ylabel('Basis (bps)')

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('fig_cross_currency_basis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: fig_cross_currency_basis.png")

# ============================================================
# Cell 10: Basis statistics by regime (pre-crisis, crisis, post-crisis)
# ============================================================

regimes = {
    'Pre-SVB (Mar 1-9)': (pd.Timestamp('2023-03-01', tz='UTC'), pd.Timestamp('2023-03-10', tz='UTC')),
    'SVB Crisis (Mar 10-13)': (pd.Timestamp('2023-03-10', tz='UTC'), pd.Timestamp('2023-03-14', tz='UTC')),
    'Post-SVB (Mar 14-21)': (pd.Timestamp('2023-03-14', tz='UTC'), pd.Timestamp('2023-03-22', tz='UTC')),
}

basis_series = {
    'USDT basis (Coinbase)': basis_usdt_coinbase,
    'USDC basis (Kraken)': basis_usdc_kraken,
    'USDT/USDC basis (Binance)': basis_usdt_usdc_binance,
    'USDT/USDC basis (Kraken)': basis_usdt_usdc_kraken,
    'Cross-exch BTC/USDT': basis_cross_exchange,
}

regime_stats = []
for regime_name, (t0, t1) in regimes.items():
    for basis_name, bs in basis_series.items():
        subset = bs[(bs.index >= t0) & (bs.index < t1)].dropna()
        regime_stats.append({
            'Regime': regime_name,
            'Basis': basis_name,
            'Mean (bps)': subset.mean(),
            'Std (bps)': subset.std(),
            'Min (bps)': subset.min(),
            'Max (bps)': subset.max(),
            'Median (bps)': subset.median(),
            'Skewness': subset.skew(),
            'Kurtosis': subset.kurtosis(),
            'N obs': len(subset),
        })

regime_df = pd.DataFrame(regime_stats)
regime_df.set_index(['Regime', 'Basis'], inplace=True)
regime_df.round(2)

# ============================================================
# Cell 11: Stablecoin peg prices over time
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Implied stablecoin rates from Kraken BTC triangulation
ax = axes[0]
ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis')
ax.axhline(1.0, color='black', linewidth=0.5, linestyle='--')
ax.plot(closes.index, closes['implied_usdt_usd_kraken'], linewidth=0.5, label='Implied USDT/USD (Kraken)', color='blue')
ax.plot(closes.index, closes['implied_usdc_usd_kraken'], linewidth=0.5, label='Implied USDC/USD (Kraken)', color='green')
ax.set_title('Implied Stablecoin Rates from BTC Triangulation (Kraken)')
ax.set_ylabel('Implied Rate')
ax.legend()
ax.set_ylim(0.85, 1.05)

# Implied stablecoin rates from BTC triangulation (Coinbase)
ax = axes[1]
ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis')
ax.axhline(1.0, color='black', linewidth=0.5, linestyle='--')
ax.plot(closes.index, closes['implied_usdt_usd_coinbase'], linewidth=0.5, label='Implied USDT/USD (Coinbase)', color='blue')
ax.set_title('Implied Stablecoin Rates from BTC Triangulation (Coinbase)')
ax.set_ylabel('Implied Rate')
ax.legend()
ax.set_ylim(0.85, 1.05)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.tight_layout()
plt.savefig('fig_stablecoin_peg.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: fig_stablecoin_peg.png")

# ============================================================
# Cell 12: Zoomed-in view of the USDC de-peg event (March 10-13)
# ============================================================

zoom_start = pd.Timestamp('2023-03-10 00:00', tz='UTC')
zoom_end = pd.Timestamp('2023-03-14 00:00', tz='UTC')
mask = (closes.index >= zoom_start) & (closes.index < zoom_end)

fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

# Panel 1: BTC prices across pairs
ax = axes[0]
for col, label, color in [
    ('coinbase_btcusd', 'Coinbase BTC/USD', 'black'),
    ('coinbase_btcusdt', 'Coinbase BTC/USDT', 'blue'),
    ('binance_btcusdt', 'Binance BTC/USDT', 'red'),
    ('binance_btcusdc', 'Binance BTC/USDC', 'purple'),
    ('kraken_btcusd', 'Kraken BTC/USD', 'darkorange'),
]:
    ax.plot(closes.loc[mask, col], linewidth=0.6, label=label, color=color)
ax.set_title('BTC Prices Across Pairs — SVB Crisis Window')
ax.set_ylabel('Price')
ax.legend(fontsize=9)

# Panel 2: Stablecoin rates
ax = axes[1]
ax.axhline(1.0, color='black', linewidth=0.5, linestyle='--')
ax.plot(closes.loc[mask, 'implied_usdc_usd_kraken'], linewidth=0.6, label='Implied USDC/USD (Kraken)', color='green')
ax.plot(closes.loc[mask, 'implied_usdt_usd_kraken'], linewidth=0.6, label='Implied USDT/USD (Kraken)', color='blue')
ax.plot(closes.loc[mask, 'implied_usdt_usd_coinbase'], linewidth=0.6, label='Implied USDT/USD (Coinbase)', color='deepskyblue', linestyle='--')
ax.set_title('Stablecoin Peg During SVB Crisis')
ax.set_ylabel('Rate')
ax.legend(fontsize=9)
ax.set_ylim(0.85, 1.06)

# Panel 3: USDC basis (BTC/USD vs BTC/USDC) on Kraken + USDT basis on Coinbase
ax = axes[2]
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.plot(basis_usdc_kraken[mask], linewidth=0.6, color='green', label='USDC basis (Kraken)')
ax.plot(basis_usdt_coinbase[mask], linewidth=0.6, color='blue', label='USDT basis (Coinbase)')
ax.set_title('Cross-Currency Basis During SVB Crisis')
ax.set_ylabel('Basis (bps)')
ax.legend(fontsize=9)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

plt.tight_layout()
plt.savefig('fig_svb_crisis_zoom.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: fig_svb_crisis_zoom.png")

# ============================================================
# Cell 13: Premium/Discount analysis â€” USDT vs USDC across exchanges
# ============================================================

# USDC premium/discount on Coinbase = (BTC_USDC / BTC_USD - 1) * 10000
# When USDC < $1, BTC/USDC > BTC/USD â†’ positive premium in USDC terms
usdt_premium_coinbase = (closes['coinbase_btcusdt'] / closes['coinbase_btcusd'] - 1) * 1e4

# On Binance: BTC/USDC vs BTC/USDT
usdc_premium_vs_usdt_binance = (closes['binance_btcusdc'] / closes['binance_btcusdt'] - 1) * 1e4

fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

for ax in axes:
    ax.axvspan(svb_start, svb_end, alpha=0.15, color='red')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

axes[0].plot(usdc_premium_coinbase, linewidth=0.4, color='green')
axes[0].set_title('USDC Premium/Discount vs USD (Coinbase BTC pairs) â€” positive = USDC weak')
axes[0].set_ylabel('bps')

axes[1].plot(usdt_premium_coinbase, linewidth=0.4, color='blue')
axes[1].set_title('USDT Premium/Discount vs USD (Coinbase BTC pairs) â€” positive = USDT weak')
axes[1].set_ylabel('bps')

axes[2].plot(usdc_premium_vs_usdt_binance, linewidth=0.4, color='purple')
axes[2].set_title('USDC Premium/Discount vs USDT (Binance BTC pairs) â€” positive = USDC weak vs USDT')
axes[2].set_ylabel('bps')

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.tight_layout()
plt.savefig('fig_stablecoin_premium_discount.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: fig_stablecoin_premium_discount.png")
# ============================================================
# Cell 14: USDC de-peg severity â€” minute-level max deviation
# ============================================================

# Focus on the implied USDC/USD rate from Kraken triangulation
usdc_rate = closes['implied_usdc_usd_kraken'].dropna()
usdc_deviation = (usdc_rate - 1.0) * 100  # in percentage points

crisis_mask = (usdc_deviation.index >= svb_start) & (usdc_deviation.index < svb_end)
crisis_dev = usdc_deviation[crisis_mask]

print("USDC/USD Implied Rate During SVB Crisis (Mar 10-13):")
print(f"  Min rate:  {usdc_rate[crisis_mask].min():.4f} (max discount: {crisis_dev.min():.2f}%)")
print(f"  Max rate:  {usdc_rate[crisis_mask].max():.4f}")
print(f"  Mean rate: {usdc_rate[crisis_mask].mean():.4f}")
print(f"  Time of min: {usdc_rate[crisis_mask].idxmin()}")

# Distribution of deviations during crisis vs normal
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

pre_crisis_dev = usdc_deviation[(usdc_deviation.index >= START_DT) & (usdc_deviation.index < svb_start)].dropna()
crisis_dev_clean = crisis_dev.dropna()

axes[0].hist(pre_crisis_dev, bins=100, color='green', alpha=0.7, edgecolor='black', linewidth=0.3)
axes[0].set_title('USDC Deviation from $1 â€” Pre-Crisis (Mar 1-9)')
axes[0].set_xlabel('Deviation (%)')
axes[0].set_ylabel('Frequency (1-min candles)')

axes[1].hist(crisis_dev_clean, bins=100, color='red', alpha=0.7, edgecolor='black', linewidth=0.3)
axes[1].set_title('USDC Deviation from $1 â€” Crisis (Mar 10-13)')
axes[1].set_xlabel('Deviation (%)')
axes[1].set_ylabel('Frequency (1-min candles)')

plt.tight_layout()
plt.savefig('fig_usdc_deviation_hist.png', dpi=150, bbox_inches='tight')
plt.show()
# ============================================================
# Cell 15: Volume comparison across pairs
# ============================================================

# Hourly volume aggregation for cleaner visualization
hourly_vol = volumes[btc_cols].resample('1h').sum()

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

for ax in axes:
    ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis')

# Coinbase pairs
ax = axes[0]
for col, label, color in [
    ('coinbase_btcusd', 'BTC/USD', 'black'),
    ('coinbase_btcusdt', 'BTC/USDT', 'blue'),
]:
    ax.plot(hourly_vol[col], linewidth=0.8, label=label, color=color)
ax.set_title('Hourly BTC Volume â€” Coinbase')
ax.set_ylabel('BTC Volume')
ax.legend()

# Binance pairs
ax = axes[1]
for col, label, color in [
    ('binance_btcusdt', 'BTC/USDT', 'blue'),
    ('binance_btcusdc', 'BTC/USDC', 'green'),
]:
    ax.plot(hourly_vol[col], linewidth=0.8, label=label, color=color)
ax.set_title('Hourly BTC Volume â€” Binance')
ax.set_ylabel('BTC Volume')
ax.legend()

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.tight_layout()
plt.savefig('fig_volume_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
# ============================================================
# Cell 16: Range proxy analysis
# ============================================================

# Intra-minute range proxy = (high - low) / close  (in bps)
rel_spreads = pd.DataFrame()
for col in btc_cols:
    rel_spreads[col] = (spreads[col] / closes[col]) * 1e4  # bps

# Hourly mean intra-minute range proxy
hourly_spread = rel_spreads.resample('1h').mean()

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

for ax in axes:
    ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis')

ax = axes[0]
for col, label, color in [
    ('coinbase_btcusd', 'BTC/USD', 'black'),
    ('coinbase_btcusdt', 'BTC/USDT', 'blue'),
]:
    ax.plot(hourly_spread[col], linewidth=0.8, label=label, color=color)
ax.set_title('Hourly Mean Intra-minute Range (High-Low/Close) â€” Coinbase')
ax.set_ylabel('Intra-minute Range (bps)')
ax.legend()

ax = axes[1]
for col, label, color in [
    ('binance_btcusdt', 'BTC/USDT', 'blue'),
    ('binance_btcusdc', 'BTC/USDC', 'green'),
]:
    ax.plot(hourly_spread[col], linewidth=0.8, label=label, color=color)
ax.set_title('Hourly Mean Intra-minute Range (High-Low/Close) â€” Binance')
ax.set_ylabel('Intra-minute Range (bps)')
ax.legend()

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.tight_layout()
plt.savefig('fig_spread_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
# ============================================================
# Cell 17: Realized volatility â€” rolling 1-hour (60 x 1-min returns)
# ============================================================

# Annualized realized vol from 1-min returns (sqrt(525600) ~ 725)
rolling_vol = returns[btc_cols].rolling(60).std() * np.sqrt(525600) * 100  # annualized %

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

for ax in axes:
    ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis')

ax = axes[0]
for col, label, color in [
    ('coinbase_btcusd', 'BTC/USD', 'black'),
    ('coinbase_btcusdt', 'BTC/USDT', 'blue'),
]:
    ax.plot(rolling_vol[col], linewidth=0.5, label=label, color=color, alpha=0.8)
ax.set_title('Rolling 1-Hour Realized Volatility (Annualized) â€” Coinbase')
ax.set_ylabel('Vol (%)')
ax.legend()

ax = axes[1]
for col, label, color in [
    ('binance_btcusdt', 'BTC/USDT', 'blue'),
    ('binance_btcusdc', 'BTC/USDC', 'green'),
]:
    ax.plot(rolling_vol[col], linewidth=0.5, label=label, color=color, alpha=0.8)
ax.set_title('Rolling 1-Hour Realized Volatility (Annualized) â€” Binance')
ax.set_ylabel('Vol (%)')
ax.legend()

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.tight_layout()
plt.savefig('fig_realized_volatility.png', dpi=150, bbox_inches='tight')
plt.show()
# ============================================================
# Cell 18: Liquidity regime comparison table
# ============================================================

liq_stats = []
for regime_name, (t0, t1) in regimes.items():
    rmask = (volumes.index >= t0) & (volumes.index < t1)
    for col, nice in zip(btc_cols, nice_names):
        vol_sub = volumes.loc[rmask, col]
        spread_sub = rel_spreads.loc[rmask, col].dropna()
        ret_sub = returns.loc[rmask, col].dropna()
        liq_stats.append({
            'Regime': regime_name,
            'Pair': nice,
            'Mean Volume (BTC/min)': vol_sub.mean(),
            'Total Volume (BTC)': vol_sub.sum(),
            'Mean Range Proxy (bps)': spread_sub.mean(),
            'Median Range Proxy (bps)': spread_sub.median(),
            'Realized Vol (ann %)': ret_sub.std() * np.sqrt(525600) * 100,
        })

liq_df = pd.DataFrame(liq_stats)
liq_df.set_index(['Regime', 'Pair'], inplace=True)
liq_df.round(4)
# ============================================================
# Cell 19: Volume share by quote currency over time
# ============================================================

# Aggregate volume by quote currency (across exchanges)
vol_by_quote = pd.DataFrame({
    'USD': volumes['coinbase_btcusd'],
    'USDT': volumes[['binance_btcusdt', 'coinbase_btcusdt']].sum(axis=1),
})

# Daily volume shares
daily_vol = vol_by_quote.resample('1D').sum()
daily_share = daily_vol.div(daily_vol.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Stacked area chart of volume shares
ax = axes[0]
ax.stackplot(daily_share.index, daily_share['USD'], daily_share['USDT'], daily_share['USDC'],
             labels=['USD', 'USDT', 'USDC'], colors=['black', 'blue', 'green'], alpha=0.7)
ax.axvspan(svb_start, svb_end, alpha=0.2, color='red')
ax.set_title('Daily Volume Share by Quote Currency')
ax.set_ylabel('Share (%)')
ax.legend(loc='upper right')

# Absolute daily volume
ax = axes[1]
ax.bar(daily_vol.index, daily_vol['USDT'], width=0.8, label='USDT', color='blue', alpha=0.7)
ax.bar(daily_vol.index, daily_vol['USD'], width=0.8, bottom=daily_vol['USDT'], label='USD', color='black', alpha=0.7)
ax.bar(daily_vol.index, daily_vol['USDC'], width=0.8, bottom=daily_vol['USDT'] + daily_vol['USD'], label='USDC', color='green', alpha=0.7)
ax.axvspan(svb_start, svb_end, alpha=0.2, color='red')
ax.set_title('Daily BTC Volume by Quote Currency')
ax.set_ylabel('BTC Volume')
ax.legend()

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.tight_layout()
plt.savefig('fig_volume_share.png', dpi=150, bbox_inches='tight')
plt.show()
# ============================================================
# Cell 20: Arbitrage after transaction costs
# ============================================================

# Typical fee assumptions (in bps, round-trip)
FEE_BINANCE_RT = 20   # 10bps * 2 (taker-taker round trip)
FEE_COINBASE_RT = 120  # 60bps * 2 (taker fee on Coinbase Pro was ~0.5-0.6% in March 2023)
FEE_CROSS_EXCHANGE_RT = 70  # ~10bps Binance + 60bps Coinbase (one leg each)

# Same-exchange arb: BTC/USD vs BTC/USDT on Coinbase
arb_usdt_coinbase = basis_usdt_coinbase.abs() - FEE_COINBASE_RT
arb_usdc_kraken = basis_usdc_kraken.abs() - FEE_COINBASE_RT

# Cross-exchange arb: Binance BTC/USDT vs Coinbase BTC/USDT
arb_cross = basis_cross_exchange.abs() - FEE_CROSS_EXCHANGE_RT

# Binance internal: BTC/USDT vs BTC/USDC
arb_binance_internal = basis_usdt_usdc_binance.abs() - FEE_BINANCE_RT

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

arb_data = [
    (arb_usdt_coinbase, 'USDT/USD arb (Coinbase)', axes[0, 0]),
    (arb_usdc_kraken, 'USDC/USD arb (Coinbase)', axes[0, 1]),
    (arb_cross, 'Cross-Exchange BTC/USDT arb', axes[1, 0]),
    (arb_binance_internal, 'USDT/USDC arb (Binance)', axes[1, 1]),
]

for arb, title, ax in arb_data:
    ax.axvspan(svb_start, svb_end, alpha=0.15, color='red')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    profitable = arb[arb > 0]
    pct_profitable = (arb > 0).mean() * 100
    ax.plot(arb, linewidth=0.3, alpha=0.6)
    ax.set_title(f'{title}\n({pct_profitable:.1f}% of minutes show profit after fees)')
    ax.set_ylabel('Net P&L (bps)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.tight_layout()
plt.savefig('fig_arbitrage_after_fees.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nArbitrage profitability summary (% of minutes with net positive P&L):")
for arb, title, _ in arb_data:
    a = arb.dropna()
    print(f"  {title:40s}: {(a > 0).mean()*100:6.2f}% profitable  |  Mean net P&L when profitable: {a[a>0].mean():.1f} bps")
# ============================================================
# Cell 21: Return correlations across pairs and regimes
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (regime_name, (t0, t1)) in enumerate(regimes.items()):
    rmask = (returns.index >= t0) & (returns.index < t1)
    corr = returns.loc[rmask, btc_cols].corr()
    corr.index = nice_names
    corr.columns = nice_names
    
    ax = axes[idx]
    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(nice_names)))
    ax.set_yticks(range(len(nice_names)))
    ax.set_xticklabels([n.replace('Coinbase ', 'CB ').replace('Binance ', 'Bin ') for n in nice_names], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([n.replace('Coinbase ', 'CB ').replace('Binance ', 'Bin ') for n in nice_names], fontsize=8)
    for i in range(len(nice_names)):
        for j in range(len(nice_names)):
            ax.text(j, i, f'{corr.values[i,j]:.3f}', ha='center', va='center', fontsize=7)
    ax.set_title(regime_name, fontsize=11)

plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Correlation')
plt.suptitle('1-Minute Return Correlations by Regime', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fig_correlations.png', dpi=150, bbox_inches='tight')
plt.show()
# ============================================================
# Cell 22: Lead-lag analysis using cross-correlation
# ============================================================

def cross_corr(x: pd.Series, y: pd.Series, max_lag: int = 10) -> pd.Series:
    """
    Compute cross-correlation at different lags (in minutes).
    Positive lag => x leads y. Negative lag => y leads x.
    """
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x_shift = x.shift(lag)
            pair = pd.concat([x_shift, y], axis=1).dropna()
        else:
            y_shift = y.shift(-lag)
            pair = pd.concat([x, y_shift], axis=1).dropna()

        if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
            results[lag] = np.nan
        else:
            results[lag] = pair.iloc[:, 0].corr(pair.iloc[:, 1])

    return pd.Series(results)

# Lead-lag: Binance BTC/USDT vs Coinbase BTC/USD
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

pairs_to_compare = [
    ('binance_btcusdt', 'coinbase_btcusd', 'Binance USDT vs Coinbase USD'),
    ('binance_btcusdt', 'coinbase_btcusdt', 'Binance USDT vs Coinbase USDT'),
]

for idx, (col1, col2, title) in enumerate(pairs_to_compare):
    ax = axes[idx]
    for regime_name, (t0, t1) in regimes.items():
        rmask = (returns.index >= t0) & (returns.index < t1)
        cc = cross_corr(returns.loc[rmask, col1], returns.loc[rmask, col2], max_lag=5)
        ax.plot(cc.index, cc.values, marker='o', markersize=3, label=regime_name)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Lag (minutes)')
    ax.set_ylabel('Cross-correlation')
    ax.legend(fontsize=8)

plt.suptitle('Lead-Lag Cross-Correlations of 1-min Returns', fontsize=13)
plt.tight_layout()
plt.savefig('fig_lead_lag.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Cell 23: Intraday patterns â€” volume, range proxy, and basis by hour of day
# ============================================================

# Add hour column
hour = volumes.index.hour

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Volume by hour
ax = axes[0, 0]
for col, label, color in [
    ('coinbase_btcusd', 'CB BTC/USD', 'black'),
    ('coinbase_btcusdt', 'CB BTC/USDT', 'blue'),
    ('binance_btcusdt', 'Bin BTC/USDT', 'red'),
    ('binance_btcusdc', 'Bin BTC/USDC', 'purple'),
]:
    hourly_mean = volumes[col].groupby(hour).mean()
    ax.plot(hourly_mean.index, hourly_mean.values, marker='o', markersize=3, label=label, color=color)
ax.set_title('Mean Volume by Hour (UTC)')
ax.set_xlabel('Hour (UTC)')
ax.set_ylabel('BTC/minute')
ax.legend(fontsize=8)

# Range proxy by hour
ax = axes[0, 1]
for col, label, color in [
    ('coinbase_btcusd', 'CB BTC/USD', 'black'),
    ('coinbase_btcusdt', 'CB BTC/USDT', 'blue'),
    ('binance_btcusdt', 'Bin BTC/USDT', 'red'),
    ('binance_btcusdc', 'Bin BTC/USDC', 'purple'),
]:
    hourly_mean = rel_spreads[col].groupby(hour).mean()
    ax.plot(hourly_mean.index, hourly_mean.values, marker='o', markersize=3, label=label, color=color)
ax.set_title('Mean Intra-minute Range Proxy by Hour (UTC)')
ax.set_xlabel('Hour (UTC)')
ax.set_ylabel('Intra-minute Range (bps)')
ax.legend(fontsize=8)

# USDT basis by hour
ax = axes[1, 0]
for regime_name, (t0, t1) in regimes.items():
    rmask = (basis_usdt_coinbase.index >= t0) & (basis_usdt_coinbase.index < t1)
    hourly_basis = basis_usdt_coinbase[rmask].groupby(basis_usdt_coinbase[rmask].index.hour).mean()
    ax.plot(hourly_basis.index, hourly_basis.values, marker='o', markersize=3, label=regime_name)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_title('USDT Basis by Hour (Coinbase)')
ax.set_xlabel('Hour (UTC)')
ax.set_ylabel('Basis (bps)')
ax.legend(fontsize=8)

# USDC basis by hour
ax = axes[1, 1]
for regime_name, (t0, t1) in regimes.items():
    rmask = (basis_usdc_kraken.index >= t0) & (basis_usdc_kraken.index < t1)
    hourly_basis = basis_usdc_kraken[rmask].groupby(basis_usdc_kraken[rmask].index.hour).mean()
    ax.plot(hourly_basis.index, hourly_basis.values, marker='o', markersize=3, label=regime_name)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_title('USDC Basis by Hour (Coinbase)')
ax.set_xlabel('Hour (UTC)')
ax.set_ylabel('Basis (bps)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('fig_intraday_patterns.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Cell 24: OU mean-reversion estimation
# ============================================================
from scipy import stats as sp_stats

def estimate_ou_halflife(series: pd.Series) -> dict:
    """
    Estimate the half-life of mean reversion via OLS on:
        delta_x(t) = alpha + beta * x(t-1) + epsilon
    Half-life = -ln(2) / beta
    """
    s = series.dropna()
    if len(s) < 100 or s.nunique() < 2:
        return {'half_life_min': np.nan, 'beta': np.nan, 'p_value': np.nan, 'n': len(s)}

    y = s.diff().iloc[1:].values
    x = s.iloc[:-1].values

    # Guard against constant x or y (linregress fails)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {'half_life_min': np.nan, 'beta': np.nan, 'p_value': np.nan, 'n': len(y)}

    # OLS regression
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)

    if slope >= 0:
        half_life = np.inf
    else:
        half_life = -np.log(2) / slope

    return {
        'half_life_min': half_life,
        'half_life_hours': half_life / 60 if np.isfinite(half_life) else np.inf,
        'beta': slope,
        'p_value': p_value,
        'mean_level': intercept / (-slope) if slope != 0 else np.nan,
        'n': len(y)
    }

print("Half-Life of Mean Reversion (Ornstein-Uhlenbeck)")
print("=" * 80)

hl_results = []
for basis_name, bs in basis_series.items():
    for regime_name, (t0, t1) in regimes.items():
        subset = bs[(bs.index >= t0) & (bs.index < t1)]
        result = estimate_ou_halflife(subset)
        result['Basis'] = basis_name
        result['Regime'] = regime_name
        hl_results.append(result)
    # Full period
    result = estimate_ou_halflife(bs)
    result['Basis'] = basis_name
    result['Regime'] = 'Full Period'
    hl_results.append(result)

hl_df = pd.DataFrame(hl_results)
hl_df = hl_df[['Basis', 'Regime', 'half_life_min', 'half_life_hours', 'beta', 'p_value', 'n']]
hl_df.set_index(['Basis', 'Regime'], inplace=True)
hl_df.round(4)

# ============================================================
# Cell 25: Granger causality tests
# ============================================================
from statsmodels.tsa.stattools import grangercausalitytests

def run_granger(x: pd.Series, y: pd.Series, maxlag: int = 5) -> dict:
    """
    Test if x Granger-causes y.
    Returns dict of {lag: p_value} for F-test.
    """
    common = x.dropna().index.intersection(y.dropna().index)
    data = pd.DataFrame({'y': y.loc[common], 'x': x.loc[common]}).dropna()
    if len(data) < 100:
        return {}
    try:
        result = grangercausalitytests(data[['y', 'x']], maxlag=maxlag, verbose=False)
        return {lag: result[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)}
    except Exception:
        return {}

# Test: Does Binance BTC/USDT Granger-cause Coinbase BTC/USD?
granger_pairs = [
    ('binance_btcusdt', 'coinbase_btcusd', 'Binance USDT â†’ Coinbase USD'),
    ('coinbase_btcusd', 'binance_btcusdt', 'Coinbase USD â†’ Binance USDT'),
    ('binance_btcusdt', 'coinbase_btcusdt', 'Binance USDT â†’ Coinbase USDT'),
    ('coinbase_btcusdt', 'binance_btcusdt', 'Coinbase USDT â†’ Binance USDT'),
]

print("Granger Causality p-values (lower = stronger evidence of causation)")
print("=" * 80)

granger_results = []
for col1, col2, label in granger_pairs:
    for regime_name, (t0, t1) in regimes.items():
        rmask = (returns.index >= t0) & (returns.index < t1)
        pvals = run_granger(returns.loc[rmask, col1], returns.loc[rmask, col2], maxlag=5)
        for lag, pv in pvals.items():
            granger_results.append({
                'Direction': label,
                'Regime': regime_name,
                'Lag (min)': lag,
                'p-value': pv,
                'Significant (5%)': '***' if pv < 0.01 else ('**' if pv < 0.05 else ('*' if pv < 0.1 else ''))
            })

granger_df = pd.DataFrame(granger_results)
# Show lag=1 results as summary
summary = granger_df[granger_df['Lag (min)'] == 1].pivot_table(
    index='Direction', columns='Regime', values='p-value'
)
print("\nGranger Causality p-values at lag=1 minute:")
summary.round(6)
# ============================================================
# Cell 26: Rolling basis statistics
# ============================================================

# Rolling 4-hour (240-min) statistics of the USDC basis
window = 240

fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

for ax in axes:
    ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis')

# Rolling mean basis
ax = axes[0]
ax.plot(basis_usdc_kraken.rolling(window).mean(), linewidth=0.8, color='green', label='USDC basis')
ax.plot(basis_usdt_coinbase.rolling(window).mean(), linewidth=0.8, color='blue', label='USDT basis')
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_title(f'Rolling {window}-min Mean Basis (Coinbase)')
ax.set_ylabel('Mean Basis (bps)')
ax.legend()

# Rolling std of basis
ax = axes[1]
ax.plot(basis_usdc_kraken.rolling(window).std(), linewidth=0.8, color='green', label='USDC basis')
ax.plot(basis_usdt_coinbase.rolling(window).std(), linewidth=0.8, color='blue', label='USDT basis')
ax.set_title(f'Rolling {window}-min Std of Basis (Coinbase)')
ax.set_ylabel('Std (bps)')
ax.legend()

# Rolling abs max basis
ax = axes[2]
ax.plot(basis_usdc_kraken.abs().rolling(window).max(), linewidth=0.8, color='green', label='USDC basis (abs)')
ax.plot(basis_usdt_coinbase.abs().rolling(window).max(), linewidth=0.8, color='blue', label='USDT basis (abs)')
ax.set_title(f'Rolling {window}-min Max |Basis| (Coinbase)')
ax.set_ylabel('|Basis| (bps)')
ax.legend()

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.tight_layout()
plt.savefig('fig_rolling_basis.png', dpi=150, bbox_inches='tight')
plt.show()
# ============================================================
# Cell 27: Engle-Granger cointegration test
# ============================================================
from statsmodels.tsa.stattools import coint, adfuller

print("Cointegration Tests (Engle-Granger)")
print("=" * 80)
print("H0: No cointegration (series are not cointegrated)")
print()

coint_pairs = [
    ('coinbase_btcusd', 'coinbase_btcusdt', 'Coinbase USD vs USDT'),
    ('binance_btcusdt', 'coinbase_btcusdt', 'Binance vs Coinbase USDT'),
    ('binance_btcusdt', 'binance_btcusdc', 'Binance USDT vs USDC'),
]

coint_results = []
for col1, col2, label in coint_pairs:
    for regime_name, (t0, t1) in regimes.items():
        rmask = (closes.index >= t0) & (closes.index < t1)
        s1 = closes.loc[rmask, col1].dropna()
        s2 = closes.loc[rmask, col2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 100:
            continue
        try:
            stat, pval, crit = coint(s1.loc[common], s2.loc[common])
            coint_results.append({
                'Pair': label,
                'Regime': regime_name,
                'Test Stat': stat,
                'p-value': pval,
                '1% Critical': crit[0],
                '5% Critical': crit[1],
                'Cointegrated (5%)': 'Yes' if pval < 0.05 else 'No'
            })
        except Exception as e:
            print(f"  Error on {label} / {regime_name}: {e}")

coint_df = pd.DataFrame(coint_results)
coint_df.set_index(['Pair', 'Regime'], inplace=True)
coint_df.round(4)
# ============================================================
# Cell 28: ADF test on basis series (stationarity)
# ============================================================

print("Augmented Dickey-Fuller Tests on Basis Series")
print("H0: Unit root (non-stationary)")
print("=" * 80)

adf_results = []
for basis_name, bs in basis_series.items():
    for regime_name, (t0, t1) in regimes.items():
        subset = bs[(bs.index >= t0) & (bs.index < t1)].dropna()
        if len(subset) < 100:
            continue
        try:
            stat, pval, usedlag, nobs, crit, icbest = adfuller(subset, maxlag=20, autolag='AIC')
            adf_results.append({
                'Basis': basis_name,
                'Regime': regime_name,
                'ADF Stat': stat,
                'p-value': pval,
                'Lags Used': usedlag,
                'Stationary (5%)': 'Yes' if pval < 0.05 else 'No'
            })
        except Exception:
            pass

adf_df = pd.DataFrame(adf_results)
adf_df.set_index(['Basis', 'Regime'], inplace=True)
adf_df.round(4)
# ============================================================
# Cell 29: Summary dashboard figure
# ============================================================

fig = plt.figure(figsize=(20, 24))
gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

# Panel 1: BTC price across all pairs
ax = fig.add_subplot(gs[0, :])
ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis')
for col, label, color in [
    ('coinbase_btcusd', 'Coinbase BTC/USD', 'black'),
    ('coinbase_btcusdt', 'Coinbase BTC/USDT', 'blue'),
    ('binance_btcusdt', 'Binance BTC/USDT', 'red'),
    ('binance_btcusdc', 'Binance BTC/USDC', 'purple'),
]:
    ax.plot(closes[col], linewidth=0.5, label=label, color=color, alpha=0.8)
ax.set_title('A) BTC Price Across Trading Pairs')
ax.set_ylabel('Price')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Panel 2: USDC peg
ax = fig.add_subplot(gs[1, 0])
ax.axvspan(svb_start, svb_end, alpha=0.15, color='red')
ax.axhline(1.0, color='black', linewidth=0.5, linestyle='--')
ax.plot(closes['implied_usdc_usd_kraken'], linewidth=0.5, color='green', label='USDC/USD (implied, Kraken)')
ax.plot(closes['implied_usdt_usd_coinbase'], linewidth=0.5, color='blue', label='USDT/USD (implied, Coinbase)')
ax.set_title('B) Implied Stablecoin Peg Rates')
ax.set_ylabel('Rate')
ax.set_ylim(0.85, 1.05)
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Panel 3: Cross-currency basis
ax = fig.add_subplot(gs[1, 1])
ax.axvspan(svb_start, svb_end, alpha=0.15, color='red')
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.plot(basis_usdc_kraken.rolling(60).mean(), linewidth=0.8, color='green', label='USDC basis (1h avg)')
ax.plot(basis_usdt_coinbase.rolling(60).mean(), linewidth=0.8, color='blue', label='USDT basis (1h avg)')
ax.set_title('C) Cross-Currency Basis (1h rolling mean)')
ax.set_ylabel('Basis (bps)')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Panel 4: Volume
ax = fig.add_subplot(gs[2, 0])
ax.axvspan(svb_start, svb_end, alpha=0.15, color='red')
hourly_vol_total = volumes[btc_cols].sum(axis=1).resample('1h').sum()
ax.bar(hourly_vol_total.index, hourly_vol_total.values, width=1/24, color='steelblue', alpha=0.7)
ax.set_title('D) Total Hourly BTC Volume (All Pairs)')
ax.set_ylabel('BTC')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Panel 5: Realized vol
ax = fig.add_subplot(gs[2, 1])
ax.axvspan(svb_start, svb_end, alpha=0.15, color='red')
for col, label, color in [
    ('coinbase_btcusd', 'CB BTC/USD', 'black'),
    ('coinbase_btcusdt', 'CB BTC/USDT', 'blue'),
    ('binance_btcusdt', 'Bin BTC/USDT', 'red'),
    ('binance_btcusdc', 'Bin BTC/USDC', 'purple'),
]:
    rv = rolling_vol[col].resample('1h').mean()
    ax.plot(rv, linewidth=0.8, label=label, color=color)
ax.set_title('E) Rolling 1h Realized Volatility (Ann. %)')
ax.set_ylabel('Vol (%)')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Panel 6: Intra-minute range proxy
ax = fig.add_subplot(gs[3, 0])
ax.axvspan(svb_start, svb_end, alpha=0.15, color='red')
for col, label, color in [
    ('coinbase_btcusd', 'CB BTC/USD', 'black'),
    ('coinbase_btcusdt', 'CB BTC/USDT', 'blue'),
    ('binance_btcusdt', 'Bin BTC/USDT', 'red'),
    ('binance_btcusdc', 'Bin BTC/USDC', 'purple'),
]:
    hs = rel_spreads[col].resample('1h').mean()
    ax.plot(hs, linewidth=0.8, label=label, color=color)
ax.set_title('F) Hourly Mean Intra-minute Range Proxy')
ax.set_ylabel('Intra-minute Range (bps)')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Panel 7: Cross-exchange basis
ax = fig.add_subplot(gs[3, 1])
ax.axvspan(svb_start, svb_end, alpha=0.15, color='red')
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.plot(basis_cross_exchange.rolling(60).mean(), linewidth=0.8, color='orange')
ax.set_title('G) Cross-Exchange BTC/USDT Basis (Binance vs Coinbase, 1h avg)')
ax.set_ylabel('Basis (bps)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.savefig('fig_summary_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("Summary dashboard saved: fig_summary_dashboard.png")

# ============================================================
# Cell 30: Final data export for reproducibility
# ============================================================

# Save key derived datasets
closes.to_parquet(os.path.join(DATA_DIR, 'closes_panel.parquet'))
volumes.to_parquet(os.path.join(DATA_DIR, 'volumes_panel.parquet'))
spreads.to_parquet(os.path.join(DATA_DIR, 'spreads_panel.parquet'))
returns.to_parquet(os.path.join(DATA_DIR, 'returns_panel.parquet'))

basis_export = pd.DataFrame({
    'usdt_basis_coinbase': basis_usdt_coinbase,
    'usdc_basis_kraken': basis_usdc_kraken,
    'usdt_usdc_basis_binance': basis_usdt_usdc_binance,
    'usdt_usdc_basis_kraken': basis_usdt_usdc_kraken,
    'cross_exchange_btcusdt': basis_cross_exchange,
})
basis_export.to_parquet(os.path.join(DATA_DIR, 'basis_panel.parquet'))

print("All derived data exported to", DATA_DIR)
print("\nFiles:")
for f in sorted(os.listdir(DATA_DIR)):
    size = os.path.getsize(os.path.join(DATA_DIR, f))
    print(f"  {f:40s}  {size/1024:.0f} KB")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
