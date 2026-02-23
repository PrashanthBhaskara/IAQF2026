import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import time
import os
from io import BytesIO
from zipfile import ZipFile

# Time window
START_DT = datetime(2023, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
END_DT   = datetime(2023, 3, 21, 23, 59, 0, tzinfo=timezone.utc)

START_MS = int(START_DT.timestamp() * 1000)
END_MS   = int(END_DT.timestamp() * 1000)

DATA_DIR = 'data_raw'
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Study window: {START_DT.isoformat()} to {END_DT.isoformat()}")

# ============================================================
# Binance Fetcher
# ============================================================
def fetch_binance_vision(symbol: str, interval: str = '1m', start_ms: int = START_MS, end_ms: int = END_MS) -> pd.DataFrame:
    start_dt_local = pd.Timestamp(start_ms, unit='ms', tz='UTC')
    end_dt_local = pd.Timestamp(end_ms, unit='ms', tz='UTC')
    all_dfs = []
    current = start_dt_local.to_period('M')
    end_period = end_dt_local.to_period('M')

    while current <= end_period:
        year_month = current.strftime('%Y-%m')
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year_month}.zip"
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
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = df[col].astype(float)
    df['trades'] = df['trades'].astype(int)
    df = df.set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df[(df.index >= pd.Timestamp(start_ms, unit='ms', tz='UTC')) &
            (df.index <= pd.Timestamp(end_ms, unit='ms', tz='UTC'))]
    return df

def fetch_binance_klines(symbol: str, interval: str = '1m', start_ms: int = START_MS, end_ms: int = END_MS, limit: int = 1000) -> pd.DataFrame:
    url = 'https://api.binance.com/api/v3/klines'
    fallback_url = 'https://api.binance.us/api/v3/klines'
    used_fallback = False
    all_data = []
    current_start = start_ms

    while current_start < end_ms:
        params = {'symbol': symbol, 'interval': interval, 'startTime': current_start, 'endTime': end_ms, 'limit': limit}
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 451 and not used_fallback:
            url = fallback_url
            used_fallback = True
            continue
        if resp.status_code == 429:
            time.sleep(60)
            continue
        if resp.status_code == 400 and used_fallback:
            return fetch_binance_vision(symbol, interval, start_ms, end_ms)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_data.extend(data)
        current_start = data[-1][6] + 1
        time.sleep(0.15)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['trades'] = df['trades'].astype(int)
    df = df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='first')]
    return df

# ============================================================
# Coinbase Fetcher
# ============================================================
def fetch_coinbase_candles(product_id: str, granularity: int = 60, start_dt: datetime = START_DT, end_dt: datetime = END_DT) -> pd.DataFrame:
    url = f'https://api.coinbase.com/api/v3/brokerage/market/products/{product_id}/candles'
    gran_str = 'ONE_MINUTE'
    all_data = []
    chunk_seconds = 300 * granularity
    current_start = start_dt

    while current_start < end_dt:
        current_end = min(current_start + timedelta(seconds=chunk_seconds), end_dt)
        params = {'granularity': gran_str, 'start': int(current_start.timestamp()), 'end': int(current_end.timestamp())}
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(5)
            continue
        if resp.status_code == 404:
            return pd.DataFrame()
        resp.raise_for_status()
        data = resp.json().get('candles', [])
        if data:
            all_data.extend(data)
        current_start = current_end
        time.sleep(0.2)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['start'].astype(int), unit='s', utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df.set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df

# ============================================================
# Kraken Fetcher (Trades to OHLCV)
# ============================================================
def fetch_kraken_ohlcv(pair: str, start_dt: datetime = START_DT, end_dt: datetime = END_DT) -> pd.DataFrame:
    url = 'https://api.kraken.com/0/public/Trades'
    since_ns = int(start_dt.timestamp() * 1e9)
    end_ts = end_dt.timestamp()
    all_trades = []
    call_count = 0

    while True:
        params = {'pair': pair, 'since': str(since_ns)}
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(10)
            continue
        resp.raise_for_status()
        data = resp.json()
        if data.get('error'):
            err = data['error'][0] if isinstance(data['error'], list) and len(data['error']) > 0 else str(data['error'])
            if 'Rate limit' in err or 'Too many requests' in err:
                print(f"  Kraken Rate Limit: {err}, sleeping 10s...")
                time.sleep(10)
                continue
            else:
                print(f"  Kraken API Error for {pair}: {err}")
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
        if call_count % 14 == 0:
            time.sleep(3)
        else:
            time.sleep(0.1)

    if not all_trades:
        return pd.DataFrame()

    df_trades = pd.DataFrame(all_trades, columns=['price', 'volume', 'time', 'side', 'type', 'misc', 'trade_id'])
    df_trades['price'] = df_trades['price'].astype(float)
    df_trades['volume'] = df_trades['volume'].astype(float)
    df_trades['timestamp'] = pd.to_datetime(df_trades['time'].astype(float), unit='s', utc=True)
    df_trades = df_trades[(df_trades['timestamp'] >= start_dt) & (df_trades['timestamp'] <= end_dt)]
    df_trades = df_trades.set_index('timestamp')
    ohlcv = df_trades['price'].resample('1min').ohlc()
    ohlcv['volume'] = df_trades['volume'].resample('1min').sum()
    ohlcv['trades'] = df_trades['price'].resample('1min').count()
    ohlcv = ohlcv.dropna(subset=['close'])
    return ohlcv

# ============================================================
# Fetch & Cache
# ============================================================
def cache_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.parquet")

def load_or_fetch(name: str, fetch_fn, *args, **kwargs) -> pd.DataFrame:
    path = cache_path(name)
    if os.path.exists(path):
        print(f"  Loading cached: {name}")
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
        return df

    sparsh_path = os.path.join('data_sparsh', f"{name}.parquet")
    if os.path.exists(sparsh_path):
        print(f"  Copying from data_sparsh: {name}")
        df = pd.read_parquet(sparsh_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
        df.to_parquet(path)
        return df

    print(f"  Fetching: {name} ...")
    df = fetch_fn(*args, **kwargs)
    if not df.empty:
        df.to_parquet(path)
        print(f"  Saved {len(df):,} rows -> {path}")
    return df

def fetch_all_data():
    pairs_conf = [
        # Binance BTC Pairs
        ('binance_btcusdt', fetch_binance_klines, 'BTCUSDT'),
        ('binance_btcusdc', fetch_binance_klines, 'BTCUSDC'),
        ('binance_btceur',  fetch_binance_klines, 'BTCEUR'),
        # Binance Stablecoin Pairs
        ('binance_usdcusdt', fetch_binance_klines, 'USDCUSDT'),

        # Coinbase BTC Pairs
        ('coinbase_btcusd', fetch_coinbase_candles, 'BTC-USD'),
        ('coinbase_btcusdt', fetch_coinbase_candles, 'BTC-USDT'),
        ('coinbase_btceur',  fetch_coinbase_candles, 'BTC-EUR'),
        # Coinbase Stablecoin Pairs
        ('coinbase_usdtusd', fetch_coinbase_candles, 'USDT-USD'),

        # Kraken BTC Pairs
        ('kraken_btcusd',  fetch_kraken_ohlcv, 'XXBTZUSD'),
        ('kraken_btcusdt', fetch_kraken_ohlcv, 'XBTUSDT'),
        ('kraken_btcusdc', fetch_kraken_ohlcv, 'XBTUSDC'),
        ('kraken_btceur',  fetch_kraken_ohlcv, 'XXBTZEUR'),
        # Kraken Stablecoin Pairs
        ('kraken_usdcusd', fetch_kraken_ohlcv, 'USDCUSD'),
        ('kraken_usdtusd', fetch_kraken_ohlcv, 'USDTZUSD'),
    ]

    for name, fn, symbol in pairs_conf:
        load_or_fetch(name, fn, symbol)
    print("All data fetched.")

if __name__ == '__main__':
    fetch_all_data()
