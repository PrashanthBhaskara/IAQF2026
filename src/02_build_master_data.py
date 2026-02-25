import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

START_DT = datetime(2023, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
END_DT   = datetime(2023, 3, 21, 23, 59, 0, tzinfo=timezone.utc)

DATA_RAW = 'data_raw'
DATA_PROCESSED = 'data_processed'
os.makedirs(DATA_PROCESSED, exist_ok=True)

def load_data(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_RAW, f"{name}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()

# 1. Load Data
datasets = {
    'binance_btcusdt': load_data('binance_btcusdt'),
    'binance_btcusdc': load_data('binance_btcusdc'),
    'binance_usdcusdt': load_data('binance_usdcusdt'),

    'coinbase_btcusd': load_data('coinbase_btcusd'),
    'coinbase_btcusdt': load_data('coinbase_btcusdt'),
    'coinbase_usdtusd': load_data('coinbase_usdtusd'),

    'kraken_btcusd': load_data('kraken_btcusd'),
    'kraken_btcusdt': load_data('kraken_btcusdt'),
    'kraken_btcusdc': load_data('kraken_btcusdc'),
    'kraken_usdcusd': load_data('kraken_usdcusd'),
    'kraken_usdtusd': load_data('kraken_usdtusd'),
}

# 2. Build common grid
full_index = pd.date_range(START_DT, END_DT, freq='1min', tz='UTC')

def extract_col(df, col, idx, fill=np.nan):
    if df.empty or col not in df.columns:
        return pd.Series(fill, index=idx, dtype=float)
    return df[col].reindex(idx)

# Forward fill prices up to 5 minutes and track ffill flags
raw_prices = {
    name: extract_col(df, 'close', full_index)
    for name, df in datasets.items()
}

prices = pd.DataFrame(index=full_index)
price_ffill_flags = pd.DataFrame(index=full_index)
for name, raw in raw_prices.items():
    filled = raw.ffill(limit=5)
    prices[name] = filled
    price_ffill_flags[name] = raw.isna() & filled.notna()

# Range Proxy: (High - Low) / Close
ranges = pd.DataFrame({
    name: ((extract_col(df, 'high', full_index) - extract_col(df, 'low', full_index)) / 
           extract_col(df, 'close', full_index)).ffill(limit=5)
    for name, df in datasets.items()
})

volumes = pd.DataFrame({
    name: extract_col(df, 'volume', full_index, fill=0.0)
    for name, df in datasets.items()
})

# 3. Calculate pricing dispersion objects and series-level ffill flags
# Unadjusted dispersion (de-peg-sensitive):
#   d_Q(t) = log(P_BTC/Q) - log(P_BTC/USD)
# Adjusted parity residual (marking stablecoin back to USD):
#   b_Q(t) = log(P_BTC/Q * P_Q/USD) - log(P_BTC/USD)
basis = pd.DataFrame(index=full_index)
basis_ffill_flags = pd.DataFrame(index=full_index)

# ---------------------------------------------------------------------
# KRAKEN: USDC and USDT channels against BTC/USD
# ---------------------------------------------------------------------
# Unadjusted dispersion D_t
basis['dispersion_usdc_kraken'] = (
    np.log(prices['kraken_btcusdc']) - np.log(prices['kraken_btcusd'])
) * 10000
basis_ffill_flags['dispersion_usdc_kraken'] = price_ffill_flags[['kraken_btcusdc', 'kraken_btcusd']].any(axis=1)

basis['dispersion_usdt_kraken'] = (
    np.log(prices['kraken_btcusdt']) - np.log(prices['kraken_btcusd'])
) * 10000
basis_ffill_flags['dispersion_usdt_kraken'] = price_ffill_flags[['kraken_btcusdt', 'kraken_btcusd']].any(axis=1)

# Adjusted parity residual B_t
basis['basis_usdc_kraken'] = (
    np.log(prices['kraken_btcusdc'] * prices['kraken_usdcusd']) - np.log(prices['kraken_btcusd'])
) * 10000 # in bps
basis_ffill_flags['basis_usdc_kraken'] = price_ffill_flags[['kraken_btcusdc', 'kraken_usdcusd', 'kraken_btcusd']].any(axis=1)

basis['basis_usdt_kraken'] = (
    np.log(prices['kraken_btcusdt'] * prices['kraken_usdtusd']) - np.log(prices['kraken_btcusd'])
) * 10000
basis_ffill_flags['basis_usdt_kraken'] = price_ffill_flags[['kraken_btcusdt', 'kraken_usdtusd', 'kraken_btcusd']].any(axis=1)

# ---------------------------------------------------------------------
# COINBASE: USDT channel against BTC/USD
# ---------------------------------------------------------------------
basis['dispersion_usdt_coinbase'] = (
    np.log(prices['coinbase_btcusdt']) - np.log(prices['coinbase_btcusd'])
) * 10000
basis_ffill_flags['dispersion_usdt_coinbase'] = price_ffill_flags[['coinbase_btcusdt', 'coinbase_btcusd']].any(axis=1)

basis['basis_usdt_coinbase'] = (
    np.log(prices['coinbase_btcusdt'] * prices['coinbase_usdtusd']) - np.log(prices['coinbase_btcusd'])
) * 10000
basis_ffill_flags['basis_usdt_coinbase'] = price_ffill_flags[['coinbase_btcusdt', 'coinbase_usdtusd', 'coinbase_btcusd']].any(axis=1)

# BINANCE: It doesn't have fiat USD. We can calculate the relative stablecoin basis (BTC/USDC vs BTC/USDT)
# basis_usdc_vs_usdt_binance = log(P_BTC/USDC * P_USDC/USDT) - log(P_BTC/USDT)
basis['basis_usdc_usdt_binance'] = (
    np.log(prices['binance_btcusdc'] * prices['binance_usdcusdt']) - np.log(prices['binance_btcusdt'])
) * 10000
basis_ffill_flags['basis_usdc_usdt_binance'] = price_ffill_flags[['binance_btcusdc', 'binance_usdcusdt', 'binance_btcusdt']].any(axis=1)

# =====================================================================
# CROSS-EXCHANGE BASIS: Same pair, different exchanges (spatial arb)
# =====================================================================
# Cross-exchange BTC/USDT: Binance vs Kraken
basis['xbasis_btcusdt_binance_kraken'] = (
    np.log(prices['binance_btcusdt']) - np.log(prices['kraken_btcusdt'])
) * 10000
basis_ffill_flags['xbasis_btcusdt_binance_kraken'] = price_ffill_flags[['binance_btcusdt', 'kraken_btcusdt']].any(axis=1)

# Cross-exchange BTC/USDT: Coinbase vs Kraken
basis['xbasis_btcusdt_coinbase_kraken'] = (
    np.log(prices['coinbase_btcusdt']) - np.log(prices['kraken_btcusdt'])
) * 10000
basis_ffill_flags['xbasis_btcusdt_coinbase_kraken'] = price_ffill_flags[['coinbase_btcusdt', 'kraken_btcusdt']].any(axis=1)

# Cross-exchange BTC/USD: Coinbase vs Kraken (fiat vs fiat)
basis['xbasis_btcusd_coinbase_kraken'] = (
    np.log(prices['coinbase_btcusd']) - np.log(prices['kraken_btcusd'])
) * 10000
basis_ffill_flags['xbasis_btcusd_coinbase_kraken'] = price_ffill_flags[['coinbase_btcusd', 'kraken_btcusd']].any(axis=1)

# =====================================================================
# Implied USDT/USD from BTC triangulation (Coinbase)
# =====================================================================
# implied = P_BTC/USD / P_BTC/USDT  (should be ~1.0 if USDT pegged)
prices['implied_usdt_usd_coinbase'] = prices['coinbase_btcusd'] / prices['coinbase_btcusdt']
prices['implied_usdt_usd_kraken']   = prices['kraken_btcusd'] / prices['kraken_btcusdt']
prices['implied_usdc_usd_kraken']   = prices['kraken_btcusd'] / prices['kraken_btcusdc']

# USDT/USD deviation from parity in bps (direct pairs)
basis['usdt_peg_dev_kraken'] = (prices['kraken_usdtusd'] - 1.0) * 10000
basis['usdt_peg_dev_coinbase'] = (prices['coinbase_usdtusd'] - 1.0) * 10000
basis['usdc_peg_dev_kraken'] = (prices['kraken_usdcusd'] - 1.0) * 10000
basis_ffill_flags['usdt_peg_dev_kraken'] = price_ffill_flags['kraken_usdtusd']
basis_ffill_flags['usdt_peg_dev_coinbase'] = price_ffill_flags['coinbase_usdtusd']
basis_ffill_flags['usdc_peg_dev_kraken'] = price_ffill_flags['kraken_usdcusd']

# 4. Save to processed
prices.to_parquet(os.path.join(DATA_PROCESSED, 'prices.parquet'))
price_ffill_flags = price_ffill_flags.reindex(columns=prices.columns).fillna(False).astype(bool)
price_ffill_flags.to_parquet(os.path.join(DATA_PROCESSED, 'price_ffill_flags.parquet'))
ranges.to_parquet(os.path.join(DATA_PROCESSED, 'intraminute_ranges.parquet'))
volumes.to_parquet(os.path.join(DATA_PROCESSED, 'volumes.parquet'))
basis.to_parquet(os.path.join(DATA_PROCESSED, 'basis.parquet'))
basis_ffill_flags = basis_ffill_flags.reindex(columns=basis.columns).fillna(False).astype(bool)
basis_ffill_flags.to_parquet(os.path.join(DATA_PROCESSED, 'basis_ffill_flags.parquet'))

print("Master datasets built and saved to data_processed/")

print("Checking for degenerate basis outputs:")
for col in basis.columns:
    valid = basis[col].dropna()
    print(f"{col}: {len(valid)} non-NaN, Mean: {valid.mean():.4f}, Std: {valid.std():.4f}, Unique vals: {valid.nunique()}")
    if valid.nunique() < 10:
        print(f"  --> WARNING: degenerate or broken basis detected for {col}")
