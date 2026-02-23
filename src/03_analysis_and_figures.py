import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (14, 6),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

DATA_PROCESSED = 'data_processed'
FIGURES_DIR = 'figures'
TABLES_DIR = 'tables'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

prices = pd.read_parquet(os.path.join(DATA_PROCESSED, 'prices.parquet'))
ranges = pd.read_parquet(os.path.join(DATA_PROCESSED, 'intraminute_ranges.parquet'))
volumes = pd.read_parquet(os.path.join(DATA_PROCESSED, 'volumes.parquet'))
basis = pd.read_parquet(os.path.join(DATA_PROCESSED, 'basis.parquet'))

returns = prices.pct_change(fill_method=None).dropna()

# REGIMES
svb_start = pd.Timestamp('2023-03-10', tz='UTC')
svb_end = pd.Timestamp('2023-03-13', tz='UTC')
regimes = {
    'Pre-SVB': (prices.index.min(), svb_start),
    'Crisis': (svb_start, svb_end),
    'Post-SVB': (svb_end, prices.index.max())
}

def assign_regime(idx):
    if idx < svb_start: return 'Pre-SVB'
    elif idx < svb_end: return 'Crisis'
    else: return 'Post-SVB'

# OU half-life helper
def ou_halflife(series):
    s = series.dropna()
    if len(s) < 100: return np.nan
    dy = s.diff().dropna()
    lag_y = s.shift().dropna()
    X = sm.add_constant(lag_y)
    model = sm.OLS(dy, X).fit()
    beta = model.params.iloc[1]
    if beta >= 0: return np.inf
    return -np.log(2) / beta

# ============================================================
# FIGURE 1: Comprehensive Basis Time Series (Intra-exchange)
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

for ax in axes:
    ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

# Panel A: USDC and USDT vs USD
ax = axes[0]
for col, lbl, c in [
    ('basis_usdc_kraken', 'USDC/USD Basis (Kraken)', '#2ecc71'),
    ('basis_usdt_kraken', 'USDT/USD Basis (Kraken)', '#3498db'),
    ('basis_usdt_coinbase', 'USDT/USD Basis (Coinbase)', '#e74c3c'),
]:
    if col in basis.columns:
        ax.plot(basis.index, basis[col], linewidth=0.4, alpha=0.85, label=lbl, color=c)
ax.set_title('Panel A: Intra-Exchange Stablecoin vs Fiat Basis')
ax.set_ylabel('Basis (bps)')
ax.legend(loc='upper right', fontsize=9)

# Panel B: USDC/USDT relative basis
ax = axes[1]
if 'basis_usdc_usdt_binance' in basis.columns:
    ax.plot(basis.index, basis['basis_usdc_usdt_binance'], linewidth=0.4, alpha=0.85,
            label='USDC/USDT Rel. Basis (Binance)', color='#9b59b6')
ax.set_title('Panel B: USDC vs USDT Relative Basis (Binance)')
ax.set_ylabel('Basis (bps)')
ax.legend(loc='upper right', fontsize=9)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_basis_timeseries.png'), dpi=150)
plt.close()

# ============================================================
# FIGURE 2: Cross-Exchange Basis (Spatial Arbitrage)
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

for ax in axes:
    ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

# Panel A: BTC/USDT Binance vs Kraken
ax = axes[0]
if 'xbasis_btcusdt_binance_kraken' in basis.columns:
    ax.plot(basis.index, basis['xbasis_btcusdt_binance_kraken'], linewidth=0.4,
            color='#e67e22', label='Binance − Kraken BTC/USDT')
ax.set_title('Panel A: Cross-Exchange BTC/USDT — Binance vs Kraken')
ax.set_ylabel('Basis (bps)')
ax.legend(loc='upper right', fontsize=9)

# Panel B: BTC/USDT Coinbase vs Kraken
ax = axes[1]
if 'xbasis_btcusdt_coinbase_kraken' in basis.columns:
    ax.plot(basis.index, basis['xbasis_btcusdt_coinbase_kraken'], linewidth=0.4,
            color='#1abc9c', label='Coinbase − Kraken BTC/USDT')
ax.set_title('Panel B: Cross-Exchange BTC/USDT — Coinbase vs Kraken')
ax.set_ylabel('Basis (bps)')
ax.legend(loc='upper right', fontsize=9)

# Panel C: BTC/USD Coinbase vs Kraken (fiat-to-fiat)
ax = axes[2]
if 'xbasis_btcusd_coinbase_kraken' in basis.columns:
    ax.plot(basis.index, basis['xbasis_btcusd_coinbase_kraken'], linewidth=0.4,
            color='#2c3e50', label='Coinbase − Kraken BTC/USD')
ax.set_title('Panel C: Cross-Exchange BTC/USD — Coinbase vs Kraken (Fiat-Fiat)')
ax.set_ylabel('Basis (bps)')
ax.legend(loc='upper right', fontsize=9)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_cross_exchange_basis.png'), dpi=150)
plt.close()

# ============================================================
# FIGURE 3: USDT/USD and USDC/USD Peg Deviation Overlay
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

for ax in axes:
    ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

# Panel A: Direct peg prices
ax = axes[0]
if 'kraken_usdcusd' in prices.columns:
    ax.plot(prices.index, prices['kraken_usdcusd'], linewidth=0.5, color='#2ecc71', label='USDC/USD (Kraken)')
if 'kraken_usdtusd' in prices.columns:
    ax.plot(prices.index, prices['kraken_usdtusd'], linewidth=0.5, color='#3498db', label='USDT/USD (Kraken)')
if 'coinbase_usdtusd' in prices.columns:
    ax.plot(prices.index, prices['coinbase_usdtusd'], linewidth=0.5, color='#e74c3c', label='USDT/USD (Coinbase)')
ax.axhline(1.0, color='grey', linewidth=1.0, linestyle='-', alpha=0.5)
ax.set_title('Panel A: Direct Stablecoin Spot Prices Against USD')
ax.set_ylabel('Price (USD)')
ax.set_ylim(0.85, 1.05)
ax.legend(loc='lower right', fontsize=9)

# Panel B: Peg deviations in bps
ax = axes[1]
for col, lbl, c in [
    ('usdc_peg_dev_kraken',   'USDC Peg Deviation (Kraken)', '#2ecc71'),
    ('usdt_peg_dev_kraken',   'USDT Peg Deviation (Kraken)', '#3498db'),
    ('usdt_peg_dev_coinbase', 'USDT Peg Deviation (Coinbase)', '#e74c3c'),
]:
    if col in basis.columns:
        ax.plot(basis.index, basis[col], linewidth=0.5, color=c, label=lbl, alpha=0.8)
ax.set_title('Panel B: Stablecoin Peg Deviations from $1.00 (bps)')
ax.set_ylabel('Deviation (bps)')
ax.legend(loc='lower right', fontsize=9)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_stablecoin_peg.png'), dpi=150)
plt.close()

# ============================================================
# FIGURE 4: Basis Distribution by Regime (USDC + USDT)
# ============================================================
basis['Regime'] = basis.index.map(assign_regime)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col, title in zip(axes, 
    ['basis_usdc_kraken', 'basis_usdt_kraken', 'basis_usdt_coinbase'],
    ['USDC/USD (Kraken)', 'USDT/USD (Kraken)', 'USDT/USD (Coinbase)']
):
    for regime, color in [('Pre-SVB', '#3498db'), ('Crisis', '#e74c3c'), ('Post-SVB', '#2ecc71')]:
        subset = basis.loc[basis['Regime'] == regime, col].dropna()
        if len(subset) > 10:
            ax.hist(subset, bins=80, alpha=0.5, label=regime, color=color, density=True)
    ax.set_title(title)
    ax.set_xlabel('Basis (bps)')
    ax.legend(fontsize=8)

plt.suptitle('Basis Distribution by Regime', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_basis_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 5: Intraminute Range (Liquidity Proxy) by Regime
# ============================================================
ranges['Regime'] = ranges.index.map(assign_regime)
cols_spread = ['kraken_btcusd', 'kraken_btcusdc', 'kraken_btcusdt', 'binance_btcusdt', 'coinbase_btcusd']
melted = ranges.reset_index().melt(id_vars=['index', 'Regime'], value_vars=cols_spread,
                                    var_name='Pair', value_name='Range_bps')
melted['Range_bps'] *= 10000
nice_map = {
    'kraken_btcusd': 'Kraken\nBTC/USD', 'kraken_btcusdc': 'Kraken\nBTC/USDC',
    'kraken_btcusdt': 'Kraken\nBTC/USDT', 'binance_btcusdt': 'Binance\nBTC/USDT',
    'coinbase_btcusd': 'Coinbase\nBTC/USD'
}
melted['Pair'] = melted['Pair'].map(nice_map)

plt.figure(figsize=(14, 6))
sns.boxplot(data=melted, x='Pair', y='Range_bps', hue='Regime', showfliers=False,
            palette={'Pre-SVB': '#3498db', 'Crisis': '#e74c3c', 'Post-SVB': '#2ecc71'})
plt.title('Intraminute Range (Liquidity Proxy) by Pair and Regime')
plt.ylabel('Range (bps)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_liquidity_regime.png'), dpi=150)
plt.close()

# ============================================================
# FIGURE 6: SVB Crisis Zoom (All Basis)
# ============================================================
svb_mask = (basis.index >= svb_start) & (basis.index <= svb_end)
svb_data = basis.loc[svb_mask]

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

ax = axes[0]
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
for col, lbl, c in [
    ('basis_usdc_kraken', 'USDC/USD (Kraken)', '#2ecc71'),
    ('basis_usdt_kraken', 'USDT/USD (Kraken)', '#3498db'),
    ('basis_usdt_coinbase', 'USDT/USD (Coinbase)', '#e74c3c'),
]:
    if col in svb_data.columns:
        ax.plot(svb_data.index, svb_data[col], linewidth=0.8, color=c, label=lbl)
ax.set_title('Panel A: Intra-Exchange Basis During SVB Crisis')
ax.set_ylabel('Basis (bps)')
ax.legend(fontsize=9)

ax = axes[1]
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
for col, lbl, c in [
    ('xbasis_btcusdt_binance_kraken', 'Binance−Kraken BTC/USDT', '#e67e22'),
    ('xbasis_btcusd_coinbase_kraken', 'Coinbase−Kraken BTC/USD', '#2c3e50'),
]:
    if col in svb_data.columns:
        ax.plot(svb_data.index, svb_data[col], linewidth=0.8, color=c, label=lbl)
ax.set_title('Panel B: Cross-Exchange Basis During SVB Crisis')
ax.set_ylabel('Basis (bps)')
ax.legend(fontsize=9)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_svb_crisis_zoom.png'), dpi=150)
plt.close()

# ============================================================
# FIGURE 7: Volume Share (Fragmentation)
# ============================================================
vol_cols = ['binance_btcusdt', 'binance_btcusdc', 'coinbase_btcusd', 'coinbase_btcusdt',
            'kraken_btcusd', 'kraken_btcusdt', 'kraken_btcusdc']
vols_daily = volumes[vol_cols].resample('D').sum()
vols_pct = vols_daily.div(vols_daily.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(14, 6))
ax.stackplot(vols_pct.index,
    vols_pct['binance_btcusdt'], vols_pct['binance_btcusdc'],
    vols_pct['coinbase_btcusd'], vols_pct['coinbase_btcusdt'],
    vols_pct['kraken_btcusd'], vols_pct['kraken_btcusdt'], vols_pct['kraken_btcusdc'],
    labels=['Binance USDT', 'Binance USDC', 'Coinbase USD', 'Coinbase USDT',
            'Kraken USD', 'Kraken USDT', 'Kraken USDC'],
    alpha=0.8)
ax.axvspan(svb_start.normalize(), svb_end.normalize(), alpha=0.3, color='red', label='SVB Crisis')
ax.set_title('Daily Volume Fragmentation Across All Pairs')
ax.set_ylabel('Volume Share (%)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
handles, labels_leg = ax.get_legend_handles_labels()
by_label = dict(zip(labels_leg, handles))
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_volume_share.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 8: Arbitrage After Fees (Multi-Pair)
# ============================================================
FEE_BPS = 10
arb_cols = {
    'basis_usdc_kraken': 'USDC/USD (Kraken)',
    'basis_usdt_kraken': 'USDT/USD (Kraken)',
    'xbasis_btcusdt_binance_kraken': 'Cross-Exch BTC/USDT (Bin−Kra)',
    'xbasis_btcusd_coinbase_kraken': 'Cross-Exch BTC/USD (CB−Kra)',
}

fig, ax = plt.subplots(figsize=(14, 6))
ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis')
colors = ['#9b59b6', '#3498db', '#e67e22', '#2c3e50']
for (col, lbl), c in zip(arb_cols.items(), colors):
    if col in basis.columns:
        net = basis[col].abs() - FEE_BPS
        net[net < 0] = 0
        ax.plot(net.index, net, linewidth=0.4, color=c, label=lbl, alpha=0.85)
ax.set_title(f'Net Arbitrage Opportunity After {FEE_BPS}bps Round-Trip Fees')
ax.set_ylabel('Net Profit (bps)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_arbitrage_after_fees.png'), dpi=150)
plt.close()

# ============================================================
# TABLE 1: Comprehensive OU + ADF Stats by Regime
# ============================================================
all_basis_cols = [c for c in basis.columns if c != 'Regime']
stats_list = []
for regime, (t0, t1) in regimes.items():
    mask = (basis.index >= t0) & (basis.index < t1)
    for col in all_basis_cols:
        series = basis.loc[mask, col].dropna()
        if len(series) < 100: continue
        hl = ou_halflife(series)
        adf_stat, adf_p = adfuller(series, maxlag=5)[:2]
        stats_list.append({
            'Regime': regime,
            'Basis': col,
            'Mean (bps)': round(series.mean(), 2),
            'Std (bps)': round(series.std(), 2),
            'Half-Life (min)': round(hl, 2) if np.isfinite(hl) else 'inf',
            'ADF Stat': round(adf_stat, 2),
            'ADF p-value': f'{adf_p:.4f}',
            'N': len(series),
        })

df_ou = pd.DataFrame(stats_list)
df_ou.to_csv(os.path.join(TABLES_DIR, 'ou_basis_stats.csv'), index=False)

# Also produce a LaTeX-ready table
with open(os.path.join(TABLES_DIR, 'ou_basis_stats.tex'), 'w') as f:
    f.write(df_ou.to_latex(index=False, caption='OU Mean Reversion and ADF Stationarity by Regime',
                            label='tab:ou_stats', column_format='llrrrrrr'))

# ============================================================
# TABLE 2: Explanatory Regression (HAC) — USDC Basis
# ============================================================
df_reg = pd.DataFrame()
df_reg['Basis'] = basis['basis_usdc_kraken']
df_reg['Crisis'] = (basis['Regime'] == 'Crisis').astype(int)
df_reg['Vol'] = returns['kraken_btcusdc'].rolling(60).std() * 10000
df_reg['Liq'] = ranges['kraken_btcusdc'] * 10000
df_reg = df_reg.dropna()

X = sm.add_constant(df_reg[['Crisis', 'Vol', 'Liq']])
y = df_reg['Basis']
model_usdc = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 60})
with open(os.path.join(TABLES_DIR, 'regression_usdc.txt'), 'w') as f:
    f.write("=== USDC/USD Basis Regression ===\n\n")
    f.write(model_usdc.summary().as_text())

# TABLE 3: Explanatory Regression (HAC) — USDT Basis
df_reg2 = pd.DataFrame()
df_reg2['Basis'] = basis['basis_usdt_kraken']
df_reg2['Crisis'] = (basis['Regime'] == 'Crisis').astype(int)
df_reg2['Vol'] = returns['kraken_btcusdt'].rolling(60).std() * 10000
df_reg2['Liq'] = ranges['kraken_btcusdt'] * 10000
df_reg2 = df_reg2.dropna()

X2 = sm.add_constant(df_reg2[['Crisis', 'Vol', 'Liq']])
y2 = df_reg2['Basis']
model_usdt = sm.OLS(y2, X2).fit(cov_type='HAC', cov_kwds={'maxlags': 60})
with open(os.path.join(TABLES_DIR, 'regression_usdt.txt'), 'w') as f:
    f.write("=== USDT/USD Basis Regression ===\n\n")
    f.write(model_usdt.summary().as_text())

# Combine into one file
with open(os.path.join(TABLES_DIR, 'regression_results.txt'), 'w') as f:
    f.write("=== USDC/USD Basis Regression (Kraken) ===\n\n")
    f.write(model_usdc.summary().as_text())
    f.write("\n\n" + "="*60 + "\n\n")
    f.write("=== USDT/USD Basis Regression (Kraken) ===\n\n")
    f.write(model_usdt.summary().as_text())

# ============================================================
# TABLE 4 & FIGURE 9: Multi-Pair Granger Causality
# ============================================================
granger_pairs = [
    ('kraken_btcusd', 'kraken_btcusdc', 'BTC/USD → BTC/USDC (Kraken)'),
    ('kraken_btcusdc', 'kraken_btcusd', 'BTC/USDC → BTC/USD (Kraken)'),
    ('kraken_btcusd', 'kraken_btcusdt', 'BTC/USD → BTC/USDT (Kraken)'),
    ('kraken_btcusdt', 'kraken_btcusd', 'BTC/USDT → BTC/USD (Kraken)'),
    ('binance_btcusdt', 'kraken_btcusdt', 'Binance USDT → Kraken USDT'),
    ('kraken_btcusdt', 'binance_btcusdt', 'Kraken USDT → Binance USDT'),
    ('coinbase_btcusd', 'kraken_btcusd', 'Coinbase USD → Kraken USD'),
    ('kraken_btcusd', 'coinbase_btcusd', 'Kraken USD → Coinbase USD'),
]

granger_results = []
for dep, indep, label in granger_pairs:
    if dep in returns.columns and indep in returns.columns:
        var_data = returns[[dep, indep]].dropna() * 10000
        if len(var_data) < 200: continue
        try:
            var_model = VAR(var_data)
            res = var_model.fit(maxlags=10, ic='aic')
            g_test = res.test_causality(dep, indep, kind='f')
            granger_results.append({
                'Test': label,
                'VAR Lags': res.k_ar,
                'F-stat': round(g_test.test_statistic, 3),
                'p-value': round(g_test.pvalue, 4),
                'Significant': '***' if g_test.pvalue < 0.001 else ('**' if g_test.pvalue < 0.01 else ('*' if g_test.pvalue < 0.05 else ''))
            })
        except Exception as e:
            print(f"  Granger test failed for {label}: {e}")

df_granger = pd.DataFrame(granger_results)
df_granger.to_csv(os.path.join(TABLES_DIR, 'granger_causality.csv'), index=False)

with open(os.path.join(TABLES_DIR, 'granger_causality.txt'), 'w') as f:
    f.write("Comprehensive Granger Causality Results\n")
    f.write("="*60 + "\n\n")
    f.write(df_granger.to_string(index=False))
    f.write("\n")

# Also produce LaTeX version
with open(os.path.join(TABLES_DIR, 'granger_causality.tex'), 'w') as f:
    f.write(df_granger.to_latex(index=False, caption='Granger Causality Tests (Multi-Pair)',
                                 label='tab:granger', column_format='lrrrr'))

# IRF Plot for the core pair
var_data_core = returns[['kraken_btcusd', 'kraken_btcusdc']].dropna() * 10000
var_model_core = VAR(var_data_core)
res_core = var_model_core.fit(maxlags=10, ic='aic')
print(f"VAR optimally selected lags: {res_core.k_ar}")
irf = res_core.irf(10)
fig = irf.plot(orth=True)
plt.savefig(os.path.join(FIGURES_DIR, 'fig_var_irf.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 10: Realized Volatility by Regime
# ============================================================
vol_cols_btc = ['kraken_btcusd', 'kraken_btcusdt', 'kraken_btcusdc', 'binance_btcusdt', 'coinbase_btcusd']
rv = returns[vol_cols_btc].rolling(60).std() * 10000 * np.sqrt(60)  # annualize to hourly vol in bps
rv['Regime'] = rv.index.map(assign_regime)

fig, ax = plt.subplots(figsize=(14, 6))
ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis')
nice_names = {'kraken_btcusd': 'Kraken USD', 'kraken_btcusdt': 'Kraken USDT',
              'kraken_btcusdc': 'Kraken USDC', 'binance_btcusdt': 'Binance USDT',
              'coinbase_btcusd': 'Coinbase USD'}
colors_rv = ['#2c3e50', '#3498db', '#2ecc71', '#e67e22', '#e74c3c']
for (col, nn), c in zip(nice_names.items(), colors_rv):
    if col in rv.columns:
        ax.plot(rv.index, rv[col], linewidth=0.5, color=c, label=nn, alpha=0.8)
ax.set_title('Hourly Realized Volatility (60-min Rolling Std of Returns)')
ax.set_ylabel('Volatility (bps/hr)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_realized_volatility.png'), dpi=150)
plt.close()

# ============================================================
# Summary: Arbitrage Statistics Table
# ============================================================
arb_summary = []
for col, lbl in arb_cols.items():
    if col not in basis.columns: continue
    s = basis[col].dropna()
    net = s.abs() - FEE_BPS
    for regime, (t0, t1) in regimes.items():
        mask = (s.index >= t0) & (s.index < t1)
        sub = s[mask]
        net_sub = net[mask]
        arb_summary.append({
            'Pair': lbl,
            'Regime': regime,
            'Mean |Basis| (bps)': round(sub.abs().mean(), 2),
            '% Minutes > Fee': round((net_sub > 0).mean() * 100, 1),
            'Mean Net Arb (bps)': round(net_sub[net_sub > 0].mean(), 2) if (net_sub > 0).any() else 0,
            'N Minutes': len(sub),
        })

df_arb = pd.DataFrame(arb_summary)
df_arb.to_csv(os.path.join(TABLES_DIR, 'arbitrage_summary.csv'), index=False)
with open(os.path.join(TABLES_DIR, 'arbitrage_summary.tex'), 'w') as f:
    f.write(df_arb.to_latex(index=False, caption='Arbitrage Profitability Summary (10bps Fee Assumption)',
                             label='tab:arb', column_format='llrrrr'))

print("All analysis and figures completed successfully.")
