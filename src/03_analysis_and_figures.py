import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order
from statsmodels.stats.multitest import multipletests
from half_life_utils import (
    estimate_half_life_from_ecm,
    half_life_from_rho,
    run_half_life_sanity_tests,
)

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
price_ff_flags_path = os.path.join(DATA_PROCESSED, 'price_ffill_flags.parquet')
if os.path.exists(price_ff_flags_path):
    price_ff_flags = pd.read_parquet(price_ff_flags_path)
else:
    price_ff_flags = pd.DataFrame(False, index=prices.index, columns=prices.columns)
price_ff_flags = price_ff_flags.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)

ranges = pd.read_parquet(os.path.join(DATA_PROCESSED, 'intraminute_ranges.parquet'))
volumes = pd.read_parquet(os.path.join(DATA_PROCESSED, 'volumes.parquet'))
basis = pd.read_parquet(os.path.join(DATA_PROCESSED, 'basis.parquet'))
basis_ff_flags_path = os.path.join(DATA_PROCESSED, 'basis_ffill_flags.parquet')
if os.path.exists(basis_ff_flags_path):
    basis_ff_flags = pd.read_parquet(basis_ff_flags_path)
else:
    basis_ff_flags = pd.DataFrame(False, index=basis.index, columns=basis.columns)
basis_ff_flags = basis_ff_flags.reindex(index=basis.index, columns=basis.columns).fillna(False).astype(bool)

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

# Utility: compact regime statistics for selected series
def build_regime_stats(df, series_map, regimes_dict):
    out = []
    for regime, (t0, t1) in regimes_dict.items():
        mask = (df.index >= t0) & (df.index < t1)
        for col, lbl in series_map:
            if col not in df.columns:
                continue
            s = df.loc[mask, col].dropna()
            if len(s) == 0:
                continue
            out.append({
                'Regime': regime,
                'Series': lbl,
                'Mean (bps)': round(s.mean(), 2),
                'Std (bps)': round(s.std(), 2),
                'Mean Abs (bps)': round(s.abs().mean(), 2),
                'N': len(s),
            })
    return pd.DataFrame(out)


def make_width_safe_latex(latex_text: str, add_footnotesize: bool = False) -> str:
    """
    Wrap the tabular in a text-width resizebox so wide tables do not clip in PDF.
    """
    if add_footnotesize:
        latex_text = latex_text.replace('\\begin{tabular}', '\\footnotesize\n\\begin{tabular}', 1)
    latex_text = latex_text.replace(
        '\\begin{tabular}',
        '\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}',
        1,
    )
    latex_text = latex_text.replace('\\end{tabular}', '\\end{tabular}%\n}', 1)
    return latex_text


def gg_component_share_from_alpha(alpha_vec):
    """
    Two-market Gonzalo-Granger component share from VECM alpha.
    For alpha = [alpha_1, alpha_2]':
      share_1 = alpha_2 / (alpha_2 - alpha_1)
      share_2 = -alpha_1 / (alpha_2 - alpha_1)
    """
    if len(alpha_vec) != 2:
        raise ValueError("Gonzalo-Granger helper expects exactly 2 alpha coefficients.")
    a1 = float(alpha_vec[0])
    a2 = float(alpha_vec[1])
    denom = a2 - a1
    if np.isclose(denom, 0.0):
        return np.nan, np.nan, 'gg_denominator_near_zero'
    s1 = a2 / denom
    s2 = -a1 / denom
    warning = ''
    if (not np.isfinite(s1)) or (not np.isfinite(s2)):
        warning = 'gg_non_finite'
    elif (s1 < 0.0) or (s1 > 1.0) or (s2 < 0.0) or (s2 > 1.0):
        warning = 'gg_non_convex_share'
    return float(s1), float(s2), warning

# Half-life unit sanity checks
df_hl_sanity = run_half_life_sanity_tests(dt_minutes=1.0)
df_hl_sanity.to_csv(os.path.join(TABLES_DIR, 'half_life_sanity_grid.csv'), index=False)

# ============================================================
# SANITY CHECK: Identity between D_t and B_t
# B_t - D_t - log(P_stablecoin/USD) * 10000 should be ~0.
# ============================================================
identity_specs = [
    ('USDC (Kraken)', 'basis_usdc_kraken', 'dispersion_usdc_kraken', 'kraken_usdcusd'),
    ('USDT (Kraken)', 'basis_usdt_kraken', 'dispersion_usdt_kraken', 'kraken_usdtusd'),
    ('USDT (Coinbase)', 'basis_usdt_coinbase', 'dispersion_usdt_coinbase', 'coinbase_usdtusd'),
]

identity_rows = []
for market, b_col, d_col, peg_col in identity_specs:
    if not ({b_col, d_col}.issubset(basis.columns) and peg_col in prices.columns):
        continue
    aligned = pd.concat(
        [basis[b_col], basis[d_col], prices[peg_col]],
        axis=1, keys=['B', 'D', 'peg']
    ).dropna()
    if aligned.empty:
        continue
    residual = aligned['B'] - aligned['D'] - np.log(aligned['peg']) * 10000
    identity_rows.append({
        'Market': market,
        'N': len(residual),
        'Mean Identity Error (bps)': float(residual.mean()),
        'Max Abs Identity Error (bps)': float(residual.abs().max()),
        'Std Identity Error (bps)': float(residual.std()),
    })

if not identity_rows:
    raise ValueError("Identity check could not run: required D_t/B_t/peg series missing.")

df_identity = pd.DataFrame(identity_rows)
df_identity.to_csv(os.path.join(TABLES_DIR, 'dispersion_adjusted_identity_check.csv'), index=False)

IDENTITY_TOL_BPS = 1e-6
if (df_identity['Max Abs Identity Error (bps)'] > IDENTITY_TOL_BPS).any():
    raise ValueError(
        f"Identity check failed: residual exceeded tolerance {IDENTITY_TOL_BPS} bps.\n"
        f"{df_identity.to_string(index=False)}"
    )

print("\nDispersion vs adjusted residual identity check (bps):")
print(df_identity.to_string(index=False))

# ============================================================
# FIGURE 1: Comprehensive Adjusted Residual Time Series (Intra-exchange)
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

for ax in axes:
    ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

# Panel A: USDC and USDT vs USD
ax = axes[0]
for col, lbl, c in [
    ('basis_usdc_kraken', 'USDC/USD Adjusted Residual $B_t$ (Kraken)', '#2ecc71'),
    ('basis_usdt_kraken', 'USDT/USD Adjusted Residual $B_t$ (Kraken)', '#3498db'),
    ('basis_usdt_coinbase', 'USDT/USD Adjusted Residual $B_t$ (Coinbase)', '#e74c3c'),
]:
    if col in basis.columns:
        ax.plot(basis.index, basis[col], linewidth=0.4, alpha=0.85, label=lbl, color=c)
ax.set_title('Panel A: Intra-Exchange Adjusted Parity Residual ($B_t$)')
ax.set_ylabel('Adjusted Residual (bps)')
ax.legend(loc='upper right', fontsize=9)

# Panel B: USDC/USDT relative basis
ax = axes[1]
if 'basis_usdc_usdt_binance' in basis.columns:
    ax.plot(basis.index, basis['basis_usdc_usdt_binance'], linewidth=0.4, alpha=0.85,
            label='USDC/USDT Rel. Basis (Binance)', color='#9b59b6')
ax.set_title('Panel B: USDC vs USDT Relative Adjusted Residual (Binance)')
ax.set_ylabel('Adjusted Residual (bps)')
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
# FIGURE 4: Crisis window comparison of D_t vs B_t (Kraken)
# ============================================================
svb_zoom_start = pd.Timestamp('2023-03-10', tz='UTC')
svb_zoom_end = pd.Timestamp('2023-03-13 23:59:00', tz='UTC')
svb_zoom_mask = (basis.index >= svb_zoom_start) & (basis.index <= svb_zoom_end)
svb_zoom = basis.loc[svb_zoom_mask]

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
for ax in axes:
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis Core')

series_pairs = [
    ('dispersion_usdc_kraken', 'basis_usdc_kraken', 'USDC Channel (Kraken)'),
    ('dispersion_usdt_kraken', 'basis_usdt_kraken', 'USDT Channel (Kraken)'),
]
for ax, (d_col, b_col, title) in zip(axes, series_pairs):
    if d_col in svb_zoom.columns:
        ax.plot(svb_zoom.index, svb_zoom[d_col], color='#e67e22', linewidth=0.9, alpha=0.85,
                label='Unadjusted dispersion $D_t$')
    if b_col in svb_zoom.columns:
        ax.plot(svb_zoom.index, svb_zoom[b_col], color='#2c3e50', linewidth=0.9, alpha=0.85,
                label='Adjusted residual $B_t$')
    ax.set_title(title)
    ax.set_ylabel('bps')
    ax.legend(loc='upper right', fontsize=9)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_dispersion_vs_adjusted_kraken.png'), dpi=150)
plt.close()

# ============================================================
# FIGURE 5: Basis Distribution by Regime (USDC + USDT)
# ============================================================
basis['Regime'] = basis.index.map(assign_regime)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col, title in zip(axes, 
    ['basis_usdc_kraken', 'basis_usdt_kraken', 'basis_usdt_coinbase'],
    ['USDC/USD $B_t$ (Kraken)', 'USDT/USD $B_t$ (Kraken)', 'USDT/USD $B_t$ (Coinbase)']
):
    for regime, color in [('Pre-SVB', '#3498db'), ('Crisis', '#e74c3c'), ('Post-SVB', '#2ecc71')]:
        subset = basis.loc[basis['Regime'] == regime, col].dropna()
        if len(subset) > 10:
            ax.hist(subset, bins=80, alpha=0.5, label=regime, color=color, density=True)
    ax.set_title(title)
    ax.set_xlabel('Adjusted Residual (bps)')
    ax.legend(fontsize=8)

plt.suptitle('Adjusted Residual ($B_t$) Distribution by Regime', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_basis_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# TABLE 1: D_t vs B_t regime statistics (Kraken)
# ============================================================
series_map_dispersion_adjusted = [
    ('dispersion_usdc_kraken', 'USDC Kraken $D_t$ (Unadjusted)'),
    ('basis_usdc_kraken', 'USDC Kraken $B_t$ (Adjusted)'),
    ('dispersion_usdt_kraken', 'USDT Kraken $D_t$ (Unadjusted)'),
    ('basis_usdt_kraken', 'USDT Kraken $B_t$ (Adjusted)'),
]
df_disp_adj = build_regime_stats(basis, series_map_dispersion_adjusted, regimes)
df_disp_adj.to_csv(os.path.join(TABLES_DIR, 'dispersion_adjusted_stats.csv'), index=False)
with open(os.path.join(TABLES_DIR, 'dispersion_adjusted_stats.tex'), 'w') as f:
    f.write(df_disp_adj.to_latex(
        index=False,
        caption='Regime Statistics for Unadjusted Dispersion ($D_t$) and Adjusted Residual ($B_t$), Kraken',
        label='tab:dispersion_vs_adjusted',
        column_format='llrrrr',
        float_format='%.2f'
    ))

# ============================================================
# FIGURE 6: Intra-minute Range (Range Proxy) by Regime
# ============================================================
ranges['Regime'] = ranges.index.map(assign_regime)
cols_range = ['kraken_btcusd', 'kraken_btcusdc', 'kraken_btcusdt', 'binance_btcusdt', 'coinbase_btcusd']
melted = ranges.reset_index().melt(
    id_vars=['index', 'Regime'],
    value_vars=cols_range,
    var_name='Pair',
    value_name='range_proxy_bps'
)
melted['range_proxy_bps'] *= 10000
nice_map = {
    'kraken_btcusd': 'Kraken\nBTC/USD', 'kraken_btcusdc': 'Kraken\nBTC/USDC',
    'kraken_btcusdt': 'Kraken\nBTC/USDT', 'binance_btcusdt': 'Binance\nBTC/USDT',
    'coinbase_btcusd': 'Coinbase\nBTC/USD'
}
melted['Pair'] = melted['Pair'].map(nice_map)

plt.figure(figsize=(14, 6))
sns.boxplot(data=melted, x='Pair', y='range_proxy_bps', hue='Regime', showfliers=False,
            palette={'Pre-SVB': '#3498db', 'Crisis': '#e74c3c', 'Post-SVB': '#2ecc71'})
plt.title('Intra-minute Range (Range Proxy) by Pair and Regime')
plt.ylabel('Intra-minute Range (bps)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_liquidity_regime.png'), dpi=150)
plt.close()

# ============================================================
# FIGURE 7: SVB Crisis Zoom (All Basis)
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
ax.set_title('Panel A: Intra-Exchange Adjusted Residual ($B_t$) During SVB Crisis')
ax.set_ylabel('Adjusted Residual (bps)')
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
# FIGURE 8: Volume Share (Fragmentation)
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
# FIGURE 9: Arbitrage After Fees (Multi-Pair)
# ============================================================
FEE_BPS_PER_LEG = 5.0
arb_channel_specs = [
    {
        'channel_key': 'basis_usdc_kraken',
        'basis_col': 'basis_usdc_kraken',
        'label_short': 'USDC/USD Kraken (3-leg)',
        'label_table': 'USDC/USD (Kraken, 3-leg triangular)',
        'n_legs': 3,
        'range_leg_cols': ['kraken_btcusdc', 'kraken_usdcusd', 'kraken_btcusd'],
        'assumption_note': 'intra_exchange_triangular',
    },
    {
        'channel_key': 'basis_usdt_kraken',
        'basis_col': 'basis_usdt_kraken',
        'label_short': 'USDT/USD Kraken (3-leg)',
        'label_table': 'USDT/USD (Kraken, 3-leg triangular)',
        'n_legs': 3,
        'range_leg_cols': ['kraken_btcusdt', 'kraken_usdtusd', 'kraken_btcusd'],
        'assumption_note': 'intra_exchange_triangular',
    },
    {
        'channel_key': 'xbasis_btcusdt_binance_kraken',
        'basis_col': 'xbasis_btcusdt_binance_kraken',
        'label_short': 'Cross BTC/USDT Bin-Kra (2-leg)',
        'label_table': 'Cross BTC/USDT (Binance-Kraken, 2-leg pre-funded)',
        'n_legs': 2,
        'range_leg_cols': ['binance_btcusdt', 'kraken_btcusdt'],
        'assumption_note': 'cross_exchange_prefunded',
    },
    {
        'channel_key': 'xbasis_btcusd_coinbase_kraken',
        'basis_col': 'xbasis_btcusd_coinbase_kraken',
        'label_short': 'Cross BTC/USD CB-Kra (2-leg)',
        'label_table': 'Cross BTC/USD (Coinbase-Kraken, 2-leg pre-funded)',
        'n_legs': 2,
        'range_leg_cols': ['coinbase_btcusd', 'kraken_btcusd'],
        'assumption_note': 'cross_exchange_prefunded',
    },
]

arb_channel_data = {}
for spec in arb_channel_specs:
    basis_col = spec['basis_col']
    missing = []
    if basis_col not in basis.columns:
        missing.append(basis_col)
    for rc in spec['range_leg_cols']:
        if rc not in ranges.columns:
            missing.append(rc)
    if missing:
        print(f"Skipping arbitrage channel {spec['channel_key']} due to missing columns: {missing}")
        continue

    channel_df = pd.DataFrame(index=basis.index)
    channel_df['abs_basis_bps'] = basis[basis_col].abs()
    leg_range_cols_bps = []
    for i, rc in enumerate(spec['range_leg_cols'], start=1):
        leg_col = f'leg_range_{i}_bps'
        channel_df[leg_col] = ranges[rc] * 10000.0
        leg_range_cols_bps.append(leg_col)

    # Require basis and all leg range proxies to be present for cost-comparable rows.
    channel_df = channel_df.dropna()
    if channel_df.empty:
        print(f"Skipping arbitrage channel {spec['channel_key']} because aligned data is empty.")
        continue

    fee_component = spec['n_legs'] * FEE_BPS_PER_LEG
    channel_df['fee_component_bps'] = fee_component
    channel_df['slippage_cost_bps'] = 0.5 * channel_df[leg_range_cols_bps].sum(axis=1)
    channel_df['cost_fee_only_bps'] = channel_df['fee_component_bps']
    channel_df['cost_fee_slippage_bps'] = channel_df['fee_component_bps'] + channel_df['slippage_cost_bps']
    channel_df['net_fee_only_bps'] = (channel_df['abs_basis_bps'] - channel_df['cost_fee_only_bps']).clip(lower=0.0)
    channel_df['net_fee_slippage_bps'] = (channel_df['abs_basis_bps'] - channel_df['cost_fee_slippage_bps']).clip(lower=0.0)

    if (channel_df['net_fee_slippage_bps'] > channel_df['net_fee_only_bps'] + 1e-10).any():
        raise AssertionError(f"Arbitrage cost monotonicity violated for {spec['channel_key']}")

    arb_channel_data[spec['channel_key']] = {
        'spec': spec,
        'df': channel_df,
    }

if not arb_channel_data:
    raise ValueError("No arbitrage channels available after applying Stage 5 trade/cost requirements.")

fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
for ax in axes:
    ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', label='SVB Crisis')

colors = ['#9b59b6', '#3498db', '#e67e22', '#2c3e50']
for (_, channel_obj), c in zip(arb_channel_data.items(), colors):
    spec = channel_obj['spec']
    channel_df = channel_obj['df']
    axes[0].plot(channel_df.index, channel_df['net_fee_only_bps'], linewidth=0.45, color=c, label=spec['label_short'], alpha=0.85)
    axes[1].plot(channel_df.index, channel_df['net_fee_slippage_bps'], linewidth=0.45, color=c, label=spec['label_short'], alpha=0.85)

axes[0].set_title(f'Panel A: Fee-Only Upper-Bound Net Arbitrage (fee = {FEE_BPS_PER_LEG:.1f} bps per taker leg)')
axes[0].set_ylabel('Net Profit (bps)')
axes[0].legend(loc='upper left', fontsize=9)

axes[1].set_title('Panel B: Fee + Range-Based Slippage Conservative Net Arbitrage')
axes[1].set_ylabel('Net Profit (bps)')
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
axes[1].legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_arbitrage_after_fees.png'), dpi=150)
plt.close()

# ============================================================
# TABLE 2: Comprehensive OU + ADF Stats by Regime
# ============================================================
all_basis_cols = [c for c in basis.columns if c != 'Regime']
stats_list = []
for regime, (t0, t1) in regimes.items():
    mask = (basis.index >= t0) & (basis.index < t1)
    for col in all_basis_cols:
        series = basis.loc[mask, col]
        clean = series.dropna()
        if len(clean) < 100:
            continue

        est = estimate_half_life_from_ecm(
            series=series,
            dt_minutes=1.0,
            ff_mask=None,   # main table includes all observations
            min_obs=100,
        )
        adf_stat, adf_p = adfuller(clean, maxlag=5)[:2]
        stats_list.append({
            'Regime': regime,
            'Basis': col,
            'Mean (bps)': round(clean.mean(), 2),
            'Std (bps)': round(clean.std(), 2),
            'Estimation Form': est['estimation_form'],
            'rho_est': round(est['rho_est'], 6) if np.isfinite(est['rho_est']) else np.nan,
            'Half-Life (min)': round(est['half_life_min'], 2) if np.isfinite(est['half_life_min']) else np.nan,
            'ADF Stat': round(adf_stat, 2),
            'ADF p-value': f'{adf_p:.4f}',
            'N': len(clean),
            'HL Warning': est['warning'],
        })

df_ou = pd.DataFrame(stats_list)
df_ou.to_csv(os.path.join(TABLES_DIR, 'ou_basis_stats.csv'), index=False)

# Also produce a LaTeX-ready table
with open(os.path.join(TABLES_DIR, 'ou_basis_stats.tex'), 'w') as f:
    f.write(df_ou.to_latex(
        index=False,
        caption='OU/AR(1) Mean Reversion and ADF Stationarity by Regime (Exact Half-Life Mapping)',
        label='tab:ou_stats',
        column_format='lllrrrrrrl',
        float_format='%.4f',
        escape=True
    ))

# ============================================================
# TABLE 3: Half-Life Robustness (1m vs 5m, all vs no-ff)
# ============================================================
robust_series = [
    ('basis_usdc_kraken', 'USDC/USD B_t (Kraken)'),
    ('basis_usdt_kraken', 'USDT/USD B_t (Kraken)'),
]
robust_rows = []

for regime, (t0, t1) in regimes.items():
    regime_mask = (basis.index >= t0) & (basis.index < t1)

    for col, series_label in robust_series:
        if col not in basis.columns:
            continue

        s_1m = basis.loc[regime_mask, col]
        ff_1m = basis_ff_flags.loc[regime_mask, col] if col in basis_ff_flags.columns else pd.Series(False, index=s_1m.index)

        freq_configs = [
            ('1m', 1.0, s_1m, ff_1m),
            (
                '5m',
                5.0,
                s_1m.resample('5min').last(),
                ff_1m.astype(float).resample('5min').last().fillna(0.0).astype(bool),
            ),
        ]

        for freq_label, dt_minutes, s_freq, ff_freq in freq_configs:
            for ff_filter, ff_arg in [('all', None), ('no_ff', ff_freq)]:
                est = estimate_half_life_from_ecm(
                    series=s_freq,
                    dt_minutes=dt_minutes,
                    ff_mask=ff_arg,
                    min_obs=80,
                )
                robust_rows.append({
                    'series': series_label,
                    'regime': regime,
                    'freq': freq_label,
                    'ff_filter': ff_filter,
                    'rho_est': est['rho_est'],
                    'half_life_min': est['half_life_min'],
                    'n_obs': est['n_obs'],
                    'warning': est['warning'],
                })

df_hl_robust = pd.DataFrame(robust_rows)

# dt/rho consistency verification for reported robustness table
freq_to_dt = {'1m': 1.0, '5m': 5.0}
df_hl_robust['dt_minutes'] = df_hl_robust['freq'].map(freq_to_dt)
df_hl_robust['half_life_recalc'] = df_hl_robust.apply(
    lambda r: half_life_from_rho(r['rho_est'], r['dt_minutes']),
    axis=1
)

valid_hl_mask = df_hl_robust['half_life_min'].notna() & df_hl_robust['half_life_recalc'].notna()
if valid_hl_mask.any():
    max_diff = (df_hl_robust.loc[valid_hl_mask, 'half_life_min'] - df_hl_robust.loc[valid_hl_mask, 'half_life_recalc']).abs().max()
    if max_diff > 1e-10:
        raise AssertionError(f"Half-life dt consistency check failed: max diff = {max_diff}")

finite_rho = df_hl_robust['rho_est'].dropna()
if ((finite_rho <= 0.0) | (finite_rho >= 1.0)).any():
    print("WARNING: Some robustness specs produced rho outside (0,1); half-life is undefined there and set to NaN.")

if set(df_hl_robust['freq'].unique()) != {'1m', '5m'}:
    raise AssertionError("Robustness table must contain both 1m and 5m frequencies.")

df_hl_robust = df_hl_robust[['series', 'regime', 'freq', 'ff_filter', 'rho_est', 'half_life_min', 'n_obs', 'warning']]
df_hl_robust.to_csv(os.path.join(TABLES_DIR, 'half_life_robustness.csv'), index=False)
with open(os.path.join(TABLES_DIR, 'half_life_robustness.tex'), 'w') as f:
    f.write(df_hl_robust.to_latex(
        index=False,
        caption='Half-Life Robustness for Adjusted Residuals ($B_t$): 1m vs 5m and All vs No-Forward-Fill',
        label='tab:half_life_robustness',
        column_format='llllrrrl',
        float_format='%.4f',
        escape=True
    ))

# ============================================================
# TABLE 4: Range Proxy vs Realized Volatility Correlation
# ============================================================
range_vol_rows = []
corr_pairs = [
    ('kraken_btcusd', 'Kraken BTC/USD'),
    ('kraken_btcusdc', 'Kraken BTC/USDC'),
    ('kraken_btcusdt', 'Kraken BTC/USDT'),
    ('binance_btcusdt', 'Binance BTC/USDT'),
    ('coinbase_btcusd', 'Coinbase BTC/USD'),
]
for col, label in corr_pairs:
    if col not in ranges.columns or col not in returns.columns:
        continue
    df_corr = pd.DataFrame({
        'range_proxy_bps': ranges[col] * 10000,
        'realized_vol_60m_bps': returns[col].rolling(60).std() * 10000,
    }).dropna()
    if len(df_corr) == 0:
        continue

    overall_corr = df_corr['range_proxy_bps'].corr(df_corr['realized_vol_60m_bps'])
    range_vol_rows.append({
        'Pair': label,
        'Regime': 'Overall',
        'Corr(Range, RV60m)': overall_corr,
        'N': len(df_corr),
    })

    for regime, (t0, t1) in regimes.items():
        m = (df_corr.index >= t0) & (df_corr.index < t1)
        sub = df_corr.loc[m]
        if len(sub) < 50:
            continue
        range_vol_rows.append({
            'Pair': label,
            'Regime': regime,
            'Corr(Range, RV60m)': sub['range_proxy_bps'].corr(sub['realized_vol_60m_bps']),
            'N': len(sub),
        })

df_range_vol_corr = pd.DataFrame(range_vol_rows)
df_range_vol_corr.to_csv(os.path.join(TABLES_DIR, 'range_vol_corr.csv'), index=False)
with open(os.path.join(TABLES_DIR, 'range_vol_corr.tex'), 'w') as f:
    f.write(df_range_vol_corr.to_latex(
        index=False,
        caption='Correlation Between Intra-minute Range Proxy and 60-minute Realized Volatility',
        label='tab:range_vol_corr',
        column_format='llrr',
        float_format='%.4f',
        escape=True
    ))

# ============================================================
# TABLE 5: Explanatory Regression (HAC) — USDC Basis
# ============================================================
df_reg = pd.DataFrame()
df_reg['Basis'] = basis['basis_usdc_kraken']
df_reg['Crisis'] = (basis['Regime'] == 'Crisis').astype(int)
df_reg['RealizedVol60m'] = returns['kraken_btcusdc'].rolling(60).std() * 10000
df_reg['RangeProxy'] = ranges['kraken_btcusdc'] * 10000
df_reg = df_reg.dropna()

X = sm.add_constant(df_reg[['Crisis', 'RealizedVol60m', 'RangeProxy']])
y = df_reg['Basis']
model_usdc = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 60})
with open(os.path.join(TABLES_DIR, 'regression_usdc.txt'), 'w') as f:
    f.write("=== USDC/USD Basis Regression ===\n\n")
    f.write(model_usdc.summary().as_text())

# TABLE 6: Explanatory Regression (HAC) — USDT Basis
df_reg2 = pd.DataFrame()
df_reg2['Basis'] = basis['basis_usdt_kraken']
df_reg2['Crisis'] = (basis['Regime'] == 'Crisis').astype(int)
df_reg2['RealizedVol60m'] = returns['kraken_btcusdt'].rolling(60).std() * 10000
df_reg2['RangeProxy'] = ranges['kraken_btcusdt'] * 10000
df_reg2 = df_reg2.dropna()

X2 = sm.add_constant(df_reg2[['Crisis', 'RealizedVol60m', 'RangeProxy']])
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
# TABLE 7: Johansen Cointegration (Primary Channels, No-FF)
# TABLE 8: VECM Price Discovery Metrics (Primary Channels, No-FF)
# ============================================================
primary_channels = [
    {
        'channel': 'Kraken BTC/USD vs BTC/USDC',
        'market_1': 'Kraken BTC/USD',
        'market_2': 'Kraken BTC/USDC',
        'col_1': 'kraken_btcusd',
        'col_2': 'kraken_btcusdc',
    },
    {
        'channel': 'Kraken BTC/USD vs BTC/USDT',
        'market_1': 'Kraken BTC/USD',
        'market_2': 'Kraken BTC/USDT',
        'col_1': 'kraken_btcusd',
        'col_2': 'kraken_btcusdt',
    },
]

johansen_rows = []
discovery_rows = []

def johansen_rank_summary(endog_levels, k_ar_diff):
    joh = coint_johansen(endog_levels, det_order=0, k_ar_diff=k_ar_diff)
    trace_r0 = float(joh.lr1[0])
    trace_r1 = float(joh.lr1[1])
    crit95_r0 = float(joh.cvt[0, 1])
    crit95_r1 = float(joh.cvt[1, 1])
    reject_r0 = trace_r0 > crit95_r0
    reject_r1 = trace_r1 > crit95_r1
    rank_95 = int(reject_r0) + int(reject_r1)
    return {
        'trace_stat_r0': trace_r0,
        'trace_stat_r1': trace_r1,
        'trace_crit95_r0': crit95_r0,
        'trace_crit95_r1': crit95_r1,
        'reject_r0_95': reject_r0,
        'reject_r1_95': reject_r1,
        'rank_95': rank_95,
    }


for ch in primary_channels:
    c1 = ch['col_1']
    c2 = ch['col_2']
    if c1 not in prices.columns or c2 not in prices.columns:
        continue

    df_levels = pd.DataFrame({
        'p1': np.log(prices[c1]),
        'p2': np.log(prices[c2]),
    }, index=prices.index)

    ff_mask = pd.Series(False, index=prices.index)
    if c1 in price_ff_flags.columns:
        ff_mask = ff_mask | price_ff_flags[c1]
    if c2 in price_ff_flags.columns:
        ff_mask = ff_mask | price_ff_flags[c2]

    # Required in Stage 4: use no-forward-fill dataset only.
    df_levels = df_levels[~ff_mask].dropna()
    if len(df_levels) < 500:
        raise ValueError(f"Insufficient no-ff observations for {ch['channel']}: {len(df_levels)}")
    endog_levels = df_levels[['p1', 'p2']].to_numpy()

    # Lag order selection on levels VAR order p; VECM uses k_ar_diff = p - 1.
    sel = select_order(endog_levels, maxlags=10, deterministic='ci')
    p_aic = sel.selected_orders.get('aic')
    p_bic = sel.selected_orders.get('bic')
    p_used = p_bic if p_bic is not None else (p_aic if p_aic is not None else 2)
    p_used = int(max(p_used, 1))
    k_ar_diff = max(p_used - 1, 0)

    joh_base = johansen_rank_summary(endog_levels, k_ar_diff=k_ar_diff)
    k_lag_minus1 = max(k_ar_diff - 1, 0)
    k_lag_plus1 = k_ar_diff + 1
    joh_minus1 = johansen_rank_summary(endog_levels, k_ar_diff=k_lag_minus1)
    joh_plus1 = johansen_rank_summary(endog_levels, k_ar_diff=k_lag_plus1)
    rank_95 = joh_base['rank_95']
    rank_used = min(rank_95, 1)  # two-market primary claim uses at most rank 1.

    johansen_rows.append({
        'channel': ch['channel'],
        'market_1': ch['market_1'],
        'market_2': ch['market_2'],
        'n_obs_no_ff': int(len(df_levels)),
        'deterministic': 'constant_in_cointegration',
        'lag_rule': 'BIC (fallback AIC)',
        'selected_p_aic': int(p_aic) if p_aic is not None else np.nan,
        'selected_p_bic': int(p_bic) if p_bic is not None else np.nan,
        'selected_p_used': p_used,
        'k_ar_diff_used': k_ar_diff,
        'trace_stat_r0': joh_base['trace_stat_r0'],
        'trace_crit95_r0': joh_base['trace_crit95_r0'],
        'reject_r0_95': joh_base['reject_r0_95'],
        'trace_stat_r1': joh_base['trace_stat_r1'],
        'trace_crit95_r1': joh_base['trace_crit95_r1'],
        'reject_r1_95': joh_base['reject_r1_95'],
        'rank_95': rank_95,
        'rank_used': rank_used,
        'rank_95_lag_minus1': joh_minus1['rank_95'],
        'rank_95_lag_plus1': joh_plus1['rank_95'],
    })

    if rank_used < 1:
        discovery_rows.append({
            'channel': ch['channel'],
            'market_1': ch['market_1'],
            'market_2': ch['market_2'],
            'k_ar_diff_used': k_ar_diff,
            'rank_used': rank_used,
            'alpha_market_1': np.nan,
            'alpha_market_2': np.nan,
            'abs_alpha_market_1': np.nan,
            'abs_alpha_market_2': np.nan,
            'gg_share_market_1': np.nan,
            'gg_share_market_2': np.nan,
            'leader_by_adjustment': 'undetermined_no_cointegration',
            'follower_by_adjustment': 'undetermined_no_cointegration',
            'gg_warning': 'no_rank1_cointegration',
            'leader_stable_lag_pm1': np.nan,
            'alpha_m1_lag_minus1': np.nan,
            'alpha_m1_lag_plus1': np.nan,
            'alpha_m2_lag_minus1': np.nan,
            'alpha_m2_lag_plus1': np.nan,
        })
        continue

    vecm = VECM(endog_levels, k_ar_diff=k_ar_diff, coint_rank=rank_used, deterministic='ci').fit()
    alpha = vecm.alpha[:, 0]
    a1 = float(alpha[0])
    a2 = float(alpha[1])
    abs_a1 = abs(a1)
    abs_a2 = abs(a2)

    if np.isclose(abs_a1, abs_a2):
        leader = 'co-adjusting'
        follower = 'co-adjusting'
    elif abs_a1 < abs_a2:
        leader = ch['market_1']
        follower = ch['market_2']
    else:
        leader = ch['market_2']
        follower = ch['market_1']

    gg1, gg2, gg_warning = gg_component_share_from_alpha(alpha)

    # Quick lag +/-1 robustness for leader direction.
    robust_alphas = {}
    robust_ranks = {}
    for delta in (-1, 1):
        k_alt = max(k_ar_diff + delta, 0)
        rank_alt = johansen_rank_summary(endog_levels, k_ar_diff=k_alt)['rank_95']
        robust_ranks[delta] = rank_alt
        if rank_alt < rank_used:
            robust_alphas[delta] = np.array([np.nan, np.nan])
            continue
        vecm_alt = VECM(endog_levels, k_ar_diff=k_alt, coint_rank=rank_used, deterministic='ci').fit()
        robust_alphas[delta] = vecm_alt.alpha[:, 0]

    def leader_from_alpha(alpha_vec):
        if not np.isfinite(alpha_vec).all():
            return 'undetermined_rank_change'
        if np.isclose(abs(alpha_vec[0]), abs(alpha_vec[1])):
            return 'co-adjusting'
        return ch['market_1'] if abs(alpha_vec[0]) < abs(alpha_vec[1]) else ch['market_2']

    leader_m1 = leader_from_alpha(robust_alphas[-1])
    leader_p1 = leader_from_alpha(robust_alphas[1])
    leader_stable = (leader_m1 == leader) and (leader_p1 == leader)
    if (robust_ranks[-1] < rank_used) or (robust_ranks[1] < rank_used):
        gg_warning = ';'.join([w for w in [gg_warning, 'rank_not_stable_lag_pm1'] if w])

    discovery_rows.append({
        'channel': ch['channel'],
        'market_1': ch['market_1'],
        'market_2': ch['market_2'],
        'k_ar_diff_used': k_ar_diff,
        'rank_used': rank_used,
        'alpha_market_1': a1,
        'alpha_market_2': a2,
        'abs_alpha_market_1': abs_a1,
        'abs_alpha_market_2': abs_a2,
        'gg_share_market_1': gg1,
        'gg_share_market_2': gg2,
        'leader_by_adjustment': leader,
        'follower_by_adjustment': follower,
        'gg_warning': gg_warning,
        'leader_stable_lag_pm1': leader_stable,
        'alpha_m1_lag_minus1': float(robust_alphas[-1][0]),
        'alpha_m1_lag_plus1': float(robust_alphas[1][0]),
        'alpha_m2_lag_minus1': float(robust_alphas[-1][1]),
        'alpha_m2_lag_plus1': float(robust_alphas[1][1]),
    })

df_johansen = pd.DataFrame(johansen_rows)
df_johansen.to_csv(os.path.join(TABLES_DIR, 'cointegration_johansen.csv'), index=False)
df_johansen_tex = df_johansen[[
    'channel',
    'n_obs_no_ff',
    'selected_p_used',
    'k_ar_diff_used',
    'trace_stat_r0',
    'trace_crit95_r0',
    'reject_r0_95',
    'rank_used',
    'rank_95_lag_minus1',
    'rank_95_lag_plus1',
]].rename(columns={
    'channel': 'Channel',
    'n_obs_no_ff': 'N (No-FF)',
    'selected_p_used': 'VAR p',
    'k_ar_diff_used': 'VECM k_diff',
    'trace_stat_r0': 'Trace r=0',
    'trace_crit95_r0': 'Crit95 r=0',
    'reject_r0_95': 'Reject r=0',
    'rank_used': 'Rank Used',
    'rank_95_lag_minus1': 'Rank (k-1)',
    'rank_95_lag_plus1': 'Rank (k+1)',
})
with open(os.path.join(TABLES_DIR, 'cointegration_johansen.tex'), 'w') as f:
    latex_johansen = df_johansen_tex.to_latex(
        index=False,
        caption='Johansen Cointegration Tests on Log Price Levels (Primary Kraken Channels, No-FF Sample)',
        label='tab:johansen',
        float_format='%.4f',
        escape=True
    )
    f.write(make_width_safe_latex(latex_johansen, add_footnotesize=True))

df_discovery = pd.DataFrame(discovery_rows)
df_discovery_csv = df_discovery.rename(columns={
    'gg_share_market_1': 'gg_component_metric_market_1',
    'gg_share_market_2': 'gg_component_metric_market_2',
})
df_discovery_csv.to_csv(os.path.join(TABLES_DIR, 'price_discovery_metrics.csv'), index=False)
df_discovery_tex = df_discovery[[
    'channel',
    'k_ar_diff_used',
    'rank_used',
    'alpha_market_1',
    'alpha_market_2',
    'leader_by_adjustment',
    'leader_stable_lag_pm1',
    'gg_share_market_1',
    'gg_share_market_2',
    'gg_warning',
]].rename(columns={
    'channel': 'Channel',
    'k_ar_diff_used': 'VECM k_diff',
    'rank_used': 'Rank Used',
    'alpha_market_1': 'alpha_market_1',
    'alpha_market_2': 'alpha_market_2',
    'leader_by_adjustment': 'Leader by |alpha|',
    'leader_stable_lag_pm1': 'Leader Stable (k±1)',
    'gg_share_market_1': 'GG component metric mkt1',
    'gg_share_market_2': 'GG component metric mkt2',
    'gg_warning': 'GG note',
})
with open(os.path.join(TABLES_DIR, 'price_discovery_metrics.tex'), 'w') as f:
    latex_discovery = df_discovery_tex.to_latex(
        index=False,
        caption='VECM Adjustment and GG Component Metrics (may be outside [0,1]) for Primary Kraken Channels (No-FF Sample)',
        label='tab:price_discovery',
        float_format='%.4f',
        escape=True
    )
    f.write(make_width_safe_latex(latex_discovery, add_footnotesize=True))

# ============================================================
# TABLE 9 & FIGURE 10: Multi-Pair Granger Causality (Secondary)
# ============================================================
# NOTE: statsmodels test_causality(caused, causing) tests whether
# the SECOND argument Granger-causes the FIRST. Labels below name
# the direction as: CAUSING → CAUSED.
granger_pairs = [
    # ('caused', 'causing', 'causing → caused')
    ('kraken_btcusd',    'kraken_btcusdc',  'BTC/USDC → BTC/USD (Kraken)'),   # does USDC cause USD?
    ('kraken_btcusdc',   'kraken_btcusd',   'BTC/USD → BTC/USDC (Kraken)'),   # does USD cause USDC?
    ('kraken_btcusd',    'kraken_btcusdt',  'BTC/USDT → BTC/USD (Kraken)'),   # does USDT cause USD?
    ('kraken_btcusdt',   'kraken_btcusd',   'BTC/USD → BTC/USDT (Kraken)'),   # does USD cause USDT?
    ('binance_btcusdt',  'kraken_btcusdt',  'Kraken USDT → Binance USDT'),    # does Kraken cause Binance?
    ('kraken_btcusdt',   'binance_btcusdt', 'Binance USDT → Kraken USDT'),    # does Binance cause Kraken?
    ('coinbase_btcusd',  'kraken_btcusd',   'Kraken USD → Coinbase USD'),      # does Kraken cause Coinbase?
    ('kraken_btcusd',    'coinbase_btcusd', 'Coinbase USD → Kraken USD'),      # does Coinbase cause Kraken?
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
                'p-value': float(g_test.pvalue),
                'Significant': '***' if g_test.pvalue < 0.001 else ('**' if g_test.pvalue < 0.01 else ('*' if g_test.pvalue < 0.05 else ''))
            })
        except Exception as e:
            print(f"  Granger test failed for {label}: {e}")

df_granger = pd.DataFrame(granger_results)
df_granger.to_csv(os.path.join(TABLES_DIR, 'granger_causality.csv'), index=False)

if not df_granger.empty:
    _, qvals, _, _ = multipletests(df_granger['p-value'].values, method='fdr_bh')
    df_granger_fdr = df_granger.copy()
    df_granger_fdr['q-value (BH/FDR)'] = qvals
    df_granger_fdr['Significant FDR'] = df_granger_fdr['q-value (BH/FDR)'].apply(
        lambda q: '***' if q < 0.001 else ('**' if q < 0.01 else ('*' if q < 0.05 else ''))
    )
else:
    df_granger_fdr = df_granger.copy()

df_granger_fdr.to_csv(os.path.join(TABLES_DIR, 'granger_causality_fdr.csv'), index=False)

with open(os.path.join(TABLES_DIR, 'granger_causality.txt'), 'w') as f:
    f.write("Comprehensive Granger Causality Results\n")
    f.write("="*60 + "\n\n")
    if 'p-value' in df_granger.columns:
        df_granger_txt = df_granger.assign(**{
            'p-value': df_granger['p-value'].map(lambda x: f"{x:.6g}")
        })
    else:
        df_granger_txt = df_granger
    f.write(df_granger_txt.to_string(index=False))
    f.write("\n")

# Also produce LaTeX version
with open(os.path.join(TABLES_DIR, 'granger_causality.tex'), 'w') as f:
    f.write(df_granger.to_latex(index=False, caption='Granger Causality Tests (Multi-Pair)',
                                 label='tab:granger', column_format='lrrrr',
                                 float_format='%.6f'))

with open(os.path.join(TABLES_DIR, 'granger_causality_fdr.tex'), 'w') as f:
    f.write(df_granger_fdr.to_latex(
        index=False,
        caption='Granger Causality Tests with BH/FDR Correction (Secondary Evidence)',
        label='tab:granger_fdr',
        float_format='%.6f',
        escape=True
    ))

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
# FIGURE 11: Realized Volatility by Regime
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
cost_variants = [
    ('fee_only_upper', 'Fee-only upper bound', 'cost_fee_only_bps', 'net_fee_only_bps'),
    ('fee_plus_slippage_conservative', 'Fee + slippage conservative bound', 'cost_fee_slippage_bps', 'net_fee_slippage_bps'),
]

for channel_obj in arb_channel_data.values():
    spec = channel_obj['spec']
    channel_df = channel_obj['df']
    for regime, (t0, t1) in regimes.items():
        mask = (channel_df.index >= t0) & (channel_df.index < t1)
        sub = channel_df.loc[mask]
        if sub.empty:
            continue

        for variant_key, variant_label, cost_col, net_col in cost_variants:
            profitable_mask = sub['abs_basis_bps'] > sub[cost_col]
            if profitable_mask.any():
                avg_net_cond = (sub.loc[profitable_mask, 'abs_basis_bps'] - sub.loc[profitable_mask, cost_col]).mean()
            else:
                avg_net_cond = 0.0

            arb_summary.append({
                'channel': spec['label_table'],
                'regime': regime,
                'cost_variant': variant_key,
                'cost_variant_label': variant_label,
                'n_legs': spec['n_legs'],
                'fee_bps_per_leg': FEE_BPS_PER_LEG,
                'fee_component_bps': spec['n_legs'] * FEE_BPS_PER_LEG,
                'mean_abs_bps': sub['abs_basis_bps'].mean(),
                'pct_profitable': profitable_mask.mean() * 100.0,
                'avg_net_cond_bps': avg_net_cond,
                'avg_net_uncond_bps': sub[net_col].mean(),
                'n_minutes': int(len(sub)),
                'execution_assumption': spec['assumption_note'],
            })

df_arb = pd.DataFrame(arb_summary).sort_values(['channel', 'regime', 'cost_variant']).reset_index(drop=True)

# Required Stage 5 consistency check: conservative net <= fee-only net pointwise.
for channel_obj in arb_channel_data.values():
    channel_df = channel_obj['df']
    if (channel_df['net_fee_slippage_bps'] > channel_df['net_fee_only_bps'] + 1e-10).any():
        raise AssertionError("Conservative arbitrage net exceeded fee-only upper bound.")

df_arb.to_csv(os.path.join(TABLES_DIR, 'arbitrage_summary.csv'), index=False)

channel_short_map = {
    'USDC/USD (Kraken, 3-leg triangular)': 'USDC/USD (Kraken, intra)',
    'USDT/USD (Kraken, 3-leg triangular)': 'USDT/USD (Kraken, intra)',
    'Cross BTC/USDT (Binance-Kraken, 2-leg pre-funded)': 'Cross BTC/USDT (Binance-Kraken)',
    'Cross BTC/USD (Coinbase-Kraken, 2-leg pre-funded)': 'Cross BTC/USD (Coinbase-Kraken)',
}
cost_variant_short_map = {
    'Fee-only upper bound': 'Fee-only',
    'Fee + slippage conservative bound': 'Fee + slippage',
}
df_arb_tex = df_arb.copy()
df_arb_tex['channel_short'] = df_arb_tex['channel'].map(channel_short_map).fillna(df_arb_tex['channel'])
df_arb_tex['cost_variant_short'] = df_arb_tex['cost_variant_label'].map(cost_variant_short_map).fillna(df_arb_tex['cost_variant_label'])
df_arb_tex = df_arb_tex[[
    'channel_short', 'regime', 'cost_variant_short',
    'mean_abs_bps', 'pct_profitable', 'avg_net_cond_bps', 'avg_net_uncond_bps'
]].rename(columns={
    'channel_short': 'Channel',
    'regime': 'Regime',
    'cost_variant_short': 'Cost Variant',
    'mean_abs_bps': 'MeanAbs (bps)',
    'pct_profitable': '%Profitable',
    'avg_net_cond_bps': 'AvgNetCond (bps)',
    'avg_net_uncond_bps': 'AvgNetUncond (bps)',
})
with open(os.path.join(TABLES_DIR, 'arbitrage_summary.tex'), 'w') as f:
    latex_arb = df_arb_tex.to_latex(
        index=False,
        caption='Arbitrage Profitability Bounds by Channel and Regime (5 bps per taker leg; 2-leg pre-funded cross-exchange and 3-leg intra-exchange stablecoin channels)',
        label='tab:arb',
        float_format='%.2f',
        escape=True
    )
    f.write(make_width_safe_latex(latex_arb, add_footnotesize=True))

# Required Stage 5 spot-checks: 3 crisis timestamps and 2 non-crisis timestamps.
spot_rows = []

def append_spot(channel_key, ts):
    spec = arb_channel_data[channel_key]['spec']
    row = arb_channel_data[channel_key]['df'].loc[ts]
    spot_rows.append({
        'timestamp_utc': ts.isoformat(),
        'channel': spec['label_table'],
        'regime': assign_regime(ts),
        'n_legs': spec['n_legs'],
        'basis_abs_bps': float(row['abs_basis_bps']),
        'cost_fee_only_bps': float(row['cost_fee_only_bps']),
        'cost_fee_slippage_bps': float(row['cost_fee_slippage_bps']),
        'net_fee_only_bps': float(row['net_fee_only_bps']),
        'net_fee_slippage_bps': float(row['net_fee_slippage_bps']),
    })

usdc_key = 'basis_usdc_kraken'
cbkra_key = 'xbasis_btcusd_coinbase_kraken'

if usdc_key in arb_channel_data and cbkra_key in arb_channel_data:
    usdc_df = arb_channel_data[usdc_key]['df']
    cbkra_df = arb_channel_data[cbkra_key]['df']

    def top_ts(df_in, mask, score_col='net_fee_slippage_bps', rank=0):
        sub = df_in.loc[mask].sort_values(score_col, ascending=False)
        if len(sub) <= rank:
            return None
        return sub.index[rank]

    usdc_crisis_mask = (usdc_df.index >= svb_start) & (usdc_df.index < svb_end)
    cbkra_crisis_mask = (cbkra_df.index >= svb_start) & (cbkra_df.index < svb_end)
    usdc_pre_mask = usdc_df.index < svb_start
    cbkra_post_mask = cbkra_df.index >= svb_end

    ts_candidates = [
        (usdc_key, top_ts(usdc_df, usdc_crisis_mask, rank=0)),   # crisis
        (usdc_key, top_ts(usdc_df, usdc_crisis_mask, rank=1)),   # crisis
        (cbkra_key, top_ts(cbkra_df, cbkra_crisis_mask, rank=0)),# crisis
        (usdc_key, top_ts(usdc_df, usdc_pre_mask, rank=0)),      # outside crisis (pre)
        (cbkra_key, top_ts(cbkra_df, cbkra_post_mask, rank=0)),  # outside crisis (post)
    ]

    if all(ts is not None for _, ts in ts_candidates):
        for channel_key, ts in ts_candidates:
            append_spot(channel_key, ts)
    else:
        print("WARNING: Could not construct full 5-point arbitrage spot-check sample.")
else:
    print("WARNING: Headline channels missing for arbitrage spot checks.")

df_spot = pd.DataFrame(spot_rows)
df_spot.to_csv(os.path.join(TABLES_DIR, 'arbitrage_spotcheck.csv'), index=False)
if not df_spot.empty:
    if (df_spot['net_fee_slippage_bps'] > df_spot['net_fee_only_bps'] + 1e-10).any():
        raise AssertionError("Spot-check monotonicity failed: conservative net exceeded fee-only net.")
    print("\nArbitrage spot-checks (bps):")
    print(df_spot.to_string(index=False))
else:
    print("\nArbitrage spot-checks unavailable (empty sample).")

print("All analysis and figures completed successfully.")
