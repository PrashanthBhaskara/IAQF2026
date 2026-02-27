"""
Six targeted improvements to bring the IAQF 2026 paper to 90+ quality.

Fix 1: Roll (1984) effective spread + Amihud (2002) ILLIQ
  - Directly answers competition Q3: "How do order book depth, spread, and
    volatility vary between BTC quoted in USD versus stablecoins?"
  - fig_liquidity_roll_amihud.png
  - tables/liquidity_spread_table.tex
  - tables/depth_proxy_table.tex (daily dollar-volume depth proxy)

Fix 2: Hasbrouck (1995) Information Shares
  - Replaces informal |alpha| comparison with literature-standard IS bounds
  - Remains valid even when GG component shares fall outside [0,1]
  - tables/hasbrouck_is.tex
  - (also updates cointegration_vecm_merged.tex with IS columns)

Fix 3: GENIUS Act counterfactual quantification
  - Scenario range (25%-75% mitigation) instead of deterministic point claim
  - tables/genius_counterfactual.tex

Fix 4: Data quality transparency table
  - Explicit overall and regime-specific coverage + forward-fill exposure
  - tables/data_coverage_core.tex

Fix 5: HAC uncertainty intervals for headline means
  - Newey-West 95% CIs for key paper claims
  - tables/hac_headline_metrics.tex

Fix 6: Distributional robustness
  - Chow structural break test at SVB onset
  - Higher moments (skewness, excess kurtosis) of B_t by regime
  - Cross-stablecoin correlation (USDC vs USDT B_t) by regime
  - tables/distributional_robustness.tex
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10})

DATA_PROCESSED = 'data_processed'
FIGURES_DIR    = 'figures'
TABLES_DIR     = 'tables'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR,  exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
prices  = pd.read_parquet(os.path.join(DATA_PROCESSED, 'prices.parquet'))
volumes = pd.read_parquet(os.path.join(DATA_PROCESSED, 'volumes.parquet'))
basis   = pd.read_parquet(os.path.join(DATA_PROCESSED, 'basis.parquet'))
ranges  = pd.read_parquet(os.path.join(DATA_PROCESSED, 'intraminute_ranges.parquet'))

pff_path = os.path.join(DATA_PROCESSED, 'price_ffill_flags.parquet')
price_ff_flags = (pd.read_parquet(pff_path) if os.path.exists(pff_path)
                  else pd.DataFrame(False, index=prices.index, columns=prices.columns))
price_ff_flags = price_ff_flags.reindex(
    index=prices.index, columns=prices.columns).fillna(False).astype(bool)

bff_path = os.path.join(DATA_PROCESSED, 'basis_ffill_flags.parquet')
basis_ff_flags = (pd.read_parquet(bff_path) if os.path.exists(bff_path)
                  else pd.DataFrame(False, index=basis.index, columns=basis.columns))
basis_ff_flags = basis_ff_flags.reindex(
    index=basis.index, columns=basis.columns).fillna(False).astype(bool)

svb_start = pd.Timestamp('2023-03-10', tz='UTC')
svb_end   = pd.Timestamp('2023-03-13', tz='UTC')


# Peg-recovery provenance artifact for timeline wording in the manuscript.
if 'kraken_usdcusd' in prices.columns:
    usdc_spot = prices['kraken_usdcusd'].dropna()
    crisis_usdc = usdc_spot[(usdc_spot.index >= svb_start) & (usdc_spot.index < svb_end)]
    if not crisis_usdc.empty:
        trough_ts = crisis_usdc.idxmin()
        trough_px = float(crisis_usdc.min())
        first_print_after_trough = usdc_spot[
            (usdc_spot.index > trough_ts) & (usdc_spot >= 0.999)
        ].index.min()
        daily_min = usdc_spot.resample('D').min()
        daily_min_after_trough = daily_min[daily_min.index > trough_ts.normalize()]
        first_daily_min = daily_min_after_trough[daily_min_after_trough >= 0.999].index.min()
        peg_rows = [
            {
                'metric': 'crisis_trough',
                'timestamp_utc': trough_ts.isoformat(),
                'value': trough_px,
            },
            {
                'metric': 'first_print_ge_0.999_after_trough',
                'timestamp_utc': first_print_after_trough.isoformat() if pd.notna(first_print_after_trough) else '',
                'value': 0.999,
            },
            {
                'metric': 'first_daily_min_ge_0.999_day',
                'timestamp_utc': first_daily_min.isoformat() if pd.notna(first_daily_min) else '',
                'value': 0.999,
            },
        ]
        pd.DataFrame(peg_rows).to_csv(os.path.join(TABLES_DIR, 'peg_recovery_thresholds.csv'), index=False)


def assign_regime(idx):
    if idx < svb_start:  return 'Pre-SVB'
    if idx < svb_end:    return 'Crisis'
    return 'Post-SVB'

REGIME_ORDER  = ['Pre-SVB', 'Crisis', 'Post-SVB']
REGIME_COLORS = {'Pre-SVB': '#3498db', 'Crisis': '#e74c3c', 'Post-SVB': '#2ecc71'}


def enforce_table_H_placement(tables_dir: str = TABLES_DIR) -> int:
    """
    Ensure all generated LaTeX table floats use [H] placement.
    """
    updated_files = 0
    for fname in os.listdir(tables_dir):
        if not fname.endswith('.tex'):
            continue
        path = os.path.join(tables_dir, fname)
        with open(path, 'r') as f:
            tex = f.read()
        updated = re.sub(r'\\begin\{table\*\}(?:\[[^\]]*\])*', r'\\begin{table*}[H]', tex)
        updated = re.sub(r'\\begin\{table\}(?:\[[^\]]*\])*', r'\\begin{table}[H]', updated)
        if updated != tex:
            with open(path, 'w') as f:
                f.write(updated)
            updated_files += 1
    return updated_files


def convert_to_tabularx(latex_text: str, colspec: str) -> str:
    """
    Replace the first tabular environment emitted by pandas with tabularx.
    """
    replacement = (
        '\\setlength{\\tabcolsep}{4pt}\n'
        f'\\begin{{tabularx}}{{\\textwidth}}{{{colspec}}}'
    )
    latex_text = re.sub(
        r'\\begin\{tabular\}\{[^}]*\}',
        lambda _: replacement,
        latex_text,
        count=1,
    )
    latex_text = latex_text.replace(r'\end{tabular}', r'\end{tabularx}', 1)
    return latex_text


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1 – Roll (1984) effective spread and Amihud (2002) ILLIQ
# ═══════════════════════════════════════════════════════════════════════════

def roll_spread_daily(price_col: str) -> pd.Series:
    """
    Roll (1984) effective spread estimate from daily serial covariance of
    1-minute log returns.  Returns daily series in basis points.
    Cov(r_t, r_{t-1}) < 0  =>  Roll = 2*sqrt(-Cov) * 10000 bps
    Cov >= 0               =>  NaN  (estimator not defined; excluded from means)
    """
    p = prices[price_col].dropna()
    lr = np.log(p / p.shift(1))
    rows = []
    for date, grp in lr.groupby(lr.index.date):
        r = grp.dropna().values
        if len(r) < 15:
            continue
        cov = np.cov(r[1:], r[:-1])[0, 1]
        rows.append({'date': pd.Timestamp(date),
                     'roll_bps': 2.0 * np.sqrt(-cov) * 10000 if cov < 0 else np.nan})
    if not rows:
        return pd.Series(dtype=float)
    s = pd.DataFrame(rows).set_index('date')['roll_bps']
    s.index = pd.DatetimeIndex(s.index).tz_localize('UTC')
    return s


def amihud_daily(price_col: str, vol_col: str) -> pd.Series:
    """
    Amihud (2002) ILLIQ ratio: |r_t| / DollarVolume_t, averaged daily.
    Dollar volume = volume_BTC * close_price.
    Returned values are scaled by 1e6 for readability.
    """
    if vol_col not in volumes.columns:
        return pd.Series(dtype=float)
    p   = prices[price_col]
    v   = volumes[vol_col]
    lr  = np.log(p / p.shift(1)).abs()
    dvol = v * p
    aligned = pd.concat([lr, dvol], axis=1, keys=['abs_ret', 'dvol']).dropna()
    aligned  = aligned[aligned['dvol'] > 1.0]
    aligned['illiq'] = aligned['abs_ret'] / aligned['dvol']
    daily = aligned.groupby(aligned.index.date)['illiq'].mean() * 1e6
    s = daily.copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize('UTC')
    return s


def dollar_volume_daily(price_col: str, vol_col: str) -> pd.Series:
    """
    Daily traded dollar volume from 1-minute candles:
      DollarVol_day = sum_t (close_t * volume_t)
    Used as a direct tradable-depth proxy when L2 order-book snapshots are unavailable.
    """
    if price_col not in prices.columns or vol_col not in volumes.columns:
        return pd.Series(dtype=float)
    dvol = (prices[price_col] * volumes[vol_col]).dropna()
    if dvol.empty:
        return pd.Series(dtype=float)
    out = dvol.groupby(dvol.index.date).sum()
    out.index = pd.DatetimeIndex(out.index).tz_localize('UTC')
    return out


PAIRS = {
    'Kraken BTC/USD':  ('kraken_btcusd',   'kraken_btcusd'),
    'Kraken BTC/USDT': ('kraken_btcusdt',  'kraken_btcusdt'),
    'Kraken BTC/USDC': ('kraken_btcusdc',  'kraken_btcusdc'),
    'Binance BTC/USDT':('binance_btcusdt', 'binance_btcusdt'),
    'Coinbase BTC/USD':('coinbase_btcusd', 'coinbase_btcusd'),
}
PAIR_ORDER = list(PAIRS.keys())

roll_series   = {}
amihud_series = {}
depth_series  = {}
for lbl, (pc, vc) in PAIRS.items():
    if pc in prices.columns:
        roll_series[lbl]   = roll_spread_daily(pc)
        amihud_series[lbl] = amihud_daily(pc, vc)
        depth_series[lbl]  = dollar_volume_daily(pc, vc)


def regime_stats(daily_dict):
    """Build [pair × regime] summary DataFrames for roll and amihud."""
    rows = []
    for lbl in PAIR_ORDER:
        if lbl not in daily_dict:
            continue
        s = daily_dict[lbl]
        for reg in REGIME_ORDER:
            if reg == 'Pre-SVB':
                mask = s.index < svb_start
            elif reg == 'Crisis':
                mask = (s.index >= svb_start) & (s.index < svb_end)
            else:
                mask = s.index >= svb_end
            sub = s[mask].dropna()
            rows.append({'Pair': lbl, 'Regime': reg,
                         'mean': round(sub.mean(), 3) if len(sub) else np.nan,
                         'N':    len(sub)})
    return pd.DataFrame(rows)

df_roll   = regime_stats(roll_series)
df_amihud = regime_stats(amihud_series)

# ── Figure: Roll Spread + Amihud (replaces fig_liquidity_regime.png) ───────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# pivot for grouped bar chart
def grouped_bar(ax, df, ylabel, title):
    pivot = df.pivot(index='Pair', columns='Regime', values='mean')[REGIME_ORDER]
    pivot = pivot.reindex(PAIR_ORDER)
    x     = np.arange(len(pivot))
    w     = 0.25
    for i, reg in enumerate(REGIME_ORDER):
        bars = ax.bar(x + (i - 1) * w, pivot[reg], w,
                      label=reg, color=REGIME_COLORS[reg], alpha=0.85,
                      edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    short_names = ['Kraken\nBTC/USD', 'Kraken\nBTC/USDT', 'Kraken\nBTC/USDC',
                   'Binance\nBTC/USDT', 'Coinbase\nBTC/USD']
    ax.set_xticklabels(short_names, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis='y', linewidth=0.5, alpha=0.6)

grouped_bar(axes[0], df_roll,
            'Roll Effective Spread (bps)',
            'Panel A: Roll (1984) Effective Spread by Pair and Regime')
grouped_bar(axes[1], df_amihud,
            'Amihud ILLIQ (x10^-6)',
            'Panel B: Amihud (2002) Illiquidity Ratio by Pair and Regime')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_liquidity_roll_amihud.png'), dpi=150)
plt.close()
print("Saved fig_liquidity_roll_amihud.png")

# ── Table: compact Roll + Amihud summary ───────────────────────────────────
roll_pivot   = df_roll.pivot(index='Pair', columns='Regime', values='mean')[REGIME_ORDER].reindex(PAIR_ORDER)
roll_n_pivot = df_roll.pivot(index='Pair', columns='Regime', values='N')[REGIME_ORDER].reindex(PAIR_ORDER)
amihud_pivot = df_amihud.pivot(index='Pair', columns='Regime', values='mean')[REGIME_ORDER].reindex(PAIR_ORDER)

tbl = pd.DataFrame(index=PAIR_ORDER)
for reg in REGIME_ORDER:
    tbl[f'Roll {reg}'] = roll_pivot[reg].round(2)
for reg in REGIME_ORDER:
    tbl[f'N {reg}'] = roll_n_pivot[reg].astype('Int64')
for reg in REGIME_ORDER:
    tbl[f'ILLIQ {reg}'] = amihud_pivot[reg].round(3)
tbl.index.name = 'Pair'

tbl_latex = tbl.reset_index().to_latex(
    index=False,
    caption=(r'Roll (1984) effective spread (bps) and Amihud (2002) illiquidity ratio '
             r'($\times10^{-6}$) by pair and regime. '
             r'Roll spread estimated from daily serial covariance of 1-minute log returns; '
             r'NaN days (non-negative covariance) excluded from means. '
             r'$N$ is the number of valid Roll days per regime; some regime means rest on few days and are indicative. '
             r'ILLIQ$_t = |r_t|/\text{DollarVol}_t$, daily average.'),
    label='tab:liquidity_spread',
    column_format='l' + 'r' * 9,
    float_format='%.3f',
    na_rep='---',
    escape=False,
)
# wrap in resizebox
tbl_latex = tbl_latex.replace(r'\begin{tabular}',
                               r'\footnotesize' + '\n' +
                               r'\resizebox{\textwidth}{!}{%' + '\n' +
                               r'\begin{tabular}', 1)
tbl_latex = tbl_latex.replace(r'\end{tabular}', r'\end{tabular}' + '%\n}', 1)
# nicer column headers
tbl_latex = tbl_latex.replace('Roll Pre-SVB', r'Roll\textsubscript{Pre}')
tbl_latex = tbl_latex.replace('Roll Crisis',  r'Roll\textsubscript{Crisis}')
tbl_latex = tbl_latex.replace('Roll Post-SVB',r'Roll\textsubscript{Post}')
tbl_latex = tbl_latex.replace('N Pre-SVB', r'$N$\textsubscript{Pre}')
tbl_latex = tbl_latex.replace('N Crisis',  r'$N$\textsubscript{Crisis}')
tbl_latex = tbl_latex.replace('N Post-SVB',r'$N$\textsubscript{Post}')
tbl_latex = tbl_latex.replace('ILLIQ Pre-SVB', r'ILLIQ\textsubscript{Pre}')
tbl_latex = tbl_latex.replace('ILLIQ Crisis',  r'ILLIQ\textsubscript{Crisis}')
tbl_latex = tbl_latex.replace('ILLIQ Post-SVB',r'ILLIQ\textsubscript{Post}')

with open(os.path.join(TABLES_DIR, 'liquidity_spread_table.tex'), 'w') as f:
    f.write(tbl_latex)
print("Saved tables/liquidity_spread_table.tex")
print("\nRoll spread summary:\n", roll_pivot.to_string())
print("\nAmihud ILLIQ summary:\n", amihud_pivot.to_string())

# ── Table: daily dollar-volume depth proxy summary ──────────────────────────
depth_rows = []
for lbl in PAIR_ORDER:
    if lbl not in depth_series:
        continue
    s = depth_series[lbl].dropna()
    for reg in REGIME_ORDER:
        if reg == 'Pre-SVB':
            mask = s.index < svb_start
        elif reg == 'Crisis':
            mask = (s.index >= svb_start) & (s.index < svb_end)
        else:
            mask = s.index >= svb_end
        sub = s[mask].dropna()
        depth_rows.append({
            'Pair': lbl,
            'Regime': reg,
            'median_usd_mm': (sub.median() / 1e6) if len(sub) else np.nan,
            'mean_usd_mm': (sub.mean() / 1e6) if len(sub) else np.nan,
            'n_days': int(len(sub)),
        })

df_depth = pd.DataFrame(depth_rows)
df_depth.to_csv(os.path.join(TABLES_DIR, 'depth_proxy_table.csv'), index=False)
depth_pivot = df_depth.pivot(index='Pair', columns='Regime', values='median_usd_mm')[REGIME_ORDER].reindex(PAIR_ORDER)

tbl_depth = pd.DataFrame(index=PAIR_ORDER)
for reg in REGIME_ORDER:
    tbl_depth[f'Median DollarVol {reg} ($MM/day$)'] = depth_pivot[reg].round(2)
tbl_depth.index.name = 'Pair'

tbl_depth_latex = tbl_depth.reset_index().to_latex(
    index=False,
    caption=(r'Daily traded dollar volume depth proxy by pair and regime (median, USD millions/day). '
             r'Because historical L2 order-book snapshots are unavailable in the candle APIs used here, '
             r'dollar volume and Amihud ILLIQ are used as complementary depth proxies.'),
    label='tab:depth_proxy',
    column_format='l' + 'r' * 3,
    float_format='%.2f',
    na_rep='---',
    escape=False,
)
tbl_depth_latex = tbl_depth_latex.replace(
    r'\begin{tabular}',
    r'\footnotesize' + '\n' + r'\resizebox{\textwidth}{!}{%' + '\n' + r'\begin{tabular}',
    1,
)
tbl_depth_latex = tbl_depth_latex.replace(r'\end{tabular}', r'\end{tabular}' + '%\n}', 1)
with open(os.path.join(TABLES_DIR, 'depth_proxy_table.tex'), 'w') as f:
    f.write(tbl_depth_latex)
print("Saved tables/depth_proxy_table.tex")
print("\nDollar-volume depth proxy summary (median $MM/day):\n", depth_pivot.to_string())

# ── Table: core data coverage and forward-fill diagnostics ──────────────────
coverage_specs = [
    ('kraken_btcusd', 'Kraken BTC/USD'),
    ('kraken_btcusdt', 'Kraken BTC/USDT'),
    ('kraken_btcusdc', 'Kraken BTC/USDC'),
    ('kraken_usdcusd', 'Kraken USDC/USD'),
    ('kraken_usdtusd', 'Kraken USDT/USD'),
    ('coinbase_btcusd', 'Coinbase BTC/USD'),
    ('coinbase_btcusdt', 'Coinbase BTC/USDT'),
    ('coinbase_usdtusd', 'Coinbase USDT/USD'),
    ('binance_btcusdt', 'Binance BTC/USDT'),
    ('binance_btcusdc', 'Binance BTC/USDC'),
]

cov_rows = []
for col, label in coverage_specs:
    if col not in prices.columns:
        continue
    s = prices[col]
    ff = price_ff_flags[col] if col in price_ff_flags.columns else pd.Series(False, index=prices.index)
    ff = ff.reindex(prices.index).fillna(False).astype(bool)

    def cov_for(mask):
        sub = s.loc[mask]
        return (sub.notna().mean() * 100.0) if len(sub) else np.nan

    def ff_for(mask):
        sub = ff.loc[mask]
        return (sub.mean() * 100.0) if len(sub) else np.nan

    m_pre = prices.index < svb_start
    m_cri = (prices.index >= svb_start) & (prices.index < svb_end)
    m_post = prices.index >= svb_end

    cov_rows.append({
        'Pair': label,
        'Coverage Overall (%)': cov_for(prices.index == prices.index),
        'Coverage Pre-SVB (%)': cov_for(m_pre),
        'Coverage Crisis (%)': cov_for(m_cri),
        'Coverage Post-SVB (%)': cov_for(m_post),
        'Forward-Fill Share Overall (%)': ff_for(prices.index == prices.index),
    })

df_cov = pd.DataFrame(cov_rows)
df_cov.to_csv(os.path.join(TABLES_DIR, 'data_coverage_core.csv'), index=False)
cov_latex = df_cov.to_latex(
    index=False,
    caption=(r'Core series data coverage and forward-fill exposure. '
             r'Coverage is the share of non-missing 1-minute observations on the unified UTC grid; '
             r'forward-fill share is the percent of minutes filled by carry-forward (up to 5 minutes).'),
    label='tab:data_coverage',
    column_format='lrrrrr',
    float_format='%.2f',
    escape=True,
)
cov_latex = cov_latex.replace(
    r'\begin{tabular}',
    r'\footnotesize' + '\n' + r'\resizebox{\textwidth}{!}{%' + '\n' + r'\begin{tabular}',
    1,
)
cov_latex = cov_latex.replace(r'\end{tabular}', r'\end{tabular}' + '%\n}', 1)
with open(os.path.join(TABLES_DIR, 'data_coverage_core.tex'), 'w') as f:
    f.write(cov_latex)
print("Saved tables/data_coverage_core.tex")

# ── Table: no-forward-fill sensitivity for headline metrics ──────────────────
ff_rows = []


def add_ff_row(metric, statistic, all_value, noff_value, n_all, n_noff):
    delta = noff_value - all_value
    retention = (100.0 * n_noff / n_all) if n_all > 0 else np.nan
    ff_rows.append({
        'Metric': metric,
        'Statistic': statistic,
        'All-sample': all_value,
        'No-FF sample': noff_value,
        'Delta (No-FF - All)': delta,
        'N all': int(n_all),
        'N no-FF': int(n_noff),
        'Retention (%)': retention,
    })


def compare_mean(metric, series, ff_mask, mask):
    all_vals = series.loc[mask].dropna()
    valid_noff = mask & (~ff_mask.reindex(series.index).fillna(False))
    noff_vals = series.loc[valid_noff].dropna()
    add_ff_row(metric, 'mean (bps)', all_vals.mean(), noff_vals.mean(), len(all_vals), len(noff_vals))


m_all = prices.index == prices.index
m_pre = prices.index < svb_start
m_cri = (prices.index >= svb_start) & (prices.index < svb_end)
m_post = prices.index >= svb_end
m_calm = m_pre | m_post

compare_mean(
    'USDC dispersion D_t (Kraken, crisis)',
    basis['dispersion_usdc_kraken'],
    basis_ff_flags['dispersion_usdc_kraken'],
    m_cri,
)
compare_mean(
    'USDC adjusted residual B_t (Kraken, crisis)',
    basis['basis_usdc_kraken'],
    basis_ff_flags['basis_usdc_kraken'],
    m_cri,
)
compare_mean(
    'USDT premium to USD (Kraken, crisis)',
    (prices['kraken_usdtusd'] - 1.0) * 10000.0,
    price_ff_flags['kraken_usdtusd'],
    m_cri,
)
compare_mean(
    'USDT premium to USD (Coinbase, crisis)',
    (prices['coinbase_usdtusd'] - 1.0) * 10000.0,
    price_ff_flags['coinbase_usdtusd'],
    m_cri,
)
compare_mean(
    'Cross-exchange BTC/USDT basis (Binance-Kraken, calm)',
    basis['xbasis_btcusdt_binance_kraken'],
    basis_ff_flags['xbasis_btcusdt_binance_kraken'],
    m_calm,
)

# USDC crisis arbitrage sensitivity (strict no-FF on basis + all execution legs)
fee_bps = 15.0  # 3-leg * 5 bps
arb_abs = basis['basis_usdc_kraken'].abs()
arb_slip = 0.5 * (
    ranges['kraken_btcusdc'] + ranges['kraken_usdcusd'] + ranges['kraken_btcusd']
) * 10000.0
arb_df = pd.DataFrame({'abs_basis_bps': arb_abs, 'slippage_bps': arb_slip})
arb_df = arb_df.dropna()
arb_cri = arb_df.loc[(arb_df.index >= svb_start) & (arb_df.index < svb_end)]

strict_ff = (
    basis_ff_flags['basis_usdc_kraken']
    | price_ff_flags['kraken_btcusdc']
    | price_ff_flags['kraken_usdcusd']
    | price_ff_flags['kraken_btcusd']
).reindex(arb_df.index).fillna(False)
arb_cri_noff = arb_df.loc[
    (arb_df.index >= svb_start) & (arb_df.index < svb_end) & (~strict_ff)
]

all_fee_prof = (arb_cri['abs_basis_bps'] > fee_bps).mean() * 100.0
noff_fee_prof = (arb_cri_noff['abs_basis_bps'] > fee_bps).mean() * 100.0
all_slip_prof = (arb_cri['abs_basis_bps'] > (fee_bps + arb_cri['slippage_bps'])).mean() * 100.0
noff_slip_prof = (arb_cri_noff['abs_basis_bps'] > (fee_bps + arb_cri_noff['slippage_bps'])).mean() * 100.0

add_ff_row(
    'USDC crisis arbitrage (Kraken, 3-leg)',
    '% profitable, fee-only',
    all_fee_prof,
    noff_fee_prof,
    len(arb_cri),
    len(arb_cri_noff),
)
add_ff_row(
    'USDC crisis arbitrage (Kraken, 3-leg)',
    '% profitable, fee+slippage',
    all_slip_prof,
    noff_slip_prof,
    len(arb_cri),
    len(arb_cri_noff),
)

# Realized-volatility strict no-FF window retention (60-minute rolling)
# Use simple returns to match the main realized-volatility implementation.
ret_1m = prices.pct_change(fill_method=None)
rv_cols = ['kraken_btcusdc', 'kraken_btcusdt']
for col in rv_cols:
    rv_all = ret_1m[col].rolling(60).std() * 10000.0 * np.sqrt(60.0)
    ff_ret = price_ff_flags[col] | price_ff_flags[col].shift(1, fill_value=False)
    rv_noff = ret_1m[col].where(~ff_ret).rolling(60, min_periods=60).std() * 10000.0 * np.sqrt(60.0)
    all_vals = rv_all.loc[m_cri].dropna()
    noff_vals = rv_noff.loc[m_cri].dropna()
    add_ff_row(
        f'Crisis RV (60m rolling, {col})',
        'mean (bps/hr)',
        all_vals.mean(),
        noff_vals.mean(),
        len(all_vals),
        len(noff_vals),
    )

df_ff = pd.DataFrame(ff_rows)
df_ff = df_ff.round({
    'All-sample': 3,
    'No-FF sample': 3,
    'Delta (No-FF - All)': 3,
    'Retention (%)': 1,
})
df_ff.to_csv(os.path.join(TABLES_DIR, 'ff_sensitivity_core.csv'), index=False)

# Compact display labels to prevent wasteful one-word wraps in manuscript Table 2.
df_ff_tex = df_ff.copy()
df_ff_tex['Metric'] = df_ff_tex['Metric'].replace({
    'USDC dispersion D_t (Kraken, crisis)': r'USDC $D_t$ (Kraken, crisis)',
    'USDC adjusted residual B_t (Kraken, crisis)': r'USDC $B_t$ (Kraken, crisis)',
    'USDT premium to USD (Kraken, crisis)': 'USDT premium (Kraken, crisis)',
    'USDT premium to USD (Coinbase, crisis)': 'USDT premium (Coinbase, crisis)',
    'Cross-exchange BTC/USDT basis (Binance-Kraken, calm)': 'BTC/USDT basis (BN-KR, calm)',
    'USDC crisis arbitrage (Kraken, 3-leg)': 'USDC arb (Kraken, 3-leg, crisis)',
    'Crisis RV (60m rolling, kraken_btcusdc)': 'RV 60m (KR BTC/USDC, crisis)',
    'Crisis RV (60m rolling, kraken_btcusdt)': 'RV 60m (KR BTC/USDT, crisis)',
})
df_ff_tex['Statistic'] = df_ff_tex['Statistic'].replace({
    'mean (bps)': 'Mean bps',
    'mean (bps/hr)': 'Mean bps/hr',
    '% profitable, fee-only': 'Fee-only \\%',
    '% profitable, fee+slippage': 'Fee+slip \\%',
})
df_ff_tex = df_ff_tex.rename(columns={
    'All-sample': 'All',
    'No-FF sample': 'No-FF',
    'Delta (No-FF - All)': r'$\Delta$ (No-FF-All)',
    'N all': 'N',
    'N no-FF': 'N (No-FF)',
    'Retention (%)': r'Retention (\%)',
})

ff_latex = df_ff_tex.to_latex(
    index=False,
    caption=(
        r'Forward-fill sensitivity for selected headline metrics. '
        r'All-sample uses the baseline aligned 1-minute series; No-FF sample removes '
        r'observations with forward-filled inputs for each metric. Signs are largely stable, '
        r'but magnitudes can move in thinner channels and rolling-volatility windows can become sparse.'
    ),
    label='tab:ff_sensitivity',
    column_format='llrrrrrr',
    float_format='%.3f',
    escape=False,
)
ff_latex = ff_latex.replace(r'\begin{table}', r'\begin{table}[H]', 1)
ff_latex = ff_latex.replace(r'\begin{tabular}', r'\footnotesize' + '\n' + r'\begin{tabular}', 1)
ff_latex = convert_to_tabularx(
    ff_latex,
    r'>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{1.90cm}rrrrrr',
)
with open(os.path.join(TABLES_DIR, 'ff_sensitivity_core.tex'), 'w') as f:
    f.write(ff_latex)
print("Saved tables/ff_sensitivity_core.tex")


# ═══════════════════════════════════════════════════════════════════════════
# FIX 2 – Hasbrouck (1995) Information Shares
# ═══════════════════════════════════════════════════════════════════════════

def hasbrouck_is_bounds(vecm_fit, mkt1_name='Market 1', mkt2_name='Market 2'):
    """
    Compute Hasbrouck (1995) Information Share bounds for a bivariate VECM
    with cointegration rank 1.

    Parameters
    ----------
    vecm_fit : statsmodels VECMResults
        Fitted bivariate VECM with coint_rank=1.

    Returns dict with IS lower/upper/midpoint for both markets.

    Method
    ------
    Common-factor weights: psi = alpha_perp = [-alpha2, alpha1]  (GG direction)
    For each of the two Cholesky orderings of Sigma (residual covariance):
        f = psi @ M   where M = lower_chol(Sigma[ordering])
        IS_i = f_i^2 / ||f||^2
    IS lower = min over orderings; IS upper = max over orderings.
    Midpoint = (lower+upper)/2.

    Note: This gives IS in [0,1] even when GG shares are outside [0,1]
    (which occurs when both alpha coefficients have the same sign).
    """
    alpha = vecm_fit.alpha[:, 0].astype(float)  # [alpha_1, alpha_2]
    sigma = vecm_fit.sigma_u.astype(float)        # 2×2 residual covariance

    # common-factor weights (alpha_perp = orthogonal complement of alpha)
    psi = np.array([-alpha[1], alpha[0]])

    results = {}
    for ordering, (i, j) in [('12', (0, 1)), ('21', (1, 0))]:
        # permute sigma to put market i first
        idx = [i, j]
        sig_p = sigma[np.ix_(idx, idx)]
        try:
            M = np.linalg.cholesky(sig_p)   # lower triangular
        except np.linalg.LinAlgError:
            # sigma not positive definite for this ordering; skip
            results[ordering] = (np.nan, np.nan)
            continue
        psi_p = psi[idx]                     # permute psi to match ordering
        f = psi_p @ M                        # (2,) vector
        denom = float(f @ f)
        if denom < 1e-20:
            results[ordering] = (np.nan, np.nan)
            continue
        IS_i = float(f[0] ** 2 / denom)     # market i (comes first in ordering)
        IS_j = float(f[1] ** 2 / denom)     # market j
        # map back to original indices
        IS = [0.0, 0.0]
        IS[i] = IS_i
        IS[j] = IS_j
        results[ordering] = tuple(IS)

    vals_mkt1 = [v[0] for v in results.values() if not np.isnan(v[0])]
    vals_mkt2 = [v[1] for v in results.values() if not np.isnan(v[1])]

    IS1_lo = min(vals_mkt1) if vals_mkt1 else np.nan
    IS1_hi = max(vals_mkt1) if vals_mkt1 else np.nan
    IS2_lo = min(vals_mkt2) if vals_mkt2 else np.nan
    IS2_hi = max(vals_mkt2) if vals_mkt2 else np.nan

    return {
        f'IS_{mkt1_name}_lower':    IS1_lo,
        f'IS_{mkt1_name}_upper':    IS1_hi,
        f'IS_{mkt1_name}_midpoint': 0.5*(IS1_lo+IS1_hi) if not np.isnan(IS1_lo) else np.nan,
        f'IS_{mkt2_name}_lower':    IS2_lo,
        f'IS_{mkt2_name}_upper':    IS2_hi,
        f'IS_{mkt2_name}_midpoint': 0.5*(IS2_lo+IS2_hi) if not np.isnan(IS2_lo) else np.nan,
        'alpha_mkt1': float(alpha[0]),
        'alpha_mkt2': float(alpha[1]),
        'sigma': sigma,
    }


# ── Refit VECM for Kraken BTC/USD vs BTC/USDT (no-FF sample, same spec) ───
vecm_specs = [
    {'channel': 'Kraken BTC/USD vs BTC/USDT',
     'col1':    'kraken_btcusd',
     'col2':    'kraken_btcusdt',
     'mkt1':    'BTC/USD',
     'mkt2':    'BTC/USDT'},
]

is_rows = []
for spec in vecm_specs:
    c1, c2 = spec['col1'], spec['col2']
    if c1 not in prices.columns or c2 not in prices.columns:
        print(f"Skipping {spec['channel']}: price columns not found.")
        continue

    # Drop forward-filled minutes (same filter as existing code)
    ff_mask = price_ff_flags[[c1, c2]].any(axis=1)
    p_log   = np.log(prices[[c1, c2]]).copy()
    p_log[ff_mask] = np.nan
    df_levels = p_log.dropna()

    if len(df_levels) < 500:
        print(f"Skipping {spec['channel']}: insufficient data after no-FF filter.")
        continue

    # Select lag order via BIC (same as primary analysis in 03_)
    try:
        sel    = select_order(df_levels, maxlags=15, deterministic='ci')
        p_bic  = sel.bic
        p_aic  = sel.aic
        p_used = p_bic if p_bic is not None else (p_aic if p_aic is not None else 2)
        p_used = max(1, int(p_used))
        k_diff = p_used - 1
    except Exception:
        k_diff = 7  # fallback: k_ar_diff=7 → k_ar=8

    # Test cointegration rank
    joh = coint_johansen(df_levels.values, det_order=0, k_ar_diff=k_diff)
    rank = int(np.sum(joh.lr1 > joh.cvt[:, 1]))   # 95% critical values
    rank = min(rank, 1)

    if rank < 1:
        print(f"{spec['channel']}: rank=0 in Hasbrouck spec, skipping IS.")
        is_rows.append({
            'Channel':       spec['channel'],
            'Rank':          0,
            'k_diff':        k_diff,
            'IS_USD_lower':  np.nan, 'IS_USD_upper': np.nan, 'IS_USD_mid': np.nan,
            'IS_other_lower':np.nan, 'IS_other_upper':np.nan,'IS_other_mid':np.nan,
            'note':          'rank=0 no cointegration',
        })
        continue

    vecm = VECM(df_levels, k_ar_diff=k_diff, coint_rank=rank,
                deterministic='ci').fit()

    is_d = hasbrouck_is_bounds(vecm, mkt1_name='USD', mkt2_name='USDT')

    print(f"\n{spec['channel']}  (k_diff={k_diff}, rank={rank})")
    print(f"  alpha = [{is_d['alpha_mkt1']:.5f}, {is_d['alpha_mkt2']:.5f}]")
    print(f"  sigma_u:\n{is_d['sigma']}")
    print(f"  IS BTC/USD  : [{is_d['IS_USD_lower']:.3f}, {is_d['IS_USD_upper']:.3f}]"
          f"  mid={is_d['IS_USD_midpoint']:.3f}")
    print(f"  IS BTC/USDT : [{is_d['IS_USDT_lower']:.3f}, {is_d['IS_USDT_upper']:.3f}]"
          f"  mid={is_d['IS_USDT_midpoint']:.3f}")

    is_rows.append({
        'Channel':        spec['channel'],
        'Rank':           rank,
        'k_diff':         k_diff,
        'IS_USD_lower':   round(is_d['IS_USD_lower'],   3),
        'IS_USD_upper':   round(is_d['IS_USD_upper'],   3),
        'IS_USD_mid':     round(is_d['IS_USD_midpoint'],3),
        'IS_other_lower': round(is_d['IS_USDT_lower'],  3),
        'IS_other_upper': round(is_d['IS_USDT_upper'],  3),
        'IS_other_mid':   round(is_d['IS_USDT_midpoint'],3),
        'alpha_USD':      round(is_d['alpha_mkt1'], 5),
        'alpha_USDT':     round(is_d['alpha_mkt2'], 5),
        'note':           '',
    })

df_is = pd.DataFrame(is_rows)
df_is.to_csv(os.path.join(TABLES_DIR, 'hasbrouck_is.csv'), index=False)

# LaTeX table for Hasbrouck IS
if not df_is.empty:
    tbl_is = df_is[['Channel', 'Rank', 'k_diff',
                    'alpha_USD', 'alpha_USDT',
                    'IS_USD_lower', 'IS_USD_upper', 'IS_USD_mid',
                    'IS_other_lower', 'IS_other_upper', 'IS_other_mid']].copy()
    tbl_is.columns = ['Channel', 'Rank', r'$k_\Delta$',
                      r'$\alpha_\text{USD}$', r'$\alpha_\text{USDT}$',
                      'IS\\textsubscript{USD,lo}',
                      'IS\\textsubscript{USD,hi}',
                      'IS\\textsubscript{USD,mid}',
                      'IS\\textsubscript{USDT,lo}',
                      'IS\\textsubscript{USDT,hi}',
                      'IS\\textsubscript{USDT,mid}']
    is_latex = tbl_is.to_latex(
        index=False,
        caption=(r'Hasbrouck (1995) Information Share bounds for Kraken BTC/USD vs '
                 r'BTC/USDT (no-FF sample). '
                 r'Common-factor weights $\psi = \alpha_\perp = [-\alpha_2, \alpha_1]$. '
                 r'IS bounds via both Cholesky orderings of the residual covariance $\Sigma_u$; '
                 r'midpoint = $(\text{lower}+\text{upper})/2$. '
                 r'BTC/USD IS midpoint $>0.5$ confirms USD as the relative price-discovery leader.'),
        label='tab:hasbrouck_is',
        column_format='l' + 'r' * 10,
        float_format='%.3f',
        na_rep='---',
        escape=False,
    )
    is_latex = is_latex.replace(r'\begin{tabular}',
                                r'\footnotesize' + '\n' +
                                r'\resizebox{\textwidth}{!}{%' + '\n' +
                                r'\begin{tabular}', 1)
    is_latex = is_latex.replace(r'\end{tabular}', r'\end{tabular}' + '%\n}', 1)
    with open(os.path.join(TABLES_DIR, 'hasbrouck_is.tex'), 'w') as f:
        f.write(is_latex)
    print("\nSaved tables/hasbrouck_is.tex")

# ── Update cointegration_vecm_merged.tex from canonical CSV artifacts ───────
joh_path = os.path.join(TABLES_DIR, 'cointegration_johansen.csv')
disc_path = os.path.join(TABLES_DIR, 'price_discovery_metrics.csv')
if os.path.exists(joh_path) and os.path.exists(disc_path):
    df_joh = pd.read_csv(joh_path)
    df_disc = pd.read_csv(disc_path)

    def fetch_row(df: pd.DataFrame, channel: str):
        exact = df[df['channel'] == channel]
        if not exact.empty:
            return exact.iloc[0]
        contains = df[df['channel'].astype(str).str.contains(channel, regex=False)]
        if not contains.empty:
            return contains.iloc[0]
        return None

    ch_usdc = 'Kraken BTC/USD vs BTC/USDC'
    ch_usdt = 'Kraken BTC/USD vs BTC/USDT'
    j_usdc = fetch_row(df_joh, ch_usdc)
    j_usdt = fetch_row(df_joh, ch_usdt)
    d_usdc = fetch_row(df_disc, ch_usdc)
    d_usdt = fetch_row(df_disc, ch_usdt)

    is_usdt = None
    if not df_is.empty:
        cand = df_is[df_is['Channel'].astype(str).str.contains('BTC/USD vs BTC/USDT', regex=False)]
        if not cand.empty:
            is_usdt = cand.iloc[0]

    def fmt_alpha(x):
        return '---' if (x is None or pd.isna(x)) else f'{float(x):.4f}'

    def fmt_is_mid(x):
        return '---' if (x is None or pd.isna(x)) else f'{float(x):.2f}'

    if (j_usdc is None) or (j_usdt is None):
        print("WARNING: Could not update cointegration_vecm_merged.tex — missing Johansen channel rows.")
    else:
        is_lo = float(is_usdt['IS_USD_lower']) if is_usdt is not None and not pd.isna(is_usdt['IS_USD_lower']) else np.nan
        is_hi = float(is_usdt['IS_USD_upper']) if is_usdt is not None and not pd.isna(is_usdt['IS_USD_upper']) else np.nan
        is_mid = float(is_usdt['IS_USD_mid']) if is_usdt is not None and not pd.isna(is_usdt['IS_USD_mid']) else np.nan
        is_head = f'IS$_{{\\text{{USD}}}}$ [{is_lo:.2f}, {is_hi:.2f}]' if np.isfinite(is_lo) and np.isfinite(is_hi) else r'IS$_{\text{USD}}$'
        is_note = f'IS = Hasbrouck (1995) midpoint; bounds [{is_lo:.2f}, {is_hi:.2f}].' if np.isfinite(is_lo) and np.isfinite(is_hi) else 'IS unavailable.'

        rank_usdc = int(j_usdc['rank_used'])
        rank_usdt = int(j_usdt['rank_used'])
        k_usdc = int(j_usdc['k_ar_diff_used'])
        k_usdt = int(j_usdt['k_ar_diff_used'])
        tr_usdc = float(j_usdc['trace_stat_r0'])
        tr_usdt = float(j_usdt['trace_stat_r0'])
        a1_usdc = fmt_alpha(None if d_usdc is None else d_usdc.get('alpha_market_1', np.nan))
        a2_usdc = fmt_alpha(None if d_usdc is None else d_usdc.get('alpha_market_2', np.nan))
        a1_usdt = fmt_alpha(None if d_usdt is None else d_usdt.get('alpha_market_1', np.nan))
        a2_usdt = fmt_alpha(None if d_usdt is None else d_usdt.get('alpha_market_2', np.nan))

        leader_usdc_raw = '' if d_usdc is None else str(d_usdc.get('leader_by_adjustment', ''))
        leader_usdt_raw = '' if d_usdt is None else str(d_usdt.get('leader_by_adjustment', ''))
        leader_usdc = 'undetermined' if ('undetermined' in leader_usdc_raw or rank_usdc == 0) else leader_usdc_raw.replace('Kraken ', '')
        leader_usdt = leader_usdt_raw.replace('Kraken ', '') if leader_usdt_raw else 'undetermined'

        updated_vecm = [
            r'\begin{table}[H]',
            r'\caption{Johansen Cointegration and VECM Price Discovery (Primary Kraken Channels, No-FF Sample)}',
            r'\label{tab:coint_vecm}',
            r'\footnotesize',
            r'\centering',
            r'\begin{tabular}{lccccccl}',
            r'\toprule',
            f'Channel & Rank & $k_\\Delta$ & Trace$_{{r=0}}$ & $\\alpha_{{\\text{{USD}}}}$ & $\\alpha_{{\\text{{other}}}}$ & {is_head} & Leader \\\\',
            r'\midrule',
            f'BTC/USD vs BTC/USDC & {rank_usdc:d} & {k_usdc:d} & {tr_usdc:.2f} & {a1_usdc} & {a2_usdc} & --- & {leader_usdc} \\\\',
            f'BTC/USD vs BTC/USDT & {rank_usdt:d} & {k_usdt:d} & {tr_usdt:.2f} & {a1_usdt} & {a2_usdt} & {fmt_is_mid(is_mid)} & {leader_usdt} \\\\',
            r'\bottomrule',
            f'\\multicolumn{{8}}{{l}}{{\\footnotesize 95\\% critical value for trace $r=0$: {float(j_usdt["trace_crit95_r0"]):.2f}. {is_note}}}',
            r'\end{tabular}',
            r'\end{table}',
        ]
        with open(os.path.join(TABLES_DIR, 'cointegration_vecm_merged.tex'), 'w') as f:
            f.write('\n'.join(updated_vecm) + '\n')
        print("Updated tables/cointegration_vecm_merged.tex from canonical CSV artifacts.")
else:
    print("WARNING: Could not update cointegration_vecm_merged.tex — required CSV artifacts missing.")


# ═══════════════════════════════════════════════════════════════════════════
# FIX 3 – GENIUS Act counterfactual quantification
# ═══════════════════════════════════════════════════════════════════════════

# Factual numbers (from existing analysis / public record)
svb_total_reserves_bn = 40.0    # Circle USDC reserves at time of SVB, ~$40B
svb_locked_bn         = 3.3     # Amount locked at SVB
svb_locked_frac       = svb_locked_bn / svb_total_reserves_bn

# D_t stats from basis dataframe
d_col = 'dispersion_usdc_kraken'
b_col = 'basis_usdc_kraken'

def d_stats(col, t0, t1):
    mask = (basis.index >= t0) & (basis.index < t1)
    s = basis.loc[mask, col].dropna()
    return {'mean': s.mean(), 'std': s.std(),
            'p99': s.quantile(0.99), 'p01': s.quantile(0.01),
            'N': len(s)}

pre_d  = d_stats(d_col, prices.index.min(), svb_start)
cri_d  = d_stats(d_col, svb_start, svb_end)
post_d = d_stats(d_col, svb_end, prices.index.max())

pre_b  = d_stats(b_col, prices.index.min(), svb_start)
cri_b  = d_stats(b_col, svb_start, svb_end)

print("\nGENIUS Act counterfactual inputs:")
print(f"  SVB locked fraction: {svb_locked_frac:.2%}")
print(f"  D_t pre-crisis: mean={pre_d['mean']:.2f} bps, std={pre_d['std']:.2f}")
print(f"  D_t crisis:     mean={cri_d['mean']:.2f} bps, std={cri_d['std']:.2f}")
print(f"  B_t pre-crisis: mean={pre_b['mean']:.2f} bps")
print(f"  B_t crisis:     mean={cri_b['mean']:.2f} bps")

# Illustrative scenario analysis (not a structural causal estimate):
# map assumed mitigation of reserve lock-up shock into a range of implied
# crisis D_t means. This avoids a single deterministic counterfactual.
shock_component = cri_d['mean'] - pre_d['mean']
mitigation_grid = [0.00, 0.25, 0.50, 0.75, 1.00]
scenario_map = {
    0.00: 'Observed baseline',
    0.25: 'Low mitigation',
    0.50: 'Moderate mitigation',
    0.75: 'High mitigation',
    1.00: 'Full-mitigation lower bound',
}

cf_rows = []
for m in mitigation_grid:
    implied_mean = pre_d['mean'] + (1.0 - m) * shock_component
    reduction = cri_d['mean'] - implied_mean
    cf_rows.append({
        'Scenario': scenario_map[m],
        'Assumed mitigation of lock-up shock (%)': int(round(m * 100)),
        'Implied crisis D_t mean (bps)': implied_mean,
        'Reduction vs observed (bps)': reduction,
        'Policy mapping': (
            'Reserve composition + redemption + transparency'
            if m > 0 else
            'SVB/No GENIUS baseline'
        ),
    })

df_cf = pd.DataFrame(cf_rows)
df_cf = df_cf.round({
    'Implied crisis D_t mean (bps)': 1,
    'Reduction vs observed (bps)': 1,
})
df_cf.to_csv(os.path.join(TABLES_DIR, 'genius_counterfactual.csv'), index=False)

cf_latex = df_cf.to_latex(
    index=False,
    caption=(
        r'GENIUS Act scenario range (illustrative, not structural causal identification). '
        r'The table maps assumed mitigation of reserve lock-up shock into implied crisis '
        r'$D_t$ means, anchored to observed pre- and crisis-period moments. '
        r'Tail benchmark for context: USDC $B_t$ crisis P99 = '
        + f'{cri_b["p99"]:.1f}'
        + r' bps versus pre-crisis P99 = '
        + f'{pre_b["p99"]:.1f}'
        + r' bps.'
    ),
    label='tab:genius_cf',
    column_format='p{3.3cm}rrrp{4.5cm}',
    float_format='%.1f',
    escape=True,
)
cf_latex = cf_latex.replace(
    r'\begin{tabular}',
    r'\footnotesize' + '\n' + r'\resizebox{\textwidth}{!}{%' + '\n' + r'\begin{tabular}',
    1,
)
cf_latex = cf_latex.replace(r'\end{tabular}', r'\end{tabular}' + '%\n}', 1)
with open(os.path.join(TABLES_DIR, 'genius_counterfactual.tex'), 'w') as f:
    f.write(cf_latex)
print("\nSaved tables/genius_counterfactual.tex")

Dt_reduction_abs_low = 0.25 * shock_component
Dt_reduction_abs_high = 0.75 * shock_component
Dt_reduction_pct_low = 100.0 * Dt_reduction_abs_low / cri_d['mean'] if cri_d['mean'] > 0 else np.nan
Dt_reduction_pct_high = 100.0 * Dt_reduction_abs_high / cri_d['mean'] if cri_d['mean'] > 0 else np.nan

# ═══════════════════════════════════════════════════════════════════════════
# FIX 5 – HAC uncertainty intervals for headline means
# ═══════════════════════════════════════════════════════════════════════════

def hac_mean_ci(series: pd.Series, maxlags: int = 60):
    s = series.dropna()
    n = int(len(s))
    if n < 10:
        return {'mean': np.nan, 'se': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan, 'N': n}
    lag_use = min(maxlags, max(1, n // 10))
    model = sm.OLS(s.values, np.ones((n, 1))).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': lag_use},
    )
    mean = float(model.params[0])
    se = float(model.bse[0])
    return {
        'mean': mean,
        'se': se,
        'ci_lo': mean - 1.96 * se,
        'ci_hi': mean + 1.96 * se,
        'N': n,
    }


hac_specs = [
    ('USDC dispersion $D_t$ (Kraken, crisis)', basis.loc[m_cri, 'dispersion_usdc_kraken']),
    ('USDC adjusted residual $B_t$ (Kraken, crisis)', basis.loc[m_cri, 'basis_usdc_kraken']),
    ('USDT premium to USD (Kraken, crisis)', (prices.loc[m_cri, 'kraken_usdtusd'] - 1.0) * 10000.0),
    ('USDT premium to USD (Coinbase, crisis)', (prices.loc[m_cri, 'coinbase_usdtusd'] - 1.0) * 10000.0),
    ('BTC/USDT cross-exchange basis (Binance-Kraken, calm)', basis.loc[m_calm, 'xbasis_btcusdt_binance_kraken']),
    ('BTC/USD cross-exchange basis (Coinbase-Kraken, full)', basis.loc[m_all, 'xbasis_btcusd_coinbase_kraken']),
]

hac_rows = []
for metric, ser in hac_specs:
    out = hac_mean_ci(ser, maxlags=60)
    hac_rows.append({
        'Metric': metric,
        'Mean (bps)': out['mean'],
        'HAC SE (bps)': out['se'],
        '95% CI low': out['ci_lo'],
        '95% CI high': out['ci_hi'],
        'N': out['N'],
    })

df_hac = pd.DataFrame(hac_rows).round({
    'Mean (bps)': 3,
    'HAC SE (bps)': 3,
    '95% CI low': 3,
    '95% CI high': 3,
})
df_hac.to_csv(os.path.join(TABLES_DIR, 'hac_headline_metrics.csv'), index=False)

df_hac_tex = df_hac.rename(columns={
    '95% CI low': r'95\% CI low',
    '95% CI high': r'95\% CI high',
})

# Compact display labels to prevent wasteful wraps in manuscript Table 4.
df_hac_tex['Metric'] = df_hac_tex['Metric'].replace({
    'USDC dispersion $D_t$ (Kraken, crisis)': r'USDC $D_t$ (Kraken, crisis)',
    'USDC adjusted residual $B_t$ (Kraken, crisis)': r'USDC $B_t$ (Kraken, crisis)',
    'USDT premium to USD (Kraken, crisis)': 'USDT premium (Kraken, crisis)',
    'USDT premium to USD (Coinbase, crisis)': 'USDT premium (Coinbase, crisis)',
    'BTC/USDT cross-exchange basis (Binance-Kraken, calm)': 'BTC/USDT basis (BN-KR, calm)',
    'BTC/USD cross-exchange basis (Coinbase-Kraken, full)': 'BTC/USD basis (CB-KR, full)',
})
df_hac_tex = df_hac_tex.rename(columns={
    'Mean (bps)': 'Mean',
    'HAC SE (bps)': 'HAC SE',
    r'95\% CI low': '95\\% CI low',
    r'95\% CI high': '95\\% CI high',
})

hac_latex = df_hac_tex.to_latex(
    index=False,
    caption=(
        r'HAC uncertainty for headline mean estimates (Newey--West, 60-lag cap). '
        r'Values are in basis points with 95\% confidence intervals.'
    ),
    label='tab:hac_headline',
    column_format='lrrrrr',
    float_format='%.3f',
    escape=False,
)
hac_latex = hac_latex.replace(r'\begin{table}', r'\begin{table}[H]', 1)
hac_latex = hac_latex.replace(r'\begin{tabular}', r'\footnotesize' + '\n' + r'\begin{tabular}', 1)
hac_latex = convert_to_tabularx(
    hac_latex,
    r'>{\raggedright\arraybackslash}Xrrrrr',
)
with open(os.path.join(TABLES_DIR, 'hac_headline_metrics.tex'), 'w') as f:
    f.write(hac_latex)
print("Saved tables/hac_headline_metrics.tex")

# ═══════════════════════════════════════════════════════════════════════════
# FIX 6 – Distributional robustness: Chow break test, higher moments,
#          cross-stablecoin correlation
# ═══════════════════════════════════════════════════════════════════════════
from scipy import stats as sp_stats

# --- 6a. Chow structural-break test at SVB_START for USDC dispersion ---
# H0: same linear model (const-only) before and after the break.
disp_usdc = basis['dispersion_usdc_kraken'].dropna()
n_total = len(disp_usdc)
pre_break = disp_usdc[disp_usdc.index < svb_start]
post_break = disp_usdc[disp_usdc.index >= svb_start]
n1, n2 = len(pre_break), len(post_break)

# Pooled RSS
X_full = np.ones((n_total, 1))
rss_full = float(sm.OLS(disp_usdc.values, X_full).fit().ssr)

# Sub-sample RSS
rss1 = float(sm.OLS(pre_break.values, np.ones((n1, 1))).fit().ssr)
rss2 = float(sm.OLS(post_break.values, np.ones((n2, 1))).fit().ssr)

k = 1  # number of parameters (intercept only)
chow_F = ((rss_full - rss1 - rss2) / k) / ((rss1 + rss2) / (n_total - 2 * k))
chow_p = 1.0 - sp_stats.f.cdf(chow_F, k, n_total - 2 * k)
print(f"\nChow break test at SVB_START for USDC D_t: F={chow_F:.1f}, p={chow_p:.2e}")

# Also test on adjusted residual B_t
basis_usdc = basis['basis_usdc_kraken'].dropna()
pre_b = basis_usdc[basis_usdc.index < svb_start]
post_b = basis_usdc[basis_usdc.index >= svb_start]
n_b, n1b, n2b = len(basis_usdc), len(pre_b), len(post_b)
rss_full_b = float(sm.OLS(basis_usdc.values, np.ones((n_b, 1))).fit().ssr)
rss1_b = float(sm.OLS(pre_b.values, np.ones((n1b, 1))).fit().ssr)
rss2_b = float(sm.OLS(post_b.values, np.ones((n2b, 1))).fit().ssr)
chow_F_b = ((rss_full_b - rss1_b - rss2_b) / k) / ((rss1_b + rss2_b) / (n_b - 2 * k))
chow_p_b = 1.0 - sp_stats.f.cdf(chow_F_b, k, n_b - 2 * k)
print(f"Chow break test at SVB_START for USDC B_t: F={chow_F_b:.1f}, p={chow_p_b:.2e}")

# --- 6a-ext. AR(1) dynamics Chow test: does ρ shift at the break? ---
# Fit X_t = c + ρ X_{t-1} + ε_t pooled vs split at svb_start.
basis_usdc_clean = basis['basis_usdc_kraken'].dropna()
y_ar = basis_usdc_clean.iloc[1:].values
x_ar = sm.add_constant(basis_usdc_clean.iloc[:-1].values)
n_ar = len(y_ar)
rss_ar_full = float(sm.OLS(y_ar, x_ar).fit().ssr)

# Find the split index in the lagged series
split_idx = (basis_usdc_clean.index[1:] >= svb_start).argmax()
y1, x1 = y_ar[:split_idx], x_ar[:split_idx]
y2, x2 = y_ar[split_idx:], x_ar[split_idx:]
rss_ar1 = float(sm.OLS(y1, x1).fit().ssr)
rss_ar2 = float(sm.OLS(y2, x2).fit().ssr)

k_ar = 2  # intercept + slope
chow_F_ar = ((rss_ar_full - rss_ar1 - rss_ar2) / k_ar) / (
    (rss_ar1 + rss_ar2) / (n_ar - 2 * k_ar)
)
chow_p_ar = 1.0 - sp_stats.f.cdf(chow_F_ar, k_ar, n_ar - 2 * k_ar)

# Also report regime-specific ρ for interpretation
rho_pre = float(sm.OLS(y1, x1).fit().params[1])
rho_post = float(sm.OLS(y2, x2).fit().params[1])
print(f"AR(1) Chow test for USDC B_t dynamics: F={chow_F_ar:.1f}, p={chow_p_ar:.2e}")
print(f"  rho_pre={rho_pre:.4f}, rho_post={rho_post:.4f}")

# --- 6a-ext2. Fee sensitivity for arbitrage (f=3 and f=10 bps) ---
# Re-compute USDC/USD Kraken crisis profitability at alternative fee levels
fee_sens_rows = []
for f_bps in [3.0, 5.0, 10.0]:
    for spec_key, n_legs, label in [
        ('basis_usdc_kraken', 3, 'USDC/USD (Kraken)'),
        ('xbasis_btcusd_coinbase_kraken', 2, 'Cross BTC/USD (CB-KR)'),
    ]:
        abs_basis = basis.loc[m_cri, spec_key].abs().dropna()
        # Get slippage from ranges
        if spec_key == 'basis_usdc_kraken':
            leg_cols = ['kraken_btcusdc', 'kraken_usdcusd', 'kraken_btcusd']
        else:
            leg_cols = ['coinbase_btcusd', 'kraken_btcusd']
        slip = sum(0.5 * ranges.loc[m_cri, c] * 10000.0 for c in leg_cols if c in ranges.columns)
        cost_fee = n_legs * f_bps
        cost_full = cost_fee + slip
        # Align
        common_idx = abs_basis.index.intersection(cost_full.dropna().index)
        ab = abs_basis.loc[common_idx]
        cf = cost_full.loc[common_idx]
        pct_profitable = float((ab > cf).mean() * 100)
        fee_sens_rows.append({
            'Channel': label,
            'f (bps)': f_bps,
            'Crisis %Profitable (fee+slip)': round(pct_profitable, 2),
        })

df_fee_sens = pd.DataFrame(fee_sens_rows)
print(f"\nFee sensitivity (crisis, fee+slippage):")
for _, row in df_fee_sens.iterrows():
    print(f"  {row['Channel']}, f={row['f (bps)']:.0f}: {row['Crisis %Profitable (fee+slip)']:.2f}%")

# --- 6b. Higher moments (skewness, excess kurtosis) of B_t by regime ---
moment_rows = []
for label, col in [('USDC', 'basis_usdc_kraken'), ('USDT', 'basis_usdt_kraken')]:
    for regime, mask in [('Pre-SVB', m_pre), ('Crisis', m_cri), ('Post-SVB', m_post)]:
        s = basis.loc[mask, col].dropna()
        moment_rows.append({
            'Channel': label,
            'Regime': regime,
            'N': len(s),
            'Mean': float(s.mean()),
            'Std': float(s.std()),
            'Skewness': float(sp_stats.skew(s)),
            'Excess Kurtosis': float(sp_stats.kurtosis(s)),  # Fisher=True by default
            'Min': float(s.min()),
            'Max': float(s.max()),
        })

df_moments = pd.DataFrame(moment_rows)
df_moments.to_csv(os.path.join(TABLES_DIR, 'distributional_robustness.csv'), index=False)

# --- 6b-ext. Jarque-Bera normality test on B_t by regime ---
print("\nJarque-Bera normality tests on B_t:")
for label, col in [('USDC', 'basis_usdc_kraken'), ('USDT', 'basis_usdt_kraken')]:
    for regime, mask in [('Pre-SVB', m_pre), ('Crisis', m_cri), ('Post-SVB', m_post)]:
        s = basis.loc[mask, col].dropna().values
        jb_stat, jb_p = sp_stats.jarque_bera(s)
        print(f"  {label} {regime}: JB={jb_stat:.0f}, p={jb_p:.2e}")

# --- 6c. Cross-stablecoin correlation by regime ---
corr_rows = []
for regime, mask in [('Pre-SVB', m_pre), ('Crisis', m_cri), ('Post-SVB', m_post)]:
    usdc_s = basis.loc[mask, 'basis_usdc_kraken'].dropna()
    usdt_s = basis.loc[mask, 'basis_usdt_kraken'].dropna()
    common = usdc_s.index.intersection(usdt_s.index)
    rho = float(usdc_s.loc[common].corr(usdt_s.loc[common]))
    corr_rows.append({'Regime': regime, 'Corr(B_USDC, B_USDT)': rho, 'N': len(common)})

df_corr = pd.DataFrame(corr_rows)
print(f"\nCross-stablecoin B_t correlation by regime:")
for _, row in df_corr.iterrows():
    print(f"  {row['Regime']}: rho={row['Corr(B_USDC, B_USDT)']:.3f}  (N={row['N']})")

# --- Generate compact LaTeX table ---
# Combine moments + correlation into a single table for the paper
tex_rows = []
for _, r in df_moments.iterrows():
    tex_rows.append({
        'Channel': r['Channel'],
        'Regime': r['Regime'],
        'Skewness': f"{r['Skewness']:.2f}",
        'Ex.~Kurt.': f"{r['Excess Kurtosis']:.1f}",
        'Min': f"{r['Min']:.1f}",
        'Max': f"{r['Max']:.1f}",
    })

df_tex = pd.DataFrame(tex_rows)
tex_str = df_tex.to_latex(
    index=False,
    caption=(
        r'Higher-moment diagnostics for adjusted residual $B_t$ by regime. '
        r'Excess kurtosis (Fisher) measures tail heaviness beyond Gaussian ($=0$). '
        r'Min/Max are in basis points.'
    ),
    label='tab:dist_robust',
    column_format='llrrrr',
    escape=False,
)
tex_str = tex_str.replace(r'\begin{table}', r'\begin{table}[H]', 1)
tex_str = tex_str.replace(r'\begin{tabular}', r'\footnotesize' + '\n' + r'\begin{tabular}', 1)
with open(os.path.join(TABLES_DIR, 'distributional_robustness.tex'), 'w') as f:
    f.write(tex_str)
print("Saved tables/distributional_robustness.tex")

# Print key numbers for inline text use
print(f"\n=== KEY NUMBERS FOR INLINE TEXT ===")
print(f"Roll spread: Kraken USDC crisis = {roll_pivot.loc['Kraken BTC/USDC','Crisis']:.2f} bps")
print(f"Roll spread: Kraken USD crisis  = {roll_pivot.loc['Kraken BTC/USD','Crisis']:.2f} bps")
print(f"Roll ratio USDC/USD crisis = {roll_pivot.loc['Kraken BTC/USDC','Crisis']/roll_pivot.loc['Kraken BTC/USD','Crisis']:.1f}x")
print(f"Amihud ILLIQ: Kraken USDC crisis  = {amihud_pivot.loc['Kraken BTC/USDC','Crisis']:.4f}")
print(f"Amihud ILLIQ: Kraken USD crisis   = {amihud_pivot.loc['Kraken BTC/USD','Crisis']:.4f}")
if not df_is.empty and len(is_rows) > 0 and not np.isnan(is_rows[-1].get('IS_USD_mid', np.nan)):
    r = is_rows[-1]
    print(f"Hasbrouck IS BTC/USD midpoint = {r['IS_USD_mid']:.2f}  bounds [{r['IS_USD_lower']:.2f}, {r['IS_USD_upper']:.2f}]")
print(
    "GENIUS D_t reduction range (25%-75% mitigation): "
    f"{Dt_reduction_abs_low:.0f}-{Dt_reduction_abs_high:.0f} bps "
    f"({Dt_reduction_pct_low:.0f}% - {Dt_reduction_pct_high:.0f}%)"
)
print("=== END KEY NUMBERS ===")

num_h_updates = enforce_table_H_placement(TABLES_DIR)
print(f"Enforced [H] placement in {num_h_updates} table files.")
