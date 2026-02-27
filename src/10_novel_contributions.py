"""
Novel mathematical contributions for the IAQF column paper:
  1. Contagion intensity model (coupled OU, lambda estimation by regime)
  2. Half-life ratio bootstrap CI (formal test of the ~940x gap)
  3. Asymmetry test (differential lambda for peg discounts vs premiums)

All results are printed for integration into LaTeX. No files are modified.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────
DATA_PROCESSED = 'data_processed'
basis = pd.read_parquet(f'{DATA_PROCESSED}/basis.parquet')
prices = pd.read_parquet(f'{DATA_PROCESSED}/prices.parquet')

basis_ff_flags_path = f'{DATA_PROCESSED}/basis_ffill_flags.parquet'
import os
if os.path.exists(basis_ff_flags_path):
    basis_ff_flags = pd.read_parquet(basis_ff_flags_path)
else:
    basis_ff_flags = pd.DataFrame(False, index=basis.index, columns=basis.columns)
basis_ff_flags = basis_ff_flags.reindex(index=basis.index, columns=basis.columns).fillna(False).astype(bool)

price_ff_flags_path = f'{DATA_PROCESSED}/price_ffill_flags.parquet'
if os.path.exists(price_ff_flags_path):
    price_ff_flags = pd.read_parquet(price_ff_flags_path)
else:
    price_ff_flags = pd.DataFrame(False, index=prices.index, columns=prices.columns)
price_ff_flags = price_ff_flags.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)

# ── Regimes ────────────────────────────────────────────────────────
svb_start = pd.Timestamp('2023-03-10', tz='UTC')
svb_end   = pd.Timestamp('2023-03-13', tz='UTC')
regimes = {
    'Pre-SVB':  (basis.index.min(), svb_start),
    'Crisis':   (svb_start, svb_end),
    'Post-SVB': (svb_end, basis.index.max()),
}

from half_life_utils import estimate_half_life_from_ecm, half_life_from_rho

# ====================================================================
# 1. CONTAGION INTENSITY MODEL
# ====================================================================
print("=" * 70)
print("1. CONTAGION INTENSITY MODEL")
print("=" * 70)

def estimate_contagion(B_col, S_col, regime_name, t0, t1, ff_filter=False):
    """
    Estimate: Delta B_t = a + beta * B_{t-1} + lambda * S_{t-1} + eps_t
    with HAC standard errors (Newey-West, 60 lags).

    Returns dict of results.
    """
    mask = (basis.index >= t0) & (basis.index < t1)

    df = pd.DataFrame({
        'B': basis.loc[mask, B_col],
        'S': basis.loc[mask, S_col],
    }).dropna()

    if ff_filter and B_col in basis_ff_flags.columns:
        ff_B = basis_ff_flags.loc[mask, B_col].reindex(df.index).fillna(False)
        df = df[~ff_B]

    if len(df) < 100:
        return None

    # Construct lagged variables
    df['B_lag'] = df['B'].shift(1)
    df['S_lag'] = df['S'].shift(1)
    df['dB'] = df['B'] - df['B_lag']
    df = df.dropna()

    if len(df) < 100:
        return None

    # OLS with HAC
    X = sm.add_constant(df[['B_lag', 'S_lag']])
    y = df['dB']
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 60})

    beta = float(model.params['B_lag'])
    lam = float(model.params['S_lag'])
    lam_se = float(model.bse['S_lag'])
    lam_p = float(model.pvalues['S_lag'])
    beta_p = float(model.pvalues['B_lag'])
    rho_implied = 1.0 + beta

    # Half-life from implied rho (with lambda controlled for)
    hl = half_life_from_rho(rho_implied, dt_minutes=1.0) if 0 < rho_implied < 1 else np.nan

    return {
        'regime': regime_name,
        'n_obs': int(model.nobs),
        'beta': beta,
        'beta_p': beta_p,
        'rho_implied': rho_implied,
        'half_life_min': hl,
        'lambda': lam,
        'lambda_se': lam_se,
        'lambda_t': lam / lam_se if lam_se > 0 else np.nan,
        'lambda_p': lam_p,
        'R2': float(model.rsquared),
        'const': float(model.params['const']),
        'const_p': float(model.pvalues['const']),
    }

# Primary: USDC channel
print("\n--- USDC Channel (Kraken) ---")
usdc_results = {}
for regime, (t0, t1) in regimes.items():
    res = estimate_contagion('basis_usdc_kraken', 'usdc_peg_dev_kraken', regime, t0, t1)
    if res:
        usdc_results[regime] = res
        sig = '***' if res['lambda_p'] < 0.001 else ('**' if res['lambda_p'] < 0.01 else ('*' if res['lambda_p'] < 0.05 else ''))
        print(f"  {regime:10s}: lambda={res['lambda']:+.5f} (SE={res['lambda_se']:.5f}, t={res['lambda_t']:.2f}, p={res['lambda_p']:.4f}) {sig}")
        print(f"             beta={res['beta']:+.5f} (p={res['beta_p']:.4f}), rho={res['rho_implied']:.4f}, HL={res['half_life_min']:.2f} min, R2={res['R2']:.4f}, N={res['n_obs']}")

# Secondary: USDT channel
print("\n--- USDT Channel (Kraken) ---")
usdt_results = {}
for regime, (t0, t1) in regimes.items():
    res = estimate_contagion('basis_usdt_kraken', 'usdt_peg_dev_kraken', regime, t0, t1)
    if res:
        usdt_results[regime] = res
        sig = '***' if res['lambda_p'] < 0.001 else ('**' if res['lambda_p'] < 0.01 else ('*' if res['lambda_p'] < 0.05 else ''))
        print(f"  {regime:10s}: lambda={res['lambda']:+.5f} (SE={res['lambda_se']:.5f}, t={res['lambda_t']:.2f}, p={res['lambda_p']:.4f}) {sig}")
        print(f"             beta={res['beta']:+.5f} (p={res['beta_p']:.4f}), rho={res['rho_implied']:.4f}, HL={res['half_life_min']:.2f} min, R2={res['R2']:.4f}, N={res['n_obs']}")

# Robustness: no-FF filter for USDC crisis
print("\n--- USDC Crisis No-FF Robustness ---")
res_noff = estimate_contagion('basis_usdc_kraken', 'usdc_peg_dev_kraken', 'Crisis', svb_start, svb_end, ff_filter=True)
if res_noff:
    print(f"  Crisis(no-FF): lambda={res_noff['lambda']:+.5f} (SE={res_noff['lambda_se']:.5f}, p={res_noff['lambda_p']:.4f}), N={res_noff['n_obs']}")

# Robustness: 5-minute sampling for USDC crisis
print("\n--- USDC Crisis 5-min Robustness ---")
mask_crisis = (basis.index >= svb_start) & (basis.index < svb_end)
B_5m = basis.loc[mask_crisis, 'basis_usdc_kraken'].resample('5min').last()
S_5m = basis.loc[mask_crisis, 'usdc_peg_dev_kraken'].resample('5min').last()
df_5m = pd.DataFrame({'B': B_5m, 'S': S_5m}).dropna()
df_5m['B_lag'] = df_5m['B'].shift(1)
df_5m['S_lag'] = df_5m['S'].shift(1)
df_5m['dB'] = df_5m['B'] - df_5m['B_lag']
df_5m = df_5m.dropna()
if len(df_5m) >= 50:
    X_5m = sm.add_constant(df_5m[['B_lag', 'S_lag']])
    m_5m = sm.OLS(df_5m['dB'], X_5m).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    print(f"  Crisis(5m): lambda={float(m_5m.params['S_lag']):+.5f} (SE={float(m_5m.bse['S_lag']):.5f}, p={float(m_5m.pvalues['S_lag']):.4f}), N={int(m_5m.nobs)}")


# ====================================================================
# 2. HALF-LIFE RATIO BOOTSTRAP CI
# ====================================================================
print("\n" + "=" * 70)
print("2. HALF-LIFE RATIO BOOTSTRAP CI")
print("=" * 70)

def estimate_rho_from_series(series, dt_minutes=1.0):
    """Estimate AR(1) rho from a series via ECM regression."""
    s = series.dropna().astype(float)
    if len(s) < 100:
        return np.nan, len(s)
    reg = pd.DataFrame({'x': s})
    reg['x_lag'] = reg['x'].shift(1)
    reg['dx'] = reg['x'] - reg['x_lag']
    reg = reg.dropna()
    if len(reg) < 100:
        return np.nan, len(reg)
    X = sm.add_constant(reg['x_lag'])
    model = sm.OLS(reg['dx'], X).fit()
    beta = float(model.params.iloc[1])
    rho = 1.0 + beta
    return rho, len(reg)


# Point estimates for crisis
B_crisis = basis.loc[mask_crisis, 'basis_usdc_kraken'].dropna()
S_crisis = basis.loc[mask_crisis, 'usdc_peg_dev_kraken'].dropna()

rho_B, n_B = estimate_rho_from_series(B_crisis)
rho_S, n_S = estimate_rho_from_series(S_crisis)

hl_B = half_life_from_rho(rho_B, 1.0)
hl_S = half_life_from_rho(rho_S, 1.0) if 0 < rho_S < 1 else np.inf
R_point = hl_S / hl_B if np.isfinite(hl_B) and hl_B > 0 else np.nan

print(f"\nPoint estimates (crisis):")
print(f"  rho_B = {rho_B:.6f}, HL_B = {hl_B:.2f} min")
print(f"  rho_S = {rho_S:.6f}, HL_S = {hl_S:.2f} min")
print(f"  R = HL_S / HL_B = {R_point:.1f}")

# Block bootstrap
np.random.seed(42)
BLOCK_LEN = 60  # 1 hour blocks
N_BOOT = 5000

def block_bootstrap_rho(series, block_len, n_boot):
    """
    Block bootstrap for AR(1) rho estimate.
    Returns array of bootstrap rho estimates.
    """
    s = series.dropna().values
    n = len(s)
    n_blocks = int(np.ceil(n / block_len))
    rhos = np.full(n_boot, np.nan)

    for b in range(n_boot):
        # Draw random block starts
        starts = np.random.randint(0, n - block_len + 1, size=n_blocks)
        # Concatenate blocks
        boot_sample = np.concatenate([s[st:st+block_len] for st in starts])[:n]

        # Estimate rho
        x = boot_sample[:-1]
        dx = np.diff(boot_sample)
        if len(x) < 50:
            continue
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, dx, rcond=None)[0][1]
            rhos[b] = 1.0 + beta
        except:
            pass

    return rhos

print("\nRunning block bootstrap (5000 replications, block=60 min)...")

rhos_B_boot = block_bootstrap_rho(B_crisis, BLOCK_LEN, N_BOOT)
rhos_S_boot = block_bootstrap_rho(S_crisis, BLOCK_LEN, N_BOOT)

# Compute half-life ratios
hls_B_boot = np.array([half_life_from_rho(r, 1.0) if 0 < r < 1 else np.nan for r in rhos_B_boot])
hls_S_boot = np.array([
    half_life_from_rho(r, 1.0) if 0 < r < 1 else np.inf
    for r in rhos_S_boot
])

# For ratio: handle cases where S is unit root (hl_S = inf)
# Ratio R = hl_S / hl_B
valid_mask = np.isfinite(hls_B_boot) & (hls_B_boot > 0)
R_boot = np.where(
    valid_mask & np.isfinite(hls_S_boot),
    hls_S_boot / hls_B_boot,
    np.where(valid_mask, np.inf, np.nan)  # inf if S is unit root
)

# For CI: use finite ratios
R_finite = R_boot[np.isfinite(R_boot) & ~np.isnan(R_boot)]
pct_infinite = np.sum(np.isinf(R_boot[~np.isnan(R_boot)])) / np.sum(~np.isnan(R_boot)) * 100

print(f"\nBootstrap results:")
print(f"  Valid bootstrap ratios (finite): {len(R_finite)} / {N_BOOT}")
print(f"  Pct with R=inf (peg unit root): {pct_infinite:.1f}%")

if len(R_finite) > 100:
    R_median = np.median(R_finite)
    R_ci_lo = np.percentile(R_finite, 2.5)
    R_ci_hi = np.percentile(R_finite, 97.5)
    R_ci_lo_90 = np.percentile(R_finite, 5.0)
    # One-sided test: fraction of bootstrap R <= 1
    pval_R1 = np.mean(R_finite <= 1.0)
    # More conservative: fraction of all bootstrap R (including inf) <= 1
    R_all_valid = R_boot[~np.isnan(R_boot)]
    pval_R1_all = np.mean(R_all_valid <= 1.0)

    print(f"  R median (finite): {R_median:.0f}")
    print(f"  R 95% CI (finite): [{R_ci_lo:.0f}, {R_ci_hi:.0f}]")
    print(f"  R 5th percentile (one-sided lower bound): {R_ci_lo_90:.0f}")
    print(f"  p-value for H0: R <= 1 (finite only): {pval_R1:.6f}")
    print(f"  p-value for H0: R <= 1 (all, inf=large): {pval_R1_all:.6f}")

    # Also report hl_B bootstrap CI
    hls_B_finite = hls_B_boot[np.isfinite(hls_B_boot)]
    print(f"\n  HL_B bootstrap: median={np.median(hls_B_finite):.2f}, 95% CI=[{np.percentile(hls_B_finite, 2.5):.2f}, {np.percentile(hls_B_finite, 97.5):.2f}]")

    # hl_S: many infinite, report fraction
    hls_S_finite = hls_S_boot[np.isfinite(hls_S_boot)]
    if len(hls_S_finite) > 10:
        print(f"  HL_S bootstrap (finite only): median={np.median(hls_S_finite):.0f}, 95% CI=[{np.percentile(hls_S_finite, 2.5):.0f}, {np.percentile(hls_S_finite, 97.5):.0f}]")


# ====================================================================
# 3. ASYMMETRY TEST
# ====================================================================
print("\n" + "=" * 70)
print("3. ASYMMETRY TEST (Conditional on lambda significant)")
print("=" * 70)

# Check if crisis lambda is significant
crisis_lam_p = usdc_results.get('Crisis', {}).get('lambda_p', 1.0)
print(f"\nCrisis lambda p-value: {crisis_lam_p:.6f}")

if crisis_lam_p < 0.05:
    print("Lambda is significant -> proceeding with asymmetry test\n")

    # S_t < 0 means USDC below par (discount/de-peg)
    # S_t > 0 means USDC above par (premium)
    # During crisis, S_t is predominantly negative (USDC de-pegged below $1)

    for regime, (t0, t1) in regimes.items():
        mask = (basis.index >= t0) & (basis.index < t1)
        df = pd.DataFrame({
            'B': basis.loc[mask, 'basis_usdc_kraken'],
            'S': basis.loc[mask, 'usdc_peg_dev_kraken'],
        }).dropna()

        if len(df) < 100:
            continue

        df['B_lag'] = df['B'].shift(1)
        df['S_lag'] = df['S'].shift(1)
        df['dB'] = df['B'] - df['B_lag']

        # Asymmetric decomposition
        # S_lag_neg: peg discount (S < 0, i.e., below $1) -- the de-peg direction
        # S_lag_pos: peg premium (S > 0, i.e., above $1)
        df['S_lag_neg'] = df['S_lag'].clip(upper=0)  # negative values (de-peg)
        df['S_lag_pos'] = df['S_lag'].clip(lower=0)  # positive values (premium)

        df = df.dropna()
        if len(df) < 100:
            continue

        X = sm.add_constant(df[['B_lag', 'S_lag_neg', 'S_lag_pos']])
        y = df['dB']
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 60})

        lam_neg = float(model.params['S_lag_neg'])
        lam_pos = float(model.params['S_lag_pos'])
        lam_neg_p = float(model.pvalues['S_lag_neg'])
        lam_pos_p = float(model.pvalues['S_lag_pos'])
        lam_neg_se = float(model.bse['S_lag_neg'])
        lam_pos_se = float(model.bse['S_lag_pos'])

        # Wald test for lambda_neg = lambda_pos
        R_mat = np.array([[0, 0, 1, -1]])  # test: S_lag_neg - S_lag_pos = 0
        try:
            wald = model.wald_test(R_mat)
            wald_F = float(wald.statistic[0][0])
            wald_p = float(wald.pvalue)
        except:
            wald_F = np.nan
            wald_p = np.nan

        print(f"  {regime:10s}:")
        print(f"    lambda_neg (de-peg) = {lam_neg:+.5f} (SE={lam_neg_se:.5f}, p={lam_neg_p:.4f})")
        print(f"    lambda_pos (premium)= {lam_pos:+.5f} (SE={lam_pos_se:.5f}, p={lam_pos_p:.4f})")
        print(f"    Wald test (lambda_neg = lambda_pos): F={wald_F:.2f}, p={wald_p:.4f}")
        print(f"    N={int(model.nobs)}, R2={model.rsquared:.4f}")

        # Count observations in each regime
        n_neg = (df['S_lag'] < 0).sum()
        n_pos = (df['S_lag'] >= 0).sum()
        print(f"    Obs with S<0: {n_neg}, S>=0: {n_pos}")
else:
    print("Lambda NOT significant -> skipping asymmetry test")


# ====================================================================
# SUMMARY TABLE FOR LATEX
# ====================================================================
print("\n" + "=" * 70)
print("SUMMARY: LaTeX-ready numbers")
print("=" * 70)

print("\n--- Contagion Intensity Table ---")
for ch_name, ch_results in [('USDC', usdc_results), ('USDT', usdt_results)]:
    print(f"\n{ch_name}:")
    for regime in ['Pre-SVB', 'Crisis', 'Post-SVB']:
        r = ch_results.get(regime)
        if r:
            sig = '***' if r['lambda_p'] < 0.001 else ('**' if r['lambda_p'] < 0.01 else ('*' if r['lambda_p'] < 0.05 else ''))
            p_str = '<0.001' if r['lambda_p'] < 0.001 else f"{r['lambda_p']:.3f}"
            print(f"  {regime}: lambda={r['lambda']:+.4f}{sig}  SE={r['lambda_se']:.4f}  p={p_str}  R2={r['R2']:.3f}  N={r['n_obs']}")
