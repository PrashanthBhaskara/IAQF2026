"""
09_additional_figures.py
========================
Generate academic-quality versions of two previously unused figures:
  1. SVB Crisis Zoom (all intra-exchange B_t + cross-exchange basis)
  2. VAR Impulse-Response Function (BTC/USD vs BTC/USDC on Kraken)

Outputs go to both:
  figures/       (full-width, for IAQF_Final.tex)
  figures_col/   (single-column, for IAQF_column_Final.tex)

Run from the project root:
    cd /path/to/IAQF2026
    python src/09_additional_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from statsmodels.tsa.api import VAR

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(ROOT, 'data_processed')
FIGURES_DIR    = os.path.join(ROOT, 'figures')
FIGURES_COL    = os.path.join(ROOT, 'figures_col')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIGURES_COL, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data…")
prices = pd.read_parquet(os.path.join(DATA_PROCESSED, 'prices.parquet'))
basis  = pd.read_parquet(os.path.join(DATA_PROCESSED, 'basis.parquet'))

returns = prices.pct_change(fill_method=None).dropna()

# ── Regime boundaries ─────────────────────────────────────────────────────────
svb_start = pd.Timestamp('2023-03-10', tz='UTC')
svb_end   = pd.Timestamp('2023-03-13', tz='UTC')

# ── Colour palette (Paul Tol "bright" – colour-blind safe) ────────────────────
CB_BLUE   = '#4477AA'
CB_RED    = '#EE6677'
CB_GREEN  = '#228833'
CB_ORANGE = '#CCBB44'
CB_PURPLE = '#AA3377'
CB_CYAN   = '#66CCEE'
CB_DARK   = '#332288'

# ── Full-width RC (for figures/) ──────────────────────────────────────────────
FULL_RC = {
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':  'stix',
    'axes.facecolor':    'white',
    'figure.facecolor':  'white',
    'axes.grid':         False,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    12,
    'legend.fontsize':   10,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
}

# ── Single-column RC (for figures_col/) ───────────────────────────────────────
COL_RC = {
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':  'stix',
    'axes.facecolor':    'white',
    'figure.facecolor':  'white',
    'axes.grid':         False,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.6,
    'font.size':         7,
    'axes.titlesize':    8,
    'axes.labelsize':    7.5,
    'legend.fontsize':   6.5,
    'xtick.labelsize':   6.5,
    'ytick.labelsize':   6.5,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'lines.linewidth':   0.65,
    'xtick.major.size':  2.5,
    'ytick.major.size':  2.5,
}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – SVB Crisis Zoom (all channels, March 10-13)
# ══════════════════════════════════════════════════════════════════════════════
def plot_crisis_zoom(rc_dict, figsize_a, figsize_b, lw, save_dir, dpi=300):
    """
    Two-panel figure:
      Panel A: Intra-exchange adjusted residual B_t for all three channels
      Panel B: Cross-exchange basis for BTC/USDT (Binance-Kraken) and BTC/USD (Coinbase-Kraken)
    """
    svb_data = basis.loc[(basis.index >= svb_start) & (basis.index < svb_end)]

    with plt.rc_context(rc_dict):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize_a, figsize_b),
                                        sharex=True)

        # --- Panel A: Intra-exchange B_t ---
        ax1.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.35)
        for col, lbl, c in [
            ('basis_usdc_kraken',   r'$B_{USDC,t}$ (Kraken)',   CB_GREEN),
            ('basis_usdt_kraken',   r'$B_{USDT,t}$ (Kraken)',   CB_BLUE),
            ('basis_usdt_coinbase', r'$B_{USDT,t}$ (Coinbase)', CB_RED),
        ]:
            if col in svb_data.columns:
                ax1.plot(svb_data.index, svb_data[col],
                         linewidth=lw, alpha=0.88, label=lbl, color=c)
        ax1.set_ylabel(r'Adjusted Residual $B_t$ (bps)')
        ax1.set_title('(A) Intra-Exchange Adjusted Residuals')
        ax1.legend(loc='upper right', frameon=True, framealpha=0.92,
                   edgecolor='#CCCCCC')

        # --- Panel B: Cross-exchange basis ---
        ax2.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.35)
        for col, lbl, c in [
            ('xbasis_btcusdt_binance_kraken', r'Binance$-$Kraken BTC/USDT', CB_ORANGE),
            ('xbasis_btcusd_coinbase_kraken', r'Coinbase$-$Kraken BTC/USD', CB_DARK),
        ]:
            if col in svb_data.columns:
                ax2.plot(svb_data.index, svb_data[col],
                         linewidth=lw, alpha=0.88, label=lbl, color=c)
        ax2.set_ylabel('Cross-Exchange Basis (bps)')
        ax2.set_title('(B) Cross-Exchange Basis')
        ax2.legend(loc='upper right', frameon=True, framealpha=0.92,
                   edgecolor='#CCCCCC')

        # --- Shared x-axis ---
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), ha='center')
        ax2.set_xlim(svb_start, svb_end)

        fig.tight_layout(pad=0.5, h_pad=0.8)
        path = os.path.join(save_dir, 'fig_svb_crisis_zoom.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved → {path}")


# Full-width version
plot_crisis_zoom(FULL_RC, figsize_a=12, figsize_b=7, lw=0.55,
                 save_dir=FIGURES_DIR)
# Column version
plot_crisis_zoom(COL_RC, figsize_a=3.4, figsize_b=4.5, lw=0.45,
                 save_dir=FIGURES_COL)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – VAR Impulse-Response Function (BTC/USD vs BTC/USDC, Kraken)
#
# Methodology notes:
#   - VAR on 1-minute percentage returns (×10000 → bps) for Kraken BTC/USD
#     and BTC/USDC.  Lag selected by AIC (maxlags=10).
#   - Orthogonalized IRF via Cholesky decomposition.
#   - Ordering: [BTC/USD, BTC/USDC]. BTC/USD is ordered first because it is
#     more liquid and the identified price-discovery leader (Hasbrouck IS = 0.73).
#     Under this ordering, BTC/USD can affect BTC/USDC contemporaneously but
#     not vice versa.
#   - IRF horizon: 10 lags (= 10 minutes at 1-min frequency).
#   - 95% confidence bands from bootstrap (default in statsmodels).
#   - Full-sample estimation (March 1-21, 2023); reflects average dynamics
#     across all three regimes.
# ══════════════════════════════════════════════════════════════════════════════

# --- Fit VAR model ---
var_data = returns[['kraken_btcusd', 'kraken_btcusdc']].dropna() * 10000
var_model = VAR(var_data)
var_result = var_model.fit(maxlags=10, ic='aic')
selected_lags = var_result.k_ar
n_obs = len(var_data)
print(f"\nVAR: AIC-selected lags = {selected_lags}, N = {n_obs}")

# --- Compute orthogonalized IRF with asymptotic 95% CI ---
irf = var_result.irf(10)

# Extract arrays: shape (steps+1, n_vars, n_vars) where steps=10 → 11 points
irf_vals  = irf.orth_irfs                         # (11, 2, 2)
irf_se    = irf.stderr(orth=True)                  # (11, 2, 2) asymptotic SE
irf_lower = irf_vals - 1.96 * irf_se              # 95% CI lower
irf_upper = irf_vals + 1.96 * irf_se              # 95% CI upper
horizons  = np.arange(irf_vals.shape[0])           # 0 … 10

# Labels for the 2×2 grid
NAMES = ['BTC/USD', 'BTC/USDC']
PANEL_LABELS = [
    ['(A)', '(B)'],
    ['(C)', '(D)'],
]


def plot_irf(rc_dict, figsize_w, figsize_h, lw_main, lw_ci, save_dir, dpi=300):
    """
    2×2 grid of orthogonalized impulse-response functions.
    Row i = response of variable i; Column j = shock to variable j.
    """
    with plt.rc_context(rc_dict):
        fig, axes = plt.subplots(2, 2, figsize=(figsize_w, figsize_h),
                                  sharex=True)

        for i in range(2):       # response variable
            for j in range(2):   # shock variable
                ax = axes[i, j]
                resp = irf_vals[:, i, j]
                lo   = irf_lower[:, i, j]
                hi   = irf_upper[:, i, j]

                ax.fill_between(horizons, lo, hi,
                                color=CB_BLUE, alpha=0.18, linewidth=0)
                ax.plot(horizons, resp,
                        color=CB_BLUE, linewidth=lw_main, zorder=3)
                ax.axhline(0, color='black', linewidth=0.5,
                           linestyle='--', alpha=0.4)

                title = (f'{PANEL_LABELS[i][j]}  '
                         f'{NAMES[j]} $\\rightarrow$ {NAMES[i]}')
                ax.set_title(title)

                if i == 1:
                    ax.set_xlabel('Lag (minutes)')
                if j == 0:
                    ax.set_ylabel('Response (bps)')

                ax.set_xlim(0, 10)
                ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

        fig.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.0)
        path = os.path.join(save_dir, 'fig_var_irf.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved → {path}")


# Full-width version
plot_irf(FULL_RC, figsize_w=10, figsize_h=7, lw_main=1.6, lw_ci=0.8,
         save_dir=FIGURES_DIR)
# Column version
plot_irf(COL_RC, figsize_w=3.4, figsize_h=3.4, lw_main=0.9, lw_ci=0.5,
         save_dir=FIGURES_COL)

print("\nAll additional figures generated successfully.")
