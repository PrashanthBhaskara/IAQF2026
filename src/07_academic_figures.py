"""
07_academic_figures.py
======================
Three high-quality academic figures for the cross-currency fragmentation paper.

Figure 1 – Two-Layer Persistence (dual-axis, crisis window)
Figure 2 – Tail Blowout KDE (pre-SVB vs. crisis)
Figure 3 – Stablecoin Substitution Scatter (D_USDC vs. D_USDT)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PROCESSED = 'data_processed'
FIGURES_DIR    = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',          # Times-compatible math rendering
    'axes.facecolor':     'white',
    'figure.facecolor':   'white',
    'axes.grid':          False,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.8,
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'legend.fontsize':    10,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
})

# Paul Tol "bright" – colour-blind safe
CB_BLUE   = '#4477AA'
CB_RED    = '#EE6677'
CB_GRAY   = '#BBBBBB'
CB_ORANGE = '#CCBB44'

# ── Load data ─────────────────────────────────────────────────────────────────
basis  = pd.read_parquet(os.path.join(DATA_PROCESSED, 'basis.parquet'))

# ── Regime boundaries ─────────────────────────────────────────────────────────
svb_start = pd.Timestamp('2023-03-10', tz='UTC')
svb_end   = pd.Timestamp('2023-03-13', tz='UTC')
pre_start = pd.Timestamp('2023-03-01', tz='UTC')

pre_mask    = (basis.index >= pre_start) & (basis.index < svb_start)
crisis_mask = (basis.index >= svb_start) & (basis.index < svb_end)

basis_pre    = basis.loc[pre_mask]
basis_crisis = basis.loc[crisis_mask]


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Two-Layer Persistence
#   Left axis : Adjusted Residual B_{USDC,t} (Kraken)   – fast mean-reversion
#   Right axis: USDC Peg Deviation (bps)                – slow, persistent trough
# ══════════════════════════════════════════════════════════════════════════════
fig1_start = pd.Timestamp('2023-03-10 00:00:00', tz='UTC')
fig1_end   = pd.Timestamp('2023-03-13 00:00:00', tz='UTC')
fig1_data  = basis.loc[(basis.index >= fig1_start) & (basis.index < fig1_end)]

fig, ax1 = plt.subplots(figsize=(12, 5))

# --- Left axis: B_t ---
line_bt, = ax1.plot(
    fig1_data.index,
    fig1_data['basis_usdc_kraken'],
    color=CB_BLUE, linewidth=0.55, alpha=0.90,
    label=r'Adjusted Residual $B_{USDC,t}$ (Kraken)',
    zorder=3,
)
ax1.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.35, zorder=1)
ax1.set_ylabel(r'Adjusted Residual $B_{USDC,t}$ (bps)', color=CB_BLUE, labelpad=8)
ax1.tick_params(axis='y', labelcolor=CB_BLUE)
ax1.set_ylim(-130, 210)
ax1.yaxis.set_major_locator(mticker.MultipleLocator(50))

# --- Right axis: peg deviation ---
ax2 = ax1.twinx()
ax2.spines['right'].set_visible(True)
ax2.spines['right'].set_linewidth(0.8)
line_peg, = ax2.plot(
    fig1_data.index,
    fig1_data['usdc_peg_dev_kraken'],
    color=CB_RED, linewidth=1.1, alpha=0.92,
    label=r'USDC Peg Deviation (bps)',
    zorder=2,
)
ax2.set_ylabel(r'USDC Peg Deviation from \$1.00 (bps)', color=CB_RED, labelpad=8)
ax2.tick_params(axis='y', labelcolor=CB_RED)
ax2.set_ylim(-1450, 230)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(250))

# --- Shared x-axis ---
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
plt.setp(ax1.xaxis.get_majorticklabels(), ha='center', fontsize=9)
ax1.set_xlabel('Date / Time (UTC)', labelpad=6)
ax1.set_xlim(fig1_start, fig1_end)

# --- Combined legend ---
lines  = [line_bt, line_peg]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True,
           framealpha=0.92, edgecolor='#CCCCCC', fontsize=10)

# --- Annotation box: half-life gap ---
ax1.text(
    0.995, 0.97,
    r'Half-life gap:  $\times$940' + '\n'
    r'$B_{USDC,t}$: $\approx$0.6 min  (half-life)' + '\n'
    r'Peg deviation: $\approx$572 min  (half-life)',
    transform=ax1.transAxes, fontsize=9,
    va='top', ha='right',
    bbox=dict(boxstyle='round,pad=0.45', facecolor='#F7F7F7',
              edgecolor='#CCCCCC', linewidth=0.9, alpha=0.95),
)

ax1.set_title(
    r'Two-Layer Persistence: $B_{USDC,t}$ vs.\ USDC Peg Deviation — Kraken'
    '\n(SVB Crisis Window: 10–13 March 2023 UTC)',
    pad=10,
)

fig.tight_layout()
fig.savefig(
    os.path.join(FIGURES_DIR, 'fig_two_layer_persistence.png'),
    dpi=300, bbox_inches='tight',
)
plt.close(fig)
print('Figure 1 saved  →  fig_two_layer_persistence.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Tail Blowout: Pre-Crisis vs. Crisis KDE
# ══════════════════════════════════════════════════════════════════════════════
pre_bt    = basis_pre['basis_usdc_kraken'].dropna()
crisis_bt = basis_crisis['basis_usdc_kraken'].dropna()

x_lo, x_hi = -50, 100   # display window; KDE is also clipped here

fig, ax = plt.subplots(figsize=(9, 5))

# --- KDE fills ---
sns.kdeplot(
    pre_bt, ax=ax,
    color=CB_GRAY, linewidth=2.0,
    fill=True, alpha=0.40,
    label=r'Pre-SVB (1–9 Mar 2023,  $n={:,}$)'.format(len(pre_bt)),
    clip=(x_lo, x_hi),
    bw_adjust=0.9,
)
sns.kdeplot(
    crisis_bt, ax=ax,
    color=CB_RED, linewidth=2.0,
    fill=True, alpha=0.35,
    label=r'Crisis (10–13 Mar 2023, $n={:,}$)'.format(len(crisis_bt)),
    clip=(x_lo, x_hi),
    bw_adjust=0.9,
)

# --- Zero reference ---
ax.axvline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.45, zorder=1)

# --- Mean marker lines ---
ax.axvline(
    pre_bt.mean(), color=CB_GRAY, linewidth=1.6, linestyle=':',
    label=fr'Pre-SVB mean ({pre_bt.mean():.1f} bps)',
)
ax.axvline(
    crisis_bt.mean(), color=CB_RED, linewidth=1.6, linestyle=':',
    label=fr'Crisis mean ({crisis_bt.mean():.1f} bps)',
)

# --- P99 callout box ---
p99_pre    = pre_bt.quantile(0.99)
p99_crisis = crisis_bt.quantile(0.99)
callout = (
    r'\textbf{P99 tail expansion}' '\n'
    f'Pre-SVB:  {p99_pre:.1f} bps\n'
    f'Crisis:     {p99_crisis:.1f} bps\n'
    r'Ratio: $\times$6'
)
# Use plain text (no usetex needed)
callout_plain = (
    'P99 tail expansion\n'
    f'Pre-SVB:  {p99_pre:.1f} bps\n'
    f'Crisis:     {p99_crisis:.1f} bps\n'
    'Ratio: \u00d76'
)
ax.text(
    0.975, 0.96, callout_plain,
    transform=ax.transAxes, fontsize=9.5,
    va='top', ha='right', linespacing=1.55,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF5F5',
              edgecolor=CB_RED, linewidth=1.1, alpha=0.96),
)

# --- Kurtosis annotation ---
ax.text(
    0.025, 0.96,
    f'Excess kurtosis\nPre-SVB: {pre_bt.kurt():.1f}\n'
    f'Crisis:   {crisis_bt.kurt():.1f}',
    transform=ax.transAxes, fontsize=9.5,
    va='top', ha='left', linespacing=1.55,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5FF',
              edgecolor=CB_BLUE, linewidth=1.1, alpha=0.96),
)

# --- Axes ---
ax.set_xlim(x_lo, x_hi)
ax.set_xlabel(r'Adjusted Residual $B_{USDC,t}$ (bps)', labelpad=6)
ax.set_ylabel('Density', labelpad=6)
ax.yaxis.grid(True, linewidth=0.4, color='#EEEEEE', zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='upper center', frameon=True, framealpha=0.92,
          edgecolor='#CCCCCC', fontsize=9.5)

ax.set_title(
    r'Tail Blowout: Distribution of $B_{USDC,t}$ (Kraken) — Pre-SVB vs.\ Crisis',
    pad=10,
)

fig.tight_layout()
fig.savefig(
    os.path.join(FIGURES_DIR, 'fig_tail_blowout_kde.png'),
    dpi=300, bbox_inches='tight',
)
plt.close(fig)
print('Figure 2 saved  →  fig_tail_blowout_kde.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Stablecoin Substitution Scatter
#   X: D_{USDC,t}  (unadjusted USDC dispersion, Kraken)
#   Y: D_{USDT,t}  (unadjusted USDT dispersion, Kraken)
#   Color: Pre-SVB (gray) vs. Crisis (red)
# ══════════════════════════════════════════════════════════════════════════════

# --- Build scatter frame (Pre-SVB + Crisis only) ---
scatter_df = basis.loc[
    pre_mask | crisis_mask,
    ['dispersion_usdc_kraken', 'dispersion_usdt_kraken'],
].copy().dropna()
scatter_df.columns = ['D_USDC', 'D_USDT']
scatter_df['Regime'] = np.where(
    (scatter_df.index >= svb_start) & (scatter_df.index < svb_end),
    'Crisis', 'Pre-SVB',
)

# --- Per-regime correlation numbers (D_t and B_t from paper) ---
def corr_pair(df, regime, ca, cb):
    sub = df.loc[df['Regime'] == regime, [ca, cb]].dropna()
    return sub.corr().iloc[0, 1], len(sub)

r_dt_pre,    n_pre    = corr_pair(scatter_df, 'Pre-SVB', 'D_USDC', 'D_USDT')
r_dt_crisis, n_crisis = corr_pair(scatter_df, 'Crisis',  'D_USDC', 'D_USDT')

# --- Clip to 98th pctile of the crisis USDC dispersion for a clean view ---
x_clip = scatter_df.loc[scatter_df['Regime'] == 'Crisis', 'D_USDC'].quantile(0.97)
y_lo_c = scatter_df.loc[scatter_df['Regime'] == 'Crisis', 'D_USDT'].quantile(0.01)
plot_df = scatter_df.loc[
    (scatter_df['D_USDC'] >= -30) & (scatter_df['D_USDC'] <= x_clip) &
    (scatter_df['D_USDT'] >= y_lo_c) & (scatter_df['D_USDT'] <= 35)
]

fig, ax = plt.subplots(figsize=(8, 7))

# --- Plot Pre-SVB (bottom layer) ---
pre_plot = plot_df.loc[plot_df['Regime'] == 'Pre-SVB']
ax.scatter(
    pre_plot['D_USDC'], pre_plot['D_USDT'],
    color=CB_GRAY, s=10, alpha=0.55, linewidths=0,
    zorder=2, label=f'Pre-SVB  ($n={len(pre_plot):,}$)',
)

# --- Plot Crisis (top layer) ---
cri_plot = plot_df.loc[plot_df['Regime'] == 'Crisis']
ax.scatter(
    cri_plot['D_USDC'], cri_plot['D_USDT'],
    color=CB_RED, s=10, alpha=0.50, linewidths=0,
    zorder=3, label=f'Crisis  ($n={len(cri_plot):,}$)',
)

# --- Reference lines ---
ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.35, zorder=1)
ax.axvline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.35, zorder=1)

# --- Regime centroid markers ---
for regime, colour, marker in [('Pre-SVB', '#555555', 'D'), ('Crisis', '#AA0000', 'D')]:
    sub = plot_df.loc[plot_df['Regime'] == regime]
    ax.scatter(sub['D_USDC'].mean(), sub['D_USDT'].mean(),
               color=colour, s=120, marker=marker, zorder=5, edgecolors='white',
               linewidths=1.2, label=f'{regime} centroid')

# --- Correlation annotation box ---
annot = (
    r'Pearson $\rho$ (unadj. $D_t$):' '\n'
    f'Pre-SVB:  {r_dt_pre:+.2f}\n'
    f'Crisis:     {r_dt_crisis:+.2f}\n'
    '\n'
    r'Pearson $\rho$ (adj. $B_t$, paper):' '\n'
    r'Pre-SVB:  $+$0.44' '\n'
    r'Crisis:     $+$0.16'
)
ax.text(
    0.975, 0.975, annot,
    transform=ax.transAxes, fontsize=9.2,
    va='top', ha='right', linespacing=1.5,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
              edgecolor='#CCCCCC', linewidth=0.9, alpha=0.96),
)

# --- Axes labels ---
ax.set_xlabel(r'USDC Unadjusted Dispersion $D_{USDC,t}$ (bps)', labelpad=6)
ax.set_ylabel(r'USDT Unadjusted Dispersion $D_{USDT,t}$ (bps)', labelpad=6)

# --- Subtle horizontal grid ---
ax.yaxis.grid(True, linewidth=0.4, color='#EEEEEE', zorder=0)
ax.set_axisbelow(True)

# --- Legend: fix icon size independently of scatter marker size ---
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=CB_GRAY,
           markersize=7, label=f'Pre-SVB  ($n={len(pre_plot):,}$)', alpha=0.8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=CB_RED,
           markersize=7, label=f'Crisis  ($n={len(cri_plot):,}$)', alpha=0.8),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#555555',
           markeredgecolor='white', markeredgewidth=0.8,
           markersize=8, label='Pre-SVB centroid'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#AA0000',
           markeredgecolor='white', markeredgewidth=0.8,
           markersize=8, label='Crisis centroid'),
]
ax.legend(handles=legend_elements, loc='upper left',
          frameon=True, framealpha=0.92,
          edgecolor='#CCCCCC', fontsize=9.5)

ax.set_title(
    r'Stablecoin Substitution: $D_{USDC,t}$ vs.\ $D_{USDT,t}$ (Kraken)'
    '\n(Pre-SVB: Gray  |  Crisis: Red)',
    pad=10,
)

fig.tight_layout()
fig.savefig(
    os.path.join(FIGURES_DIR, 'fig_stablecoin_substitution_scatter.png'),
    dpi=300, bbox_inches='tight',
)
plt.close(fig)
print('Figure 3 saved  →  fig_stablecoin_substitution_scatter.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Cross-Channel Correlation Regime Heatmap
#   Three panels: Pre-SVB / Crisis / Post-SVB
#   Shows decorrelation of adjusted residuals during stress
# ══════════════════════════════════════════════════════════════════════════════

post_start = pd.Timestamp('2023-03-13', tz='UTC')
post_end   = pd.Timestamp('2023-03-22', tz='UTC')
post_mask  = (basis.index >= post_start) & (basis.index < post_end)

# Channels for correlation matrix
corr_cols = [
    'basis_usdc_kraken',
    'basis_usdt_kraken',
    'basis_usdt_coinbase',
    'xbasis_btcusdt_binance_kraken',
    'xbasis_btcusd_coinbase_kraken',
]
corr_labels = [
    r'$B_{USDC}$ KR',
    r'$B_{USDT}$ KR',
    r'$B_{USDT}$ CB',
    r'X-USDT BN-KR',
    r'X-USD CB-KR',
]

regimes = [
    ('Pre-SVB',  pre_mask),
    ('Crisis',   crisis_mask),
    ('Post-SVB', post_mask),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                         gridspec_kw={'wspace': 0.15})

for ax_i, (regime_name, mask) in zip(axes, regimes):
    sub = basis.loc[mask, corr_cols].dropna()
    corr_mat = sub.corr()

    im = ax_i.imshow(corr_mat.values, vmin=-0.2, vmax=1.0,
                     cmap='RdBu_r', aspect='equal')

    n_ch = len(corr_cols)
    for i in range(n_ch):
        for j in range(n_ch):
            val = corr_mat.values[i, j]
            color = 'white' if abs(val) > 0.55 else 'black'
            ax_i.text(j, i, f'{val:.2f}', ha='center', va='center',
                      fontsize=8.5, color=color, fontweight='bold')

    ax_i.set_xticks(range(n_ch))
    ax_i.set_yticks(range(n_ch))
    ax_i.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=8)
    ax_i.set_yticklabels(corr_labels if ax_i == axes[0] else [], fontsize=8)
    ax_i.set_title(f'{regime_name}\n($n={len(sub):,}$)', fontsize=11, pad=8)

    # Remove spines
    for spine in ax_i.spines.values():
        spine.set_visible(False)
    ax_i.tick_params(length=0)

# Colorbar
cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.04, shrink=0.82)
cbar.set_label('Pearson Correlation', fontsize=10)

fig.suptitle(
    'Cross-Channel Correlation by Regime: Adjusted Residuals and Cross-Exchange Basis',
    fontsize=13, y=1.02,
)

fig.savefig(
    os.path.join(FIGURES_DIR, 'fig_correlation_regime_heatmap.png'),
    dpi=300, bbox_inches='tight',
)
plt.close(fig)
print('Figure 4 saved  →  fig_correlation_regime_heatmap.png')

print('\nAll four academic figures generated successfully.')
