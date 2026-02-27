"""
08_column_figures.py
====================
Generate single-column-optimised figures for IAQF_column_Final.tex.

Every figure is designed to sit in ONE column at [H] placement
(width = columnwidth ~ 3.36" on A4 with the current margins).

Design targets
--------------
  All figures: matplotlib figsize (3.4, h)  → ~1:1 display scale
  Fonts: base 7 pt, axes labels 7.5 pt, title 8 pt, legend 6.5 pt

  Multi-panel figures that were previously horizontal are STACKED VERTICALLY
  so they still fit comfortably within one column.
  Exceptions (kept side-by-side because each cell is compact):
    • fig_half_life_robustness   – dot-plot, 3 regime cols
    • fig_correlation_regime_heatmap – heatmap, 3 regime cols

Output: figures_col/   (referenced by IAQF_column_Final.tex)

Run from the project root:
    cd /path/to/IAQF2026
    python src/08_column_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.lines import Line2D

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(ROOT, 'data_processed')
TABLES_DIR     = os.path.join(ROOT, 'tables')
FIGURES_COL    = os.path.join(ROOT, 'figures_col')
os.makedirs(FIGURES_COL, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data…")
prices  = pd.read_parquet(os.path.join(DATA_PROCESSED, 'prices.parquet'))
volumes = pd.read_parquet(os.path.join(DATA_PROCESSED, 'volumes.parquet'))
basis   = pd.read_parquet(os.path.join(DATA_PROCESSED, 'basis.parquet'))
ranges  = pd.read_parquet(os.path.join(DATA_PROCESSED, 'intraminute_ranges.parquet'))

# ── Regime boundaries ─────────────────────────────────────────────────────────
svb_start = pd.Timestamp('2023-03-10', tz='UTC')
svb_end   = pd.Timestamp('2023-03-13', tz='UTC')

# ── Colour palette (Paul Tol "bright" – colour-blind safe) ────────────────────
CB_BLUE   = '#4477AA'
CB_RED    = '#EE6677'
CB_GRAY   = '#BBBBBB'
CB_ORANGE = '#CCBB44'
CB_GREEN  = '#228833'
CB_PURPLE = '#AA3377'
CB_CYAN   = '#66CCEE'

REGIME_COLORS = {'Pre-SVB': CB_BLUE, 'Crisis': CB_RED, 'Post-SVB': CB_GREEN}

# ── Single-column RC (applies to EVERY figure) ────────────────────────────────
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

# ── Helpers ───────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(FIGURES_COL, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved → {name}")

def shade_crisis(ax):
    ax.axvspan(svb_start, svb_end, alpha=0.15, color='red', zorder=0)

def fmt_date(ax, fmt='%b %d', rotation=30):
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation, ha='right')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1  fig_stablecoin_peg   —  2 panels STACKED vertically  (3.4 × 4.0)
# ══════════════════════════════════════════════════════════════════════════════
def fig_stablecoin_peg():
    ps = pd.Timestamp('2023-03-09', tz='UTC')
    pe = pd.Timestamp('2023-03-16', tz='UTC')
    pz = prices.loc[(prices.index >= ps) & (prices.index <= pe)]
    bz = basis.loc[(basis.index  >= ps) & (basis.index  <= pe)]

    with plt.rc_context(COL_RC):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.0), sharex=True)
        for ax in (ax1, ax2):
            shade_crisis(ax)
            ax.set_xlim(ps, pe)

        # Panel A – spot prices
        for col, lbl, c in [
            ('kraken_usdcusd',   'USDC/USD (Kraken)',   CB_GREEN),
            ('kraken_usdtusd',   'USDT/USD (Kraken)',   CB_BLUE),
            ('coinbase_usdtusd', 'USDT/USD (Coinbase)', CB_RED),
        ]:
            if col in pz.columns:
                ax1.plot(pz.index, pz[col], linewidth=0.65, alpha=0.9, label=lbl, color=c)
        ax1.axhline(1.0, color='grey', linewidth=0.7, linestyle='--', alpha=0.5)
        ax1.set_ylabel('Price (USD)')
        ax1.set_ylim(0.85, 1.04)
        ax1.set_title('(A) Stablecoin Spot Prices')
        ax1.legend(loc='lower right', frameon=True, framealpha=0.9,
                   fontsize=6, ncol=1)

        # Panel B – peg deviations
        ax2.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4)
        for col, lbl, c in [
            ('usdc_peg_dev_kraken',   'USDC Dev (Kraken)',   CB_GREEN),
            ('usdt_peg_dev_kraken',   'USDT Dev (Kraken)',   CB_BLUE),
            ('usdt_peg_dev_coinbase', 'USDT Dev (Coinbase)', CB_RED),
        ]:
            if col in bz.columns:
                ax2.plot(bz.index, bz[col], linewidth=0.65, alpha=0.9, label=lbl, color=c)
        ax2.set_ylabel('Deviation (bps)')
        ax2.set_title('(B) Peg Deviations from $1.00')
        ax2.legend(loc='lower right', frameon=True, framealpha=0.9,
                   fontsize=6, ncol=1)
        fmt_date(ax2)

        fig.tight_layout(pad=0.5, h_pad=0.7)
        save(fig, 'fig_stablecoin_peg.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2  fig_dispersion_vs_adjusted_kraken  —  2 panels STACKED  (3.4 × 4.5)
# ══════════════════════════════════════════════════════════════════════════════
def fig_dispersion_vs_adjusted():
    zs = pd.Timestamp('2023-03-10', tz='UTC')
    ze = pd.Timestamp('2023-03-13 23:59', tz='UTC')
    bz = basis.loc[(basis.index >= zs) & (basis.index <= ze)]

    with plt.rc_context(COL_RC):
        fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.5), sharex=True)
        for ax in axes:
            ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4)
            shade_crisis(ax)

        for ax, d_col, b_col, title in [
            (axes[0], 'dispersion_usdc_kraken', 'basis_usdc_kraken',
             '(A) USDC Channel (Kraken)'),
            (axes[1], 'dispersion_usdt_kraken', 'basis_usdt_kraken',
             '(B) USDT Channel (Kraken)'),
        ]:
            if d_col in bz.columns:
                ax.plot(bz.index, bz[d_col], color=CB_ORANGE, linewidth=0.65,
                        alpha=0.9, label=r'$D_t$ (unadj.)')
            if b_col in bz.columns:
                ax.plot(bz.index, bz[b_col], color=CB_BLUE, linewidth=0.65,
                        alpha=0.9, label=r'$B_t$ (adj.)')
            ax.set_ylabel('bps')
            ax.set_title(title)
            ax.legend(loc='upper right', ncol=2, frameon=True,
                      framealpha=0.9, fontsize=6)

        fmt_date(axes[1], fmt='%b %d\n%H:%M', rotation=0)
        fig.tight_layout(pad=0.5, h_pad=0.7)
        save(fig, 'fig_dispersion_vs_adjusted_kraken.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3  fig_stablecoin_substitution_scatter  —  1 panel  (3.4 × 3.8)
# ══════════════════════════════════════════════════════════════════════════════
def fig_substitution_scatter():
    pre_mask    = (basis.index >= pd.Timestamp('2023-03-01', tz='UTC')) & \
                  (basis.index < svb_start)
    crisis_mask = (basis.index >= svb_start) & (basis.index < svb_end)

    scatter_df = basis.loc[
        pre_mask | crisis_mask,
        ['dispersion_usdc_kraken', 'dispersion_usdt_kraken'],
    ].copy().dropna()
    scatter_df.columns = ['D_USDC', 'D_USDT']
    scatter_df['Regime'] = np.where(
        (scatter_df.index >= svb_start) & (scatter_df.index < svb_end),
        'Crisis', 'Pre-SVB')

    r_pre    = scatter_df.loc[scatter_df['Regime']=='Pre-SVB',
               ['D_USDC','D_USDT']].corr().iloc[0,1]
    r_crisis = scatter_df.loc[scatter_df['Regime']=='Crisis',
               ['D_USDC','D_USDT']].corr().iloc[0,1]

    x_clip = scatter_df.loc[scatter_df['Regime']=='Crisis','D_USDC'].quantile(0.97)
    y_lo   = scatter_df.loc[scatter_df['Regime']=='Crisis','D_USDT'].quantile(0.01)
    plot_df = scatter_df.loc[
        (scatter_df['D_USDC'] >= -30) & (scatter_df['D_USDC'] <= x_clip) &
        (scatter_df['D_USDT'] >= y_lo) & (scatter_df['D_USDT'] <= 35)]

    with plt.rc_context(COL_RC):
        fig, ax = plt.subplots(figsize=(3.4, 3.8))
        pre_p = plot_df.loc[plot_df['Regime']=='Pre-SVB']
        cri_p = plot_df.loc[plot_df['Regime']=='Crisis']

        ax.scatter(pre_p['D_USDC'], pre_p['D_USDT'],
                   color=CB_GRAY, s=4, alpha=0.55, linewidths=0, zorder=2)
        ax.scatter(cri_p['D_USDC'], cri_p['D_USDT'],
                   color=CB_RED,  s=4, alpha=0.50, linewidths=0, zorder=3)
        for regime, colour in [('Pre-SVB','#555555'),('Crisis','#AA0000')]:
            sub = plot_df.loc[plot_df['Regime']==regime]
            ax.scatter(sub['D_USDC'].mean(), sub['D_USDT'].mean(),
                       color=colour, s=50, marker='D', zorder=5,
                       edgecolors='white', linewidths=0.7)

        ax.axhline(0, color='black', linewidth=0.4, linestyle='--', alpha=0.35)
        ax.axvline(0, color='black', linewidth=0.4, linestyle='--', alpha=0.35)

        annot = (r'$\rho$ (unadj. $D_t$):' '\n'
                 f'Pre-SVB: {r_pre:+.2f}\n'
                 f'Crisis:   {r_crisis:+.2f}\n\n'
                 r'$\rho$ (adj. $B_t$):' '\n'
                 r'Pre-SVB: $+$0.44' '\n'
                 r'Crisis:   $+$0.16')
        ax.text(0.97, 0.97, annot, transform=ax.transAxes,
                fontsize=6, va='top', ha='right', linespacing=1.4,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='#CCCCCC', linewidth=0.6, alpha=0.95))

        legend_els = [
            Line2D([0],[0],marker='o',color='w',markerfacecolor=CB_GRAY,
                   markersize=5,label=f'Pre-SVB ($n={len(pre_p):,}$)',alpha=0.8),
            Line2D([0],[0],marker='o',color='w',markerfacecolor=CB_RED,
                   markersize=5,label=f'Crisis ($n={len(cri_p):,}$)',alpha=0.8),
            Line2D([0],[0],marker='D',color='w',markerfacecolor='#555555',
                   markeredgecolor='white',markeredgewidth=0.5,markersize=5,
                   label='Pre-SVB centroid'),
            Line2D([0],[0],marker='D',color='w',markerfacecolor='#AA0000',
                   markeredgecolor='white',markeredgewidth=0.5,markersize=5,
                   label='Crisis centroid'),
        ]
        ax.legend(handles=legend_els, loc='upper left',
                  frameon=True, framealpha=0.9, edgecolor='#CCCCCC', fontsize=6)
        ax.set_xlabel(r'$D_{USDC,t}$ (bps)')
        ax.set_ylabel(r'$D_{USDT,t}$ (bps)')
        ax.set_title(r'Stablecoin Substitution ($D_t$, Kraken)')
        ax.yaxis.grid(True, linewidth=0.3, color='#EEEEEE')
        ax.set_axisbelow(True)
        fig.tight_layout(pad=0.4)
        save(fig, 'fig_stablecoin_substitution_scatter.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4  fig_half_life_robustness  —  3 regime cols side-by-side  (3.4 × 2.6)
# (kept horizontal: dot-plot cells are compact enough at ~1.1" each)
# ══════════════════════════════════════════════════════════════════════════════
def fig_half_life_robustness():
    csv_path = os.path.join(TABLES_DIR, 'half_life_robustness.csv')
    if not os.path.exists(csv_path):
        print("  SKIP fig_half_life_robustness (CSV not found)")
        return
    df = pd.read_csv(csv_path)
    df['spec']       = df['freq'] + ' ' + df['ff_filter']
    df['stablecoin'] = df['series'].str.extract(r'(USDC|USDT)')[0]

    regimes = ['Pre-SVB', 'Crisis', 'Post-SVB']
    specs   = ['1m all', '1m no_ff', '5m all', '5m no_ff']
    colors  = {'USDC': CB_BLUE, 'USDT': CB_RED}
    markers = {'USDC': 'o', 'USDT': 's'}

    with plt.rc_context(COL_RC):
        fig, axes = plt.subplots(1, 3, figsize=(3.4, 2.6), sharey=True)
        for ax, regime in zip(axes, regimes):
            sub = df[df['regime'] == regime]
            for j, coin in enumerate(['USDC', 'USDT']):
                csub = sub[sub['stablecoin'] == coin]
                for i, spec in enumerate(specs):
                    row = csub[csub['spec'] == spec]
                    x   = i + (j - 0.5) * 0.20
                    if len(row) == 0 or pd.isna(row['half_life_min'].values[0]):
                        ax.scatter(x, 0, marker='x', color=colors[coin],
                                   s=20, zorder=5)
                        ax.annotate('NaN', (x, 0.1), fontsize=5, ha='center',
                                    color=colors[coin])
                    else:
                        val = row['half_life_min'].values[0]
                        ax.scatter(x, val, marker=markers[coin],
                                   color=colors[coin], s=22, zorder=5,
                                   edgecolors='k', linewidths=0.3)
                        ax.annotate(f'{val:.1f}', (x, val+0.09),
                                    fontsize=5, ha='center', color=colors[coin])
            ax.set_title(regime, fontsize=7, fontweight='bold')
            ax.set_xticks(range(len(specs)))
            ax.set_xticklabels(specs, fontsize=5.5, rotation=35, ha='right')
            ax.set_xlim(-0.5, len(specs)-0.5)
            ax.grid(axis='y', alpha=0.3, linewidth=0.3)

        axes[0].set_ylabel('Half-Life (min)', fontsize=7)
        legend_els = [
            Line2D([0],[0],marker='o',color='w',markerfacecolor=CB_BLUE,
                   markeredgecolor='k',markersize=5,label='USDC $B_t$'),
            Line2D([0],[0],marker='s',color='w',markerfacecolor=CB_RED,
                   markeredgecolor='k',markersize=5,label='USDT $B_t$'),
        ]
        fig.legend(handles=legend_els, loc='lower center', ncol=2,
                   fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle('Half-Life Robustness: Adjusted Residual $B_t$',
                     fontsize=8, fontweight='bold', y=1.02)
        fig.tight_layout(pad=0.4, w_pad=0.3)
        save(fig, 'fig_half_life_robustness.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5  fig_two_layer_persistence  —  1 panel dual-axis  (3.4 × 3.0)
# ══════════════════════════════════════════════════════════════════════════════
def fig_two_layer_persistence():
    t0 = pd.Timestamp('2023-03-10', tz='UTC')
    t1 = pd.Timestamp('2023-03-13', tz='UTC')
    d  = basis.loc[(basis.index >= t0) & (basis.index < t1)]

    with plt.rc_context(COL_RC):
        fig, ax1 = plt.subplots(figsize=(3.4, 3.0))

        line_bt, = ax1.plot(d.index, d['basis_usdc_kraken'],
                            color=CB_BLUE, linewidth=0.55, alpha=0.90, zorder=3,
                            label=r'$B_{USDC,t}$ (Kraken, left)')
        ax1.axhline(0, color='black', linewidth=0.4, linestyle='--', alpha=0.35)
        ax1.set_ylabel(r'$B_{USDC,t}$ (bps)', color=CB_BLUE, labelpad=3)
        ax1.tick_params(axis='y', labelcolor=CB_BLUE)
        ax1.set_ylim(-130, 210)
        ax1.yaxis.set_major_locator(mticker.MultipleLocator(50))

        ax2 = ax1.twinx()
        ax2.spines['right'].set_visible(True)
        ax2.spines['right'].set_linewidth(0.6)
        line_peg, = ax2.plot(d.index, d['usdc_peg_dev_kraken'],
                             color=CB_RED, linewidth=0.9, alpha=0.92, zorder=2,
                             label='Peg Dev. (right)')
        ax2.set_ylabel('USDC Peg Dev. (bps)', color=CB_RED, labelpad=3)
        ax2.tick_params(axis='y', labelcolor=CB_RED)
        ax2.set_ylim(-1450, 230)
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(250))

        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), ha='center', fontsize=6)
        ax1.set_xlim(t0, t1)
        ax1.set_xlabel('Date / Time (UTC)', labelpad=3)

        lines  = [line_bt, line_peg]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', frameon=True,
                   framealpha=0.92, edgecolor='#CCCCCC', fontsize=6)

        ax1.text(0.995, 0.97,
            r'HL gap: $\times$940' '\n'
            r'$B_{USDC,t}$: $\approx$0.6 min' '\n'
            r'Peg dev: $\approx$572 min',
            transform=ax1.transAxes, fontsize=6,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F7F7F7',
                      edgecolor='#CCCCCC', linewidth=0.6, alpha=0.95))

        ax1.set_title(
            r'Two-Layer Persistence: $B_{USDC,t}$ vs.\ Peg Dev.' '\n'
            '(Kraken, 10--13 Mar 2023)', pad=4)
        fig.tight_layout(pad=0.5)
        save(fig, 'fig_two_layer_persistence.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6  fig_cross_exchange_basis  —  3 panels STACKED vertically  (3.4 × 5.5)
# ══════════════════════════════════════════════════════════════════════════════
def fig_cross_exchange_basis():
    zs = pd.Timestamp('2023-03-09', tz='UTC')
    ze = pd.Timestamp('2023-03-21 23:59', tz='UTC')
    bz = basis.loc[(basis.index >= zs) & (basis.index <= ze)]

    specs = [
        ('xbasis_btcusdt_binance_kraken',  '(A) BTC/USDT Binance−Kraken', CB_ORANGE),
        ('xbasis_btcusdt_coinbase_kraken',  '(B) BTC/USDT Coinbase−Kraken', CB_CYAN),
        ('xbasis_btcusd_coinbase_kraken',   '(C) BTC/USD Coinbase−Kraken',  CB_PURPLE),
    ]

    with plt.rc_context(COL_RC):
        fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.5), sharex=True)
        for ax, (col, title, c) in zip(axes, specs):
            shade_crisis(ax)
            ax.axhline(0, color='black', linewidth=0.4, linestyle='--', alpha=0.4)
            ax.set_xlim(zs, ze)
            if col in bz.columns:
                ax.plot(bz.index, bz[col], linewidth=0.5, color=c, alpha=0.85)
            ax.set_title(title, fontsize=7.5)
            ax.set_ylabel('Basis (bps)')

        fmt_date(axes[2])
        fig.tight_layout(pad=0.5, h_pad=0.6)
        save(fig, 'fig_cross_exchange_basis.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7  fig_liquidity_roll_amihud  —  2 panels STACKED vertically  (3.4 × 4.5)
# ══════════════════════════════════════════════════════════════════════════════
def fig_liquidity_roll_amihud():
    REGIME_ORDER = ['Pre-SVB', 'Crisis', 'Post-SVB']
    PAIRS = {
        'Kraken BTC/USD':   ('kraken_btcusd',   'kraken_btcusd'),
        'Kraken BTC/USDT':  ('kraken_btcusdt',  'kraken_btcusdt'),
        'Kraken BTC/USDC':  ('kraken_btcusdc',  'kraken_btcusdc'),
        'Binance BTC/USDT': ('binance_btcusdt', 'binance_btcusdt'),
        'Coinbase BTC/USD': ('coinbase_btcusd', 'coinbase_btcusd'),
    }
    PAIR_ORDER = list(PAIRS.keys())
    short_names = ['KR/USD', 'KR/USDT', 'KR/USDC', 'BN/USDT', 'CB/USD']

    def roll_spread_daily(price_col):
        p  = prices[price_col].dropna()
        lr = np.log(p / p.shift(1))
        rows = []
        for date, grp in lr.groupby(lr.index.date):
            r = grp.dropna().values
            if len(r) < 15: continue
            cov = np.cov(r[1:], r[:-1])[0, 1]
            rows.append({'date': pd.Timestamp(date),
                         'roll_bps': 2.0*np.sqrt(-cov)*10000 if cov < 0 else np.nan})
        if not rows: return pd.Series(dtype=float)
        s = pd.DataFrame(rows).set_index('date')['roll_bps']
        s.index = pd.DatetimeIndex(s.index).tz_localize('UTC')
        return s

    def amihud_daily(price_col, vol_col):
        if vol_col not in volumes.columns: return pd.Series(dtype=float)
        p    = prices[price_col]; v = volumes[vol_col]
        lr   = np.log(p / p.shift(1)).abs()
        dvol = v * p
        aligned = pd.concat([lr, dvol], axis=1, keys=['abs_ret','dvol']).dropna()
        aligned  = aligned[aligned['dvol'] > 1.0]
        aligned['illiq'] = aligned['abs_ret'] / aligned['dvol']
        daily = aligned.groupby(aligned.index.date)['illiq'].mean() * 1e6
        s = daily.copy(); s.index = pd.DatetimeIndex(s.index).tz_localize('UTC')
        return s

    roll_series   = {lbl: roll_spread_daily(pc) for lbl, (pc, _)  in PAIRS.items()
                     if pc in prices.columns}
    amihud_series = {lbl: amihud_daily(pc, vc) for lbl, (pc, vc) in PAIRS.items()
                     if pc in prices.columns}

    def regime_stats(daily_dict):
        rows = []
        for lbl in PAIR_ORDER:
            if lbl not in daily_dict: continue
            s = daily_dict[lbl]
            for reg in REGIME_ORDER:
                if reg == 'Pre-SVB':  mask = s.index < svb_start
                elif reg == 'Crisis': mask = (s.index >= svb_start) & (s.index < svb_end)
                else:                 mask = s.index >= svb_end
                sub = s[mask].dropna()
                rows.append({'Pair': lbl, 'Regime': reg,
                             'mean': round(sub.mean(), 3) if len(sub) else np.nan,
                             'N':    len(sub)})
        return pd.DataFrame(rows)

    df_roll   = regime_stats(roll_series)
    df_amihud = regime_stats(amihud_series)

    reg_colors = {'Pre-SVB': CB_BLUE, 'Crisis': CB_RED, 'Post-SVB': CB_GREEN}
    w = 0.22

    def grouped_bar(ax, df, ylabel, title):
        pivot = df.pivot(index='Pair', columns='Regime', values='mean')[REGIME_ORDER]
        pivot = pivot.reindex(PAIR_ORDER)
        x = np.arange(len(pivot))
        for i, reg in enumerate(REGIME_ORDER):
            ax.bar(x + (i-1)*w, pivot[reg], w,
                   label=reg, color=reg_colors[reg], alpha=0.85,
                   edgecolor='white', linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=6)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(title, fontsize=7.5)
        ax.legend(fontsize=6, ncol=3)
        ax.grid(axis='y', linewidth=0.3, alpha=0.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    with plt.rc_context(COL_RC):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.5))
        grouped_bar(ax1, df_roll,
                    'Roll Spread (bps)',
                    '(A) Roll (1984) Effective Spread')
        grouped_bar(ax2, df_amihud,
                    r'Amihud ILLIQ ($\times10^{-6}$)',
                    '(B) Amihud (2002) ILLIQ Ratio')
        fig.tight_layout(pad=0.5, h_pad=0.8)
        save(fig, 'fig_liquidity_roll_amihud.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8  fig_volume_share  —  1 panel  (3.4 × 3.0)
# ══════════════════════════════════════════════════════════════════════════════
def fig_volume_share():
    vol_cols  = ['binance_btcusdt', 'binance_btcusdc', 'coinbase_btcusd',
                 'coinbase_btcusdt', 'kraken_btcusd', 'kraken_btcusdt', 'kraken_btcusdc']
    available = [c for c in vol_cols if c in volumes.columns]
    if not available:
        print("  SKIP fig_volume_share (no volume columns found)"); return
    vols_daily = volumes[available].resample('D').sum()
    vols_pct   = vols_daily.div(vols_daily.sum(axis=1), axis=0) * 100

    label_map = {
        'binance_btcusdt': 'Binance USDT',  'binance_btcusdc': 'Binance USDC',
        'coinbase_btcusd': 'Coinbase USD',  'coinbase_btcusdt': 'Coinbase USDT',
        'kraken_btcusd':   'Kraken USD',    'kraken_btcusdt':   'Kraken USDT',
        'kraken_btcusdc':  'Kraken USDC',
    }
    colors_vol = [CB_BLUE, CB_CYAN, CB_ORANGE, CB_PURPLE, '#444444', CB_GREEN, CB_RED]

    with plt.rc_context(COL_RC):
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        ax.stackplot(
            vols_pct.index,
            *[vols_pct[c] for c in available],
            labels=[label_map.get(c, c) for c in available],
            colors=colors_vol[:len(available)],
            alpha=0.82,
        )
        ax.axvspan(svb_start.normalize(), svb_end.normalize(),
                   alpha=0.25, color='red', label='SVB Crisis', zorder=10)
        ax.set_title('Daily Volume Fragmentation')
        ax.set_ylabel('Volume Share (%)')
        fmt_date(ax)
        ax.legend(loc='upper left', fontsize=5.5, frameon=True,
                  framealpha=0.9, ncol=2)
        fig.tight_layout(pad=0.4)
        save(fig, 'fig_volume_share.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 9  fig_realized_volatility  —  1 panel  (3.4 × 3.0)
# ══════════════════════════════════════════════════════════════════════════════
def fig_realized_volatility():
    vol_cols_btc = ['kraken_btcusd', 'kraken_btcusdt', 'kraken_btcusdc',
                    'binance_btcusdt', 'coinbase_btcusd']
    available = [c for c in vol_cols_btc if c in prices.columns]
    rv = prices[available].pct_change(fill_method=None).rolling(60).std() * 10000 * np.sqrt(60)

    nice_names = {
        'kraken_btcusd':  'Kraken USD',   'kraken_btcusdt': 'Kraken USDT',
        'kraken_btcusdc': 'Kraken USDC',  'binance_btcusdt':'Binance USDT',
        'coinbase_btcusd':'Coinbase USD',
    }
    colors_rv = ['#2c3e50', CB_BLUE, CB_GREEN, CB_ORANGE, CB_RED]

    with plt.rc_context(COL_RC):
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        shade_crisis(ax)
        for col, c in zip(available, colors_rv):
            ax.plot(rv.index, rv[col], linewidth=0.5, color=c,
                    label=nice_names.get(col, col), alpha=0.85)
        ax.set_title('Hourly Realized Volatility (60-min, $\\times\\sqrt{60}$)')
        ax.set_ylabel('Volatility (bps/hr)')
        fmt_date(ax)
        ax.legend(loc='upper right', fontsize=6, ncol=1, frameon=True)
        fig.tight_layout(pad=0.4)
        save(fig, 'fig_realized_volatility.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10  fig_tail_blowout_kde  —  1 panel  (3.4 × 3.2)
# ══════════════════════════════════════════════════════════════════════════════
def fig_tail_blowout_kde():
    pre_mask    = (basis.index >= pd.Timestamp('2023-03-01', tz='UTC')) & \
                  (basis.index < svb_start)
    crisis_mask = (basis.index >= svb_start) & (basis.index < svb_end)
    pre_bt    = basis.loc[pre_mask,    'basis_usdc_kraken'].dropna()
    crisis_bt = basis.loc[crisis_mask, 'basis_usdc_kraken'].dropna()
    x_lo, x_hi = -50, 100

    with plt.rc_context(COL_RC):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))

        sns.kdeplot(pre_bt,    ax=ax, color=CB_GRAY, linewidth=1.4,
                    fill=True, alpha=0.40,
                    label=f'Pre-SVB ($n={len(pre_bt):,}$)',
                    clip=(x_lo, x_hi), bw_adjust=0.9)
        sns.kdeplot(crisis_bt, ax=ax, color=CB_RED,  linewidth=1.4,
                    fill=True, alpha=0.35,
                    label=f'Crisis ($n={len(crisis_bt):,}$)',
                    clip=(x_lo, x_hi), bw_adjust=0.9)

        ax.axvline(0,               color='black',  linewidth=0.5, linestyle='--', alpha=0.45)
        ax.axvline(pre_bt.mean(),   color=CB_GRAY,  linewidth=1.0, linestyle=':',
                   label=f'Pre-SVB mean ({pre_bt.mean():.1f} bps)')
        ax.axvline(crisis_bt.mean(),color=CB_RED,   linewidth=1.0, linestyle=':',
                   label=f'Crisis mean ({crisis_bt.mean():.1f} bps)')

        p99_pre, p99_crisis = pre_bt.quantile(0.99), crisis_bt.quantile(0.99)
        ax.text(0.97, 0.97,
            f'P99 expansion\nPre-SVB: {p99_pre:.1f} bps\nCrisis: {p99_crisis:.1f} bps\nRatio: \u00d76',
            transform=ax.transAxes, fontsize=6,
            va='top', ha='right', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF5F5',
                      edgecolor=CB_RED, linewidth=0.7, alpha=0.96))

        ax.text(0.03, 0.97,
            f'Ex. kurtosis\nPre-SVB: {pre_bt.kurt():.1f}\nCrisis: {crisis_bt.kurt():.1f}',
            transform=ax.transAxes, fontsize=6,
            va='top', ha='left', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5FF',
                      edgecolor=CB_BLUE, linewidth=0.7, alpha=0.96))

        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel(r'$B_{USDC,t}$ (bps)')
        ax.set_ylabel('Density')
        ax.yaxis.grid(True, linewidth=0.3, color='#EEEEEE')
        ax.set_axisbelow(True)
        ax.legend(loc='upper center', frameon=True, framealpha=0.9,
                  edgecolor='#CCCCCC', fontsize=6)
        ax.set_title(r'Tail Blowout: $B_{USDC,t}$ (Kraken)', pad=4)
        fig.tight_layout(pad=0.4)
        save(fig, 'fig_tail_blowout_kde.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 11  fig_correlation_regime_heatmap  —  2 on top (Pre-SVB, Crisis) + 1 centred
# below (Post-SVB), triangle layout.  Each heatmap is forced square via aspect='equal'.
# ══════════════════════════════════════════════════════════════════════════════
def fig_correlation_regime_heatmap():
    pre_mask    = (basis.index >= pd.Timestamp('2023-03-01', tz='UTC')) & \
                  (basis.index < svb_start)
    crisis_mask = (basis.index >= svb_start) & (basis.index < svb_end)
    post_mask   = (basis.index >= pd.Timestamp('2023-03-13', tz='UTC')) & \
                  (basis.index < pd.Timestamp('2023-03-22', tz='UTC'))

    corr_cols = [
        'basis_usdc_kraken', 'basis_usdt_kraken', 'basis_usdt_coinbase',
        'xbasis_btcusdt_binance_kraken', 'xbasis_btcusd_coinbase_kraken',
    ]
    corr_labels = [
        r'$B_{USDC}$ KR', r'$B_{USDT}$ KR', r'$B_{USDT}$ CB',
        r'X-USDT BN-KR',  r'X-USD CB-KR',
    ]

    def _draw_heatmap(ax, mask, regime_name):
        sub = basis.loc[mask, corr_cols].dropna()
        corr_mat = sub.corr()
        im = ax.imshow(corr_mat.values, vmin=-0.2, vmax=1.0,
                       cmap='RdBu_r', aspect='equal')
        n = len(corr_cols)
        for i in range(n):
            for j in range(n):
                val   = corr_mat.values[i, j]
                color = 'white' if abs(val) > 0.55 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=5.5, color=color, fontweight='bold')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=5.5)
        ax.set_yticklabels(corr_labels, fontsize=5.5)
        ax.set_title(f'{regime_name}  ($n={len(sub):,}$)', fontsize=7, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        return im

    with plt.rc_context(COL_RC):
        # 2×2 grid; top-left=Pre-SVB, top-right=Crisis, bottom-centre=Post-SVB.
        # aspect='auto' on imshow, but we force equal axes via set_box_aspect(1)
        # so every heatmap is physically square and identical in size.
        fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.0),
                                 gridspec_kw={'hspace': 0.65, 'wspace': 0.45})

        ax_pre    = axes[0, 0]
        ax_crisis = axes[0, 1]
        # Hide top-right placeholder; Post-SVB goes in the centred bottom slot
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)

        # Add a new axes manually centred in the bottom row
        # GridSpec bottom-row spans cols 0–1, so centre = midpoint of the two col slots
        gs = axes[1, 0].get_gridspec()
        ax_post = fig.add_subplot(gs[1, 0:2])
        # Shrink it to the same width as one column slot and centre it
        pos0 = axes[0, 0].get_position()   # left panel position
        pos1 = axes[0, 1].get_position()   # right panel position
        panel_w = pos0.width
        panel_h = pos0.height
        centre_x = (pos0.x0 + pos1.x1) / 2.0
        pos_bot = axes[1, 0].get_position()
        ax_post.set_position([centre_x - panel_w / 2.0,
                               pos_bot.y0,
                               panel_w,
                               panel_h])

        im = _draw_heatmap(ax_pre,    pre_mask,    'Pre-SVB')
        _draw_heatmap(ax_crisis, crisis_mask, 'Crisis')
        _draw_heatmap(ax_post,   post_mask,   'Post-SVB')

        # Force all three to have identical square aspect
        for ax in (ax_pre, ax_crisis, ax_post):
            ax.set_box_aspect(1)

        # Colorbar anchored to the right of the top-right panel
        cbar_ax = fig.add_axes([pos1.x1 + 0.02, pos1.y0, 0.02, pos1.height])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Pearson Corr.', fontsize=6.5)
        cbar.ax.tick_params(labelsize=6)

        fig.suptitle('Cross-Channel Correlation by Regime',
                     fontsize=8, y=0.99)
        fig.savefig(os.path.join(FIGURES_COL, 'fig_correlation_regime_heatmap.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("  saved → fig_correlation_regime_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 12  fig_arbitrage_after_fees  —  2 panels STACKED vertically  (3.4 × 4.5)
# ══════════════════════════════════════════════════════════════════════════════
def fig_arbitrage_after_fees():
    FEE_BPS = 5.0
    arb_specs = [
        {'key':    'basis_usdc_kraken',
         'basis':  'basis_usdc_kraken',
         'label':  'USDC/USD KR (3-leg)',
         'n_legs': 3,
         'ranges': ['kraken_btcusdc', 'kraken_usdcusd', 'kraken_btcusd']},
        {'key':    'basis_usdt_kraken',
         'basis':  'basis_usdt_kraken',
         'label':  'USDT/USD KR (3-leg)',
         'n_legs': 3,
         'ranges': ['kraken_btcusdt', 'kraken_usdtusd', 'kraken_btcusd']},
        {'key':    'xbasis_btcusdt_binance_kraken',
         'basis':  'xbasis_btcusdt_binance_kraken',
         'label':  'USDT BN-KR (2-leg)',
         'n_legs': 2,
         'ranges': ['binance_btcusdt', 'kraken_btcusdt']},
        {'key':    'xbasis_btcusd_coinbase_kraken',
         'basis':  'xbasis_btcusd_coinbase_kraken',
         'label':  'USD CB-KR (2-leg)',
         'n_legs': 2,
         'ranges': ['coinbase_btcusd', 'kraken_btcusd']},
    ]
    colors_arb = [CB_ORANGE, CB_BLUE, CB_RED, '#2c3e50']
    chan_data  = {}
    for spec in arb_specs:
        if spec['basis'] not in basis.columns: continue
        if any(r not in ranges.columns for r in spec['ranges']): continue
        df = pd.DataFrame(index=basis.index)
        df['abs_basis'] = basis[spec['basis']].abs()
        for i, rc in enumerate(spec['ranges'], 1):
            df[f'lr{i}'] = ranges[rc] * 10000.0
        df = df.dropna()
        if df.empty: continue
        df['fee']     = spec['n_legs'] * FEE_BPS
        df['slip']    = 0.5 * df[[f'lr{i}' for i in range(1,len(spec['ranges'])+1)]].sum(axis=1)
        df['net_fee'] = (df['abs_basis'] - df['fee']).clip(lower=0.0)
        df['net_fs']  = (df['abs_basis'] - df['fee'] - df['slip']).clip(lower=0.0)
        chan_data[spec['key']] = (spec['label'], df)

    if not chan_data:
        print("  SKIP fig_arbitrage_after_fees (no channels available)"); return

    with plt.rc_context(COL_RC):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.5), sharex=True)
        for ax in (ax1, ax2):
            shade_crisis(ax)

        for (lbl, df), c in zip(chan_data.values(), colors_arb):
            ax1.plot(df.index, df['net_fee'], linewidth=0.45, color=c,
                     label=lbl, alpha=0.85)
            ax2.plot(df.index, df['net_fs'],  linewidth=0.45, color=c,
                     label=lbl, alpha=0.85)

        ax1.set_title(f'(A) Fee-Only Net Arb. ({FEE_BPS:.0f} bps/leg)')
        ax1.set_ylabel('Net Profit (bps)')
        ax1.legend(loc='upper left', fontsize=6, ncol=1, frameon=True)

        ax2.set_title('(B) Fee + Slippage Net Arb.')
        ax2.set_ylabel('Net Profit (bps)')
        fmt_date(ax2)

        fig.tight_layout(pad=0.5, h_pad=0.7)
        save(fig, 'fig_arbitrage_after_fees.png')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.chdir(ROOT)
    print(f"\nGenerating single-column figures → {FIGURES_COL}\n")

    fig_stablecoin_peg()
    fig_dispersion_vs_adjusted()
    fig_substitution_scatter()
    fig_half_life_robustness()
    fig_two_layer_persistence()
    fig_cross_exchange_basis()
    fig_liquidity_roll_amihud()
    fig_volume_share()
    fig_realized_volatility()
    fig_tail_blowout_kde()
    fig_correlation_regime_heatmap()
    fig_arbitrage_after_fees()

    print(f"\nAll figures written to {FIGURES_COL}/")
