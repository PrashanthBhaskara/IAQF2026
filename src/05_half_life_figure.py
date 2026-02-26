"""Generate a compact half-life robustness dot plot from half_life_robustness.csv."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'tables', 'half_life_robustness.csv')
OUT = os.path.join(ROOT, 'figures', 'fig_half_life_robustness.png')


def main():
    df = pd.read_csv(CSV)

    # Build spec label: freq + ff_filter
    df['spec'] = df['freq'] + ' ' + df['ff_filter']
    # Clean series name
    df['stablecoin'] = df['series'].str.extract(r'(USDC|USDT)')[0]

    regimes = ['Pre-SVB', 'Crisis', 'Post-SVB']
    specs = ['1m all', '1m no_ff', '5m all', '5m no_ff']
    coins = ['USDC', 'USDT']

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)

    colors = {'USDC': '#2166ac', 'USDT': '#b2182b'}
    markers = {'USDC': 'o', 'USDT': 's'}

    for ax, regime in zip(axes, regimes):
        sub = df[df['regime'] == regime]
        for j, coin in enumerate(coins):
            csub = sub[sub['stablecoin'] == coin]
            for i, spec in enumerate(specs):
                row = csub[csub['spec'] == spec]
                x = i + (j - 0.5) * 0.25
                if len(row) == 0 or pd.isna(row['half_life_min'].values[0]):
                    ax.scatter(x, 0, marker='x', color=colors[coin], s=60, zorder=5)
                    ax.annotate('NaN', (x, 0.15), fontsize=6, ha='center',
                                color=colors[coin])
                else:
                    val = row['half_life_min'].values[0]
                    ax.scatter(x, val, marker=markers[coin], color=colors[coin],
                               s=50, zorder=5, edgecolors='k', linewidths=0.5)
                    ax.annotate(f'{val:.2f}', (x, val + 0.12), fontsize=6,
                                ha='center', color=colors[coin])

        ax.set_title(regime, fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(specs)))
        ax.set_xticklabels(specs, fontsize=7, rotation=30, ha='right')
        ax.set_xlim(-0.5, len(specs) - 0.5)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Half-Life (minutes)', fontsize=9)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['USDC'],
               markeredgecolor='k', markersize=7, label='USDC $B_t$ (Kraken)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['USDT'],
               markeredgecolor='k', markersize=7, label='USDT $B_t$ (Kraken)'),
        Line2D([0], [0], marker='x', color='gray', markersize=7, linestyle='None',
               label='NaN (invalid ρ)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Half-Life Robustness: Adjusted Residual $B_t$', fontsize=11,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
