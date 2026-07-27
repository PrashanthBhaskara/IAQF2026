"""Import-safe deterministic rendering of the paper's frozen figure set."""

# Backend selection must precede pyplot, so the import blocks are intentional.
# ruff: noqa: E402, I001

from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from iaqf import metrics
from iaqf.config import CRISIS_END, CRISIS_START, EXPECTED_FIGURES, RepoPaths
from iaqf.data import AnalysisData

CB_BLUE = "#4477AA"
CB_RED = "#EE6677"
CB_GRAY = "#BBBBBB"
CB_ORANGE = "#CCBB44"
CB_GREEN = "#228833"
CB_PURPLE = "#AA3377"
CB_CYAN = "#66CCEE"
CB_DARK = "#332288"
REGIME_COLORS = {"Pre-SVB": CB_BLUE, "Crisis": CB_RED, "Post-SVB": CB_GREEN}
COL_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 0.65,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
}


def _save(figure: plt.Figure, name: str, figures_dir: Path) -> None:
    figure.savefig(figures_dir / name, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _shade_crisis(axis: plt.Axes) -> None:
    axis.axvspan(CRISIS_START, CRISIS_END, alpha=0.15, color="red", zorder=0)


def _format_date(
    axis: plt.Axes, date_format: str = "%b %d", rotation: float = 30
) -> None:
    axis.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.setp(axis.xaxis.get_majorticklabels(), rotation=rotation, ha="right")


def _annotation_box(
    facecolor: str,
    edgecolor: str,
    *,
    pad: float = 0.3,
    linewidth: float = 0.7,
    alpha: float = 0.96,
) -> dict[str, str | float]:
    return {
        "boxstyle": f"round,pad={pad}",
        "facecolor": facecolor,
        "edgecolor": edgecolor,
        "linewidth": linewidth,
        "alpha": alpha,
    }


def _legend_marker(
    marker: str,
    facecolor: str,
    label: str,
    *,
    edgecolor: str | None = None,
    edgewidth: float | None = None,
    alpha: float | None = None,
) -> Line2D:
    style = {}
    if edgecolor is not None:
        style["markeredgecolor"] = edgecolor
    if edgewidth is not None:
        style["markeredgewidth"] = edgewidth
    if alpha is not None:
        style["alpha"] = alpha
    return Line2D(
        [0],
        [0],
        marker=marker,
        color="w",
        markerfacecolor=facecolor,
        markersize=5,
        label=label,
        **style,
    )


def _plot_columns(axis, frame, specifications, **style) -> None:
    for column, label, color in specifications:
        axis.plot(
            frame.index,
            frame[column],
            label=label,
            color=color,
            **style,
        )


def _fig_stablecoin_peg(data: AnalysisData, figures_dir: Path) -> None:
    period_start = pd.Timestamp("2023-03-09", tz="UTC")
    period_end = pd.Timestamp("2023-03-16", tz="UTC")
    prices = data.prices.loc[
        (data.prices.index >= period_start) & (data.prices.index <= period_end)
    ]
    basis = data.basis.loc[
        (data.basis.index >= period_start) & (data.basis.index <= period_end)
    ]
    with plt.rc_context(COL_RC):
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 4.0), sharex=True)
        for axis in (top, bottom):
            _shade_crisis(axis)
            axis.set_xlim(period_start, period_end)
        _plot_columns(
            top,
            prices,
            [
                ("kraken_usdcusd", "USDC/USD (Kraken)", CB_GREEN),
                ("kraken_usdtusd", "USDT/USD (Kraken)", CB_BLUE),
                ("coinbase_usdtusd", "USDT/USD (Coinbase)", CB_RED),
            ],
            linewidth=0.65,
            alpha=0.9,
        )
        top.axhline(1.0, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
        top.set_ylabel("Price (USD)")
        top.set_ylim(0.85, 1.04)
        top.set_title("(A) Stablecoin Spot Prices")
        top.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=6, ncol=1)
        bottom.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        _plot_columns(
            bottom,
            basis,
            [
                ("usdc_peg_dev_kraken", "USDC Dev (Kraken)", CB_GREEN),
                ("usdt_peg_dev_kraken", "USDT Dev (Kraken)", CB_BLUE),
                ("usdt_peg_dev_coinbase", "USDT Dev (Coinbase)", CB_RED),
            ],
            linewidth=0.65,
            alpha=0.9,
        )
        bottom.set_ylabel("Deviation (bps)")
        bottom.set_title("(B) Peg Deviations from $1.00")
        bottom.legend(
            loc="lower right", frameon=True, framealpha=0.9, fontsize=6, ncol=1
        )
        _format_date(bottom)
        figure.tight_layout(pad=0.5, h_pad=0.7)
        _save(figure, "fig_stablecoin_peg.png", figures_dir)


def _fig_dispersion_vs_adjusted(data: AnalysisData, figures_dir: Path) -> None:
    period_start = pd.Timestamp("2023-03-10", tz="UTC")
    period_end = pd.Timestamp("2023-03-13 23:59", tz="UTC")
    basis = data.basis.loc[
        (data.basis.index >= period_start) & (data.basis.index <= period_end)
    ]
    with plt.rc_context(COL_RC):
        figure, axes = plt.subplots(2, 1, figsize=(3.4, 4.5), sharex=True)
        for axis in axes:
            axis.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
            _shade_crisis(axis)
        for axis, dispersion, adjusted, title in [
            (
                axes[0],
                "dispersion_usdc_kraken",
                "basis_usdc_kraken",
                "(A) USDC Channel (Kraken)",
            ),
            (
                axes[1],
                "dispersion_usdt_kraken",
                "basis_usdt_kraken",
                "(B) USDT Channel (Kraken)",
            ),
        ]:
            axis.plot(
                basis.index,
                basis[dispersion],
                color=CB_ORANGE,
                linewidth=0.65,
                alpha=0.9,
                label="$D_t$ (unadj.)",
            )
            axis.plot(
                basis.index,
                basis[adjusted],
                color=CB_BLUE,
                linewidth=0.65,
                alpha=0.9,
                label="$B_t$ (adj.)",
            )
            axis.set_ylabel("bps")
            axis.set_title(title)
            axis.legend(
                loc="upper right", ncol=2, frameon=True, framealpha=0.9, fontsize=6
            )
        _format_date(axes[1], date_format="%b %d\n%H:%M", rotation=0)
        figure.tight_layout(pad=0.5, h_pad=0.7)
        _save(figure, "fig_dispersion_vs_adjusted_kraken.png", figures_dir)


def _fig_substitution_scatter(data: AnalysisData, figures_dir: Path) -> None:
    pre_mask = (data.basis.index >= pd.Timestamp("2023-03-01", tz="UTC")) & (
        data.basis.index < CRISIS_START
    )
    crisis_mask = (data.basis.index >= CRISIS_START) & (data.basis.index < CRISIS_END)
    scatter = (
        data.basis.loc[
            pre_mask | crisis_mask, ["dispersion_usdc_kraken", "dispersion_usdt_kraken"]
        ]
        .copy()
        .dropna()
    )
    scatter.columns = ["D_USDC", "D_USDT"]
    scatter["Regime"] = np.where(
        (scatter.index >= CRISIS_START) & (scatter.index < CRISIS_END),
        "Crisis",
        "Pre-SVB",
    )
    pre_correlation = (
        scatter.loc[scatter["Regime"] == "Pre-SVB", ["D_USDC", "D_USDT"]]
        .corr()
        .iloc[0, 1]
    )
    crisis_correlation = (
        scatter.loc[scatter["Regime"] == "Crisis", ["D_USDC", "D_USDT"]]
        .corr()
        .iloc[0, 1]
    )
    x_clip = scatter.loc[scatter["Regime"] == "Crisis", "D_USDC"].quantile(0.97)
    y_low = scatter.loc[scatter["Regime"] == "Crisis", "D_USDT"].quantile(0.01)
    plotted = scatter.loc[
        (scatter["D_USDC"] >= -30)
        & (scatter["D_USDC"] <= x_clip)
        & (scatter["D_USDT"] >= y_low)
        & (scatter["D_USDT"] <= 35)
    ]
    with plt.rc_context(COL_RC):
        figure, axis = plt.subplots(figsize=(3.4, 3.8))
        pre = plotted.loc[plotted["Regime"] == "Pre-SVB"]
        crisis = plotted.loc[plotted["Regime"] == "Crisis"]
        axis.scatter(
            pre["D_USDC"],
            pre["D_USDT"],
            color=CB_GRAY,
            s=4,
            alpha=0.55,
            linewidths=0,
            zorder=2,
        )
        axis.scatter(
            crisis["D_USDC"],
            crisis["D_USDT"],
            color=CB_RED,
            s=4,
            alpha=0.5,
            linewidths=0,
            zorder=3,
        )
        for regime, color in [("Pre-SVB", "#555555"), ("Crisis", "#AA0000")]:
            sample = plotted.loc[plotted["Regime"] == regime]
            axis.scatter(
                sample["D_USDC"].mean(),
                sample["D_USDT"].mean(),
                color=color,
                s=50,
                marker="D",
                zorder=5,
                edgecolors="white",
                linewidths=0.7,
            )
        axis.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.35)
        axis.axvline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.35)
        annotation = f"$\\rho$ (unadj. $D_t$):\nPre-SVB: {pre_correlation:+.2f}\nCrisis:   {crisis_correlation:+.2f}\n\n$\\rho$ (adj. $B_t$):\nPre-SVB: $+$0.44\nCrisis:   $+$0.16"
        axis.text(
            0.97,
            0.97,
            annotation,
            transform=axis.transAxes,
            fontsize=6,
            va="top",
            ha="right",
            linespacing=1.4,
            bbox=_annotation_box(
                "white",
                "#CCCCCC",
                pad=0.35,
                linewidth=0.6,
                alpha=0.95,
            ),
        )
        legend_elements = [
            _legend_marker(
                "o",
                CB_GRAY,
                f"Pre-SVB ($n={len(pre):,}$)",
                alpha=0.8,
            ),
            _legend_marker(
                "o",
                CB_RED,
                f"Crisis ($n={len(crisis):,}$)",
                alpha=0.8,
            ),
            _legend_marker(
                "D",
                "#555555",
                "Pre-SVB centroid",
                edgecolor="white",
                edgewidth=0.5,
            ),
            _legend_marker(
                "D",
                "#AA0000",
                "Crisis centroid",
                edgecolor="white",
                edgewidth=0.5,
            ),
        ]
        axis.legend(
            handles=legend_elements,
            loc="upper left",
            frameon=True,
            framealpha=0.9,
            edgecolor="#CCCCCC",
            fontsize=6,
        )
        axis.set_xlabel("$D_{USDC,t}$ (bps)")
        axis.set_ylabel("$D_{USDT,t}$ (bps)")
        axis.set_title("Stablecoin Substitution ($D_t$, Kraken)")
        axis.yaxis.grid(True, linewidth=0.3, color="#EEEEEE")
        axis.set_axisbelow(True)
        figure.tight_layout(pad=0.4)
        _save(figure, "fig_stablecoin_substitution_scatter.png", figures_dir)


def _fig_half_life_robustness(data: AnalysisData, figures_dir: Path) -> None:
    table = metrics.half_life_robustness(data)
    table["spec"] = table["freq"] + " " + table["ff_filter"]
    table["stablecoin"] = table["series"].str.extract("(USDC|USDT)")[0]
    regimes = ["Pre-SVB", "Crisis", "Post-SVB"]
    specifications = ["1m all", "1m no_ff", "5m all", "5m no_ff"]
    colors = {"USDC": CB_BLUE, "USDT": CB_RED}
    markers = {"USDC": "o", "USDT": "s"}
    with plt.rc_context(COL_RC):
        figure, axes = plt.subplots(1, 3, figsize=(3.4, 2.6), sharey=True)
        for axis, regime in zip(axes, regimes, strict=True):
            regime_data = table[table["regime"] == regime]
            for coin_index, stablecoin in enumerate(["USDC", "USDT"]):
                coin_data = regime_data[regime_data["stablecoin"] == stablecoin]
                for specification_index, specification in enumerate(specifications):
                    row = coin_data[coin_data["spec"] == specification]
                    x_value = specification_index + (coin_index - 0.5) * 0.2
                    if len(row) == 0 or pd.isna(row["half_life_min"].values[0]):
                        axis.scatter(
                            x_value,
                            0,
                            marker="x",
                            color=colors[stablecoin],
                            s=20,
                            zorder=5,
                        )
                        axis.annotate(
                            "NaN",
                            (x_value, 0.1),
                            fontsize=5,
                            ha="center",
                            color=colors[stablecoin],
                        )
                    else:
                        value = row["half_life_min"].values[0]
                        axis.scatter(
                            x_value,
                            value,
                            marker=markers[stablecoin],
                            color=colors[stablecoin],
                            s=22,
                            zorder=5,
                            edgecolors="k",
                            linewidths=0.3,
                        )
                        axis.annotate(
                            f"{value:.1f}",
                            (x_value, value + 0.09),
                            fontsize=5,
                            ha="center",
                            color=colors[stablecoin],
                        )
            axis.set_title(regime, fontsize=7, fontweight="bold")
            axis.set_xticks(range(len(specifications)))
            axis.set_xticklabels(specifications, fontsize=5.5, rotation=35, ha="right")
            axis.set_xlim(-0.5, len(specifications) - 0.5)
            axis.grid(axis="y", alpha=0.3, linewidth=0.3)
        axes[0].set_ylabel("Half-Life (min)", fontsize=7)
        legend_elements = [
            _legend_marker("o", CB_BLUE, "USDC $B_t$", edgecolor="k"),
            _legend_marker("s", CB_RED, "USDT $B_t$", edgecolor="k"),
        ]
        figure.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=2,
            fontsize=6,
            frameon=False,
            bbox_to_anchor=(0.5, -0.06),
        )
        figure.suptitle(
            "Half-Life Robustness: Adjusted Residual $B_t$",
            fontsize=8,
            fontweight="bold",
            y=1.02,
        )
        figure.tight_layout(pad=0.4, w_pad=0.3)
        _save(figure, "fig_half_life_robustness.png", figures_dir)


def _fig_two_layer_persistence(data: AnalysisData, figures_dir: Path) -> None:
    period_start = pd.Timestamp("2023-03-10", tz="UTC")
    period_end = pd.Timestamp("2023-03-13", tz="UTC")
    basis = data.basis.loc[
        (data.basis.index >= period_start) & (data.basis.index < period_end)
    ]
    with plt.rc_context(COL_RC):
        figure, left_axis = plt.subplots(figsize=(3.4, 3.0))
        (basis_line,) = left_axis.plot(
            basis.index,
            basis["basis_usdc_kraken"],
            color=CB_BLUE,
            linewidth=0.55,
            alpha=0.9,
            zorder=3,
            label="$B_{USDC,t}$ (Kraken, left)",
        )
        left_axis.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.35)
        left_axis.set_ylabel("$B_{USDC,t}$ (bps)", color=CB_BLUE, labelpad=3)
        left_axis.tick_params(axis="y", labelcolor=CB_BLUE)
        left_axis.set_ylim(-130, 210)
        left_axis.yaxis.set_major_locator(mticker.MultipleLocator(50))
        right_axis = left_axis.twinx()
        right_axis.spines["right"].set_visible(True)
        right_axis.spines["right"].set_linewidth(0.6)
        (peg_line,) = right_axis.plot(
            basis.index,
            basis["usdc_peg_dev_kraken"],
            color=CB_RED,
            linewidth=0.9,
            alpha=0.92,
            zorder=2,
            label="Peg Dev. (right)",
        )
        right_axis.set_ylabel("USDC Peg Dev. (bps)", color=CB_RED, labelpad=3)
        right_axis.tick_params(axis="y", labelcolor=CB_RED)
        right_axis.set_ylim(-1450, 230)
        right_axis.yaxis.set_major_locator(mticker.MultipleLocator(250))
        left_axis.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        left_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        plt.setp(left_axis.xaxis.get_majorticklabels(), ha="center", fontsize=6)
        left_axis.set_xlim(period_start, period_end)
        left_axis.set_xlabel("Date / Time (UTC)", labelpad=3)
        lines = [basis_line, peg_line]
        labels = [line.get_label() for line in lines]
        left_axis.legend(
            lines,
            labels,
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            edgecolor="#CCCCCC",
            fontsize=6,
        )
        left_axis.text(
            0.995,
            0.97,
            "HL gap: $\\times$940\n$B_{USDC,t}$: $\\approx$0.6 min\nPeg dev: $\\approx$572 min",
            transform=left_axis.transAxes,
            fontsize=6,
            va="top",
            ha="right",
            bbox=_annotation_box("#F7F7F7", "#CCCCCC", linewidth=0.6, alpha=0.95),
        )
        left_axis.set_title(
            "Two-Layer Persistence: $B_{USDC,t}$ vs.\\ Peg Dev.\n(Kraken, 10--13 Mar 2023)",
            pad=4,
        )
        figure.tight_layout(pad=0.5)
        _save(figure, "fig_two_layer_persistence.png", figures_dir)


def _fig_cross_exchange_basis(data: AnalysisData, figures_dir: Path) -> None:
    period_start = pd.Timestamp("2023-03-09", tz="UTC")
    period_end = pd.Timestamp("2023-03-21 23:59", tz="UTC")
    basis = data.basis.loc[
        (data.basis.index >= period_start) & (data.basis.index <= period_end)
    ]
    specifications = [
        ("xbasis_btcusdt_binance_kraken", "(A) BTC/USDT Binance−Kraken", CB_ORANGE),
        ("xbasis_btcusdt_coinbase_kraken", "(B) BTC/USDT Coinbase−Kraken", CB_CYAN),
        ("xbasis_btcusd_coinbase_kraken", "(C) BTC/USD Coinbase−Kraken", CB_PURPLE),
    ]
    with plt.rc_context(COL_RC):
        figure, axes = plt.subplots(3, 1, figsize=(3.4, 5.5), sharex=True)
        for axis, (column, title, color) in zip(axes, specifications, strict=True):
            _shade_crisis(axis)
            axis.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.4)
            axis.set_xlim(period_start, period_end)
            axis.plot(
                basis.index, basis[column], linewidth=0.5, color=color, alpha=0.85
            )
            axis.set_title(title, fontsize=7.5)
            axis.set_ylabel("Basis (bps)")
        _format_date(axes[2])
        figure.tight_layout(pad=0.5, h_pad=0.6)
        _save(figure, "fig_cross_exchange_basis.png", figures_dir)


def _grouped_liquidity_bars(
    axis: plt.Axes, frame: pd.DataFrame, ylabel: str, title: str
) -> None:
    regime_order = ["Pre-SVB", "Crisis", "Post-SVB"]
    pair_order = [
        "Kraken BTC/USD",
        "Kraken BTC/USDT",
        "Kraken BTC/USDC",
        "Binance BTC/USDT",
        "Coinbase BTC/USD",
    ]
    short_names = ["KR/USD", "KR/USDT", "KR/USDC", "BN/USDT", "CB/USD"]
    pivot = frame.pivot(index="Pair", columns="Regime", values="mean")[
        regime_order
    ].reindex(pair_order)
    positions = np.arange(len(pivot))
    width = 0.22
    for regime_index, regime in enumerate(regime_order):
        axis.bar(
            positions + (regime_index - 1) * width,
            pivot[regime],
            width,
            label=regime,
            color=REGIME_COLORS[regime],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(short_names, fontsize=6)
    axis.set_ylabel(ylabel, fontsize=7)
    axis.set_title(title, fontsize=7.5)
    axis.legend(fontsize=6, ncol=3)
    axis.grid(axis="y", linewidth=0.3, alpha=0.5)
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)


def _fig_liquidity_roll_amihud(data: AnalysisData, figures_dir: Path) -> None:
    pairs = {
        "Kraken BTC/USD": "kraken_btcusd",
        "Kraken BTC/USDT": "kraken_btcusdt",
        "Kraken BTC/USDC": "kraken_btcusdc",
        "Binance BTC/USDT": "binance_btcusdt",
        "Coinbase BTC/USD": "coinbase_btcusd",
    }
    roll_series = {
        label: metrics.daily_roll_spread(data.prices[column])
        for label, column in pairs.items()
    }
    amihud_series = {
        label: metrics.daily_amihud_illiquidity(
            data.prices[column], data.volumes[column]
        )
        for label, column in pairs.items()
    }
    pair_order = tuple(pairs)
    roll_means = metrics.daily_regime_means(roll_series, pair_order)
    amihud_means = metrics.daily_regime_means(amihud_series, pair_order)
    with plt.rc_context(COL_RC):
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 4.5))
        _grouped_liquidity_bars(
            top, roll_means, "Roll Spread (bps)", "(A) Roll (1984) Effective Spread"
        )
        _grouped_liquidity_bars(
            bottom,
            amihud_means,
            "Amihud ILLIQ ($\\times10^{-6}$)",
            "(B) Amihud (2002) ILLIQ Ratio",
        )
        figure.tight_layout(pad=0.5, h_pad=0.8)
        _save(figure, "fig_liquidity_roll_amihud.png", figures_dir)


def _fig_volume_share(data: AnalysisData, figures_dir: Path) -> None:
    volume_columns = [
        "binance_btcusdt",
        "binance_btcusdc",
        "coinbase_btcusd",
        "coinbase_btcusdt",
        "kraken_btcusd",
        "kraken_btcusdt",
        "kraken_btcusdc",
    ]
    volume_percentages = metrics.daily_volume_shares(
        data.volumes, tuple(volume_columns)
    )
    labels = {
        "binance_btcusdt": "Binance USDT",
        "binance_btcusdc": "Binance USDC",
        "coinbase_btcusd": "Coinbase USD",
        "coinbase_btcusdt": "Coinbase USDT",
        "kraken_btcusd": "Kraken USD",
        "kraken_btcusdt": "Kraken USDT",
        "kraken_btcusdc": "Kraken USDC",
    }
    colors = [CB_BLUE, CB_CYAN, CB_ORANGE, CB_PURPLE, "#444444", CB_GREEN, CB_RED]
    with plt.rc_context(COL_RC):
        figure, axis = plt.subplots(figsize=(3.4, 3.0))
        axis.stackplot(
            volume_percentages.index,
            *[volume_percentages[column] for column in volume_columns],
            labels=[labels[column] for column in volume_columns],
            colors=colors,
            alpha=0.82,
        )
        axis.axvspan(
            CRISIS_START.normalize(),
            CRISIS_END.normalize(),
            alpha=0.25,
            color="red",
            label="SVB Crisis",
            zorder=10,
        )
        axis.set_title("Daily Volume Fragmentation")
        axis.set_ylabel("Volume Share (%)")
        _format_date(axis)
        axis.legend(
            loc="upper left", fontsize=5.5, frameon=True, framealpha=0.9, ncol=2
        )
        figure.tight_layout(pad=0.4)
        _save(figure, "fig_volume_share.png", figures_dir)


def _fig_realized_volatility(data: AnalysisData, figures_dir: Path) -> None:
    columns = [
        "kraken_btcusd",
        "kraken_btcusdt",
        "kraken_btcusdc",
        "binance_btcusdt",
        "coinbase_btcusd",
    ]
    realized_volatility = metrics.realized_volatility(data.prices, tuple(columns))
    labels = {
        "kraken_btcusd": "Kraken USD",
        "kraken_btcusdt": "Kraken USDT",
        "kraken_btcusdc": "Kraken USDC",
        "binance_btcusdt": "Binance USDT",
        "coinbase_btcusd": "Coinbase USD",
    }
    colors = ["#2c3e50", CB_BLUE, CB_GREEN, CB_ORANGE, CB_RED]
    with plt.rc_context(COL_RC):
        figure, axis = plt.subplots(figsize=(3.4, 3.0))
        _shade_crisis(axis)
        for column, color in zip(columns, colors, strict=True):
            axis.plot(
                realized_volatility.index,
                realized_volatility[column],
                linewidth=0.5,
                color=color,
                label=labels[column],
                alpha=0.85,
            )
        axis.set_title("Hourly Realized Volatility (60-min, $\\times\\sqrt{60}$)")
        axis.set_ylabel("Volatility (bps/hr)")
        _format_date(axis)
        axis.legend(loc="upper right", fontsize=6, ncol=1, frameon=True)
        figure.tight_layout(pad=0.4)
        _save(figure, "fig_realized_volatility.png", figures_dir)


def _fig_tail_blowout_kde(data: AnalysisData, figures_dir: Path) -> None:
    pre_mask = (data.basis.index >= pd.Timestamp("2023-03-01", tz="UTC")) & (
        data.basis.index < CRISIS_START
    )
    crisis_mask = (data.basis.index >= CRISIS_START) & (data.basis.index < CRISIS_END)
    pre_basis = data.basis.loc[pre_mask, "basis_usdc_kraken"].dropna()
    crisis_basis = data.basis.loc[crisis_mask, "basis_usdc_kraken"].dropna()
    x_low, x_high = (-50, 100)
    with plt.rc_context(COL_RC):
        figure, axis = plt.subplots(figsize=(3.4, 3.2))
        for series, color, alpha, label in [
            (pre_basis, CB_GRAY, 0.4, f"Pre-SVB ($n={len(pre_basis):,}$)"),
            (crisis_basis, CB_RED, 0.35, f"Crisis ($n={len(crisis_basis):,}$)"),
        ]:
            sns.kdeplot(
                series,
                ax=axis,
                color=color,
                linewidth=1.4,
                fill=True,
                alpha=alpha,
                label=label,
                clip=(x_low, x_high),
                bw_adjust=0.9,
            )
        axis.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.45)
        for series, color, label in [
            (pre_basis, CB_GRAY, "Pre-SVB"),
            (crisis_basis, CB_RED, "Crisis"),
        ]:
            axis.axvline(
                series.mean(),
                color=color,
                linewidth=1.0,
                linestyle=":",
                label=f"{label} mean ({series.mean():.1f} bps)",
            )
        pre_p99 = pre_basis.quantile(0.99)
        crisis_p99 = crisis_basis.quantile(0.99)
        axis.text(
            0.97,
            0.97,
            f"P99 expansion\nPre-SVB: {pre_p99:.1f} bps\nCrisis: {crisis_p99:.1f} bps\nRatio: ×6",
            transform=axis.transAxes,
            fontsize=6,
            va="top",
            ha="right",
            linespacing=1.4,
            bbox=_annotation_box("#FFF5F5", CB_RED),
        )
        axis.text(
            0.03,
            0.97,
            f"Ex. kurtosis\nPre-SVB: {pre_basis.kurt():.1f}\nCrisis: {crisis_basis.kurt():.1f}",
            transform=axis.transAxes,
            fontsize=6,
            va="top",
            ha="left",
            linespacing=1.4,
            bbox=_annotation_box("#F5F5FF", CB_BLUE),
        )
        axis.set_xlim(x_low, x_high)
        axis.set_xlabel("$B_{USDC,t}$ (bps)")
        axis.set_ylabel("Density")
        axis.yaxis.grid(True, linewidth=0.3, color="#EEEEEE")
        axis.set_axisbelow(True)
        axis.legend(
            loc="upper center",
            frameon=True,
            framealpha=0.9,
            edgecolor="#CCCCCC",
            fontsize=6,
        )
        axis.set_title("Tail Blowout: $B_{USDC,t}$ (Kraken)", pad=4)
        figure.tight_layout(pad=0.4)
        _save(figure, "fig_tail_blowout_kde.png", figures_dir)


def _draw_correlation_heatmap(
    axis: plt.Axes, basis: pd.DataFrame, mask: pd.Series | np.ndarray, regime_name: str
) -> matplotlib.image.AxesImage:
    columns = [
        "basis_usdc_kraken",
        "basis_usdt_kraken",
        "basis_usdt_coinbase",
        "xbasis_btcusdt_binance_kraken",
        "xbasis_btcusd_coinbase_kraken",
    ]
    labels = [
        "$B_{USDC}$ KR",
        "$B_{USDT}$ KR",
        "$B_{USDT}$ CB",
        "X-USDT BN-KR",
        "X-USD CB-KR",
    ]
    sample = basis.loc[mask, columns].dropna()
    correlation = sample.corr()
    image = axis.imshow(
        correlation.values, vmin=-0.2, vmax=1.0, cmap="RdBu_r", aspect="equal"
    )
    for row in range(len(columns)):
        for column in range(len(columns)):
            value = correlation.values[row, column]
            color = "white" if abs(value) > 0.55 else "black"
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=color,
                fontweight="bold",
            )
    axis.set_xticks(range(len(columns)))
    axis.set_yticks(range(len(columns)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=5.5)
    axis.set_yticklabels(labels, fontsize=5.5)
    axis.set_title(f"{regime_name}  ($n={len(sample):,}$)", fontsize=7, pad=3)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)
    return image


def _fig_correlation_regime_heatmap(data: AnalysisData, figures_dir: Path) -> None:
    pre_mask = (data.basis.index >= pd.Timestamp("2023-03-01", tz="UTC")) & (
        data.basis.index < CRISIS_START
    )
    crisis_mask = (data.basis.index >= CRISIS_START) & (data.basis.index < CRISIS_END)
    post_mask = (data.basis.index >= pd.Timestamp("2023-03-13", tz="UTC")) & (
        data.basis.index < pd.Timestamp("2023-03-22", tz="UTC")
    )
    with plt.rc_context(COL_RC):
        figure, axes = plt.subplots(
            2, 2, figsize=(6.8, 6.0), gridspec_kw={"hspace": 0.65, "wspace": 0.45}
        )
        pre_axis = axes[0, 0]
        crisis_axis = axes[0, 1]
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)
        grid = axes[1, 0].get_gridspec()
        post_axis = figure.add_subplot(grid[1, 0:2])
        pre_position = axes[0, 0].get_position()
        crisis_position = axes[0, 1].get_position()
        panel_width = pre_position.width
        panel_height = pre_position.height
        center_x = (pre_position.x0 + crisis_position.x1) / 2.0
        bottom_position = axes[1, 0].get_position()
        post_axis.set_position(
            [
                center_x - panel_width / 2.0,
                bottom_position.y0,
                panel_width,
                panel_height,
            ]
        )
        image = _draw_correlation_heatmap(pre_axis, data.basis, pre_mask, "Pre-SVB")
        _draw_correlation_heatmap(crisis_axis, data.basis, crisis_mask, "Crisis")
        _draw_correlation_heatmap(post_axis, data.basis, post_mask, "Post-SVB")
        for axis in (pre_axis, crisis_axis, post_axis):
            axis.set_box_aspect(1)
        colorbar_axis = figure.add_axes(
            [
                crisis_position.x1 + 0.02,
                crisis_position.y0,
                0.02,
                crisis_position.height,
            ]
        )
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Pearson Corr.", fontsize=6.5)
        colorbar.ax.tick_params(labelsize=6)
        figure.suptitle("Cross-Channel Correlation by Regime", fontsize=8, y=0.99)
        _save(figure, "fig_correlation_regime_heatmap.png", figures_dir)


def _fig_arbitrage_after_fees(data: AnalysisData, figures_dir: Path) -> None:
    fee_bps = 5.0
    specifications = [
        (
            "basis_usdc_kraken",
            "USDC/USD KR (3-leg)",
            3,
            ("kraken_btcusdc", "kraken_usdcusd", "kraken_btcusd"),
        ),
        (
            "basis_usdt_kraken",
            "USDT/USD KR (3-leg)",
            3,
            ("kraken_btcusdt", "kraken_usdtusd", "kraken_btcusd"),
        ),
        (
            "xbasis_btcusdt_binance_kraken",
            "USDT BN-KR (2-leg)",
            2,
            ("binance_btcusdt", "kraken_btcusdt"),
        ),
        (
            "xbasis_btcusd_coinbase_kraken",
            "USD CB-KR (2-leg)",
            2,
            ("coinbase_btcusd", "kraken_btcusd"),
        ),
    ]
    colors = [CB_ORANGE, CB_BLUE, CB_RED, "#2c3e50"]
    with plt.rc_context(COL_RC):
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 4.5), sharex=True)
        for axis in (top, bottom):
            _shade_crisis(axis)
        for (basis_column, label, n_legs, range_columns), color in zip(
            specifications, colors, strict=True
        ):
            channel = metrics.arbitrage_after_costs(
                data.basis[basis_column],
                data.ranges,
                range_columns,
                n_legs,
                fee_bps=fee_bps,
            )
            top.plot(
                channel.index,
                channel["net_fee"],
                linewidth=0.45,
                color=color,
                label=label,
                alpha=0.85,
            )
            bottom.plot(
                channel.index,
                channel["net_fee_slippage"],
                linewidth=0.45,
                color=color,
                label=label,
                alpha=0.85,
            )
        top.set_title(f"(A) Fee-Only Net Arb. ({fee_bps:.0f} bps/leg)")
        top.set_ylabel("Net Profit (bps)")
        top.legend(loc="upper left", fontsize=6, ncol=1, frameon=True)
        bottom.set_title("(B) Fee + Slippage Net Arb.")
        bottom.set_ylabel("Net Profit (bps)")
        _format_date(bottom)
        figure.tight_layout(pad=0.5, h_pad=0.7)
        _save(figure, "fig_arbitrage_after_fees.png", figures_dir)


def _fig_svb_crisis_zoom(data: AnalysisData, figures_dir: Path) -> None:
    crisis_basis = data.basis.loc[
        (data.basis.index >= CRISIS_START) & (data.basis.index < CRISIS_END)
    ]
    with plt.rc_context(COL_RC):
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 4.5), sharex=True)
        top.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.35)
        _plot_columns(
            top,
            crisis_basis,
            [
                ("basis_usdc_kraken", "$B_{USDC,t}$ (Kraken)", CB_GREEN),
                ("basis_usdt_kraken", "$B_{USDT,t}$ (Kraken)", CB_BLUE),
                ("basis_usdt_coinbase", "$B_{USDT,t}$ (Coinbase)", CB_RED),
            ],
            linewidth=0.45,
            alpha=0.88,
        )
        top.set_ylabel("Adjusted Residual $B_t$ (bps)")
        top.set_title("(A) Intra-Exchange Adjusted Residuals")
        top.legend(
            loc="upper right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC"
        )
        bottom.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.35)
        _plot_columns(
            bottom,
            crisis_basis,
            [
                (
                    "xbasis_btcusdt_binance_kraken",
                    "Binance$-$Kraken BTC/USDT",
                    CB_ORANGE,
                ),
                (
                    "xbasis_btcusd_coinbase_kraken",
                    "Coinbase$-$Kraken BTC/USD",
                    CB_DARK,
                ),
            ],
            linewidth=0.45,
            alpha=0.88,
        )
        bottom.set_ylabel("Cross-Exchange Basis (bps)")
        bottom.set_title("(B) Cross-Exchange Basis")
        bottom.legend(
            loc="upper right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC"
        )
        bottom.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        plt.setp(bottom.xaxis.get_majorticklabels(), ha="center")
        bottom.set_xlim(CRISIS_START, CRISIS_END)
        figure.tight_layout(pad=0.5, h_pad=0.8)
        _save(figure, "fig_svb_crisis_zoom.png", figures_dir)


def _fig_var_irf(data: AnalysisData, figures_dir: Path) -> None:
    result = metrics.var_irf(data)
    values = result["values"]
    lower = result["lower"]
    upper = result["upper"]
    horizons = result["horizons"]
    names = ["BTC/USD", "BTC/USDC"]
    panel_labels = [["(A)", "(B)"], ["(C)", "(D)"]]
    with plt.rc_context(COL_RC):
        figure, axes = plt.subplots(2, 2, figsize=(3.4, 3.4), sharex=True)
        for response_index in range(2):
            for shock_index in range(2):
                axis = axes[response_index, shock_index]
                response = values[:, response_index, shock_index]
                axis.fill_between(
                    horizons,
                    lower[:, response_index, shock_index],
                    upper[:, response_index, shock_index],
                    color=CB_BLUE,
                    alpha=0.18,
                    linewidth=0,
                )
                axis.plot(horizons, response, color=CB_BLUE, linewidth=0.9, zorder=3)
                axis.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
                axis.set_title(
                    f"{panel_labels[response_index][shock_index]}  {names[shock_index]} $\\rightarrow$ {names[response_index]}"
                )
                if response_index == 1:
                    axis.set_xlabel("Lag (minutes)")
                if shock_index == 0:
                    axis.set_ylabel("Response (bps)")
                axis.set_xlim(0, 10)
                axis.xaxis.set_major_locator(mticker.MultipleLocator(2))
        figure.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.0)
        _save(figure, "fig_var_irf.png", figures_dir)


def write_figures(data: AnalysisData, paths: RepoPaths) -> None:
    """Render the exact frozen figure set under ``paths.figures``."""
    paths.figures.mkdir(parents=True, exist_ok=True)
    for existing in paths.figures.iterdir():
        if existing.is_file():
            existing.unlink()
    _fig_stablecoin_peg(data, paths.figures)
    _fig_dispersion_vs_adjusted(data, paths.figures)
    _fig_substitution_scatter(data, paths.figures)
    _fig_half_life_robustness(data, paths.figures)
    _fig_two_layer_persistence(data, paths.figures)
    _fig_cross_exchange_basis(data, paths.figures)
    _fig_liquidity_roll_amihud(data, paths.figures)
    _fig_volume_share(data, paths.figures)
    _fig_realized_volatility(data, paths.figures)
    _fig_tail_blowout_kde(data, paths.figures)
    _fig_correlation_regime_heatmap(data, paths.figures)
    _fig_arbitrage_after_fees(data, paths.figures)
    _fig_svb_crisis_zoom(data, paths.figures)
    _fig_var_irf(data, paths.figures)
    observed = {path.name for path in paths.figures.iterdir() if path.is_file()}
    if observed != EXPECTED_FIGURES:
        missing = sorted(EXPECTED_FIGURES - observed)
        unexpected = sorted(observed - EXPECTED_FIGURES)
        raise ValueError(
            f"figure output mismatch; missing={missing}, unexpected={unexpected}"
        )
