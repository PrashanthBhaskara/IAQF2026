from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pandas as pd

STUDY_START = pd.Timestamp("2023-03-01 00:00:00", tz="UTC")
STUDY_END = pd.Timestamp("2023-03-21 23:59:00", tz="UTC")
CRISIS_START = pd.Timestamp("2023-03-10 00:00:00", tz="UTC")
CRISIS_END = pd.Timestamp("2023-03-13 00:00:00", tz="UTC")

PAPER_SHA256 = MappingProxyType(
    {
        "IAQF_column_Final.tex": (
            "de22c4fb09e64dbae89abfe340471d4eca68aa31746bd5ba7e47e28529f2e574"
        ),
        "IAQF_column_Final.pdf": (
            "7c84ae52903ff8004c2fa8a8e772e2f082722157a70f94172d6a5d5586337258"
        ),
    }
)

EXPECTED_FIGURES = frozenset(
    {
        "fig_arbitrage_after_fees.png",
        "fig_correlation_regime_heatmap.png",
        "fig_cross_exchange_basis.png",
        "fig_dispersion_vs_adjusted_kraken.png",
        "fig_half_life_robustness.png",
        "fig_liquidity_roll_amihud.png",
        "fig_realized_volatility.png",
        "fig_stablecoin_peg.png",
        "fig_stablecoin_substitution_scatter.png",
        "fig_svb_crisis_zoom.png",
        "fig_tail_blowout_kde.png",
        "fig_two_layer_persistence.png",
        "fig_var_irf.png",
        "fig_volume_share.png",
    }
)

EXPECTED_FIGURE_DIMENSIONS = MappingProxyType(
    {
        "fig_arbitrage_after_fees.png": (1047, 1381),
        "fig_correlation_regime_heatmap.png": (1949, 1787),
        "fig_cross_exchange_basis.png": (1047, 1679),
        "fig_dispersion_vs_adjusted_kraken.png": (1047, 1384),
        "fig_half_life_robustness.png": (1051, 889),
        "fig_liquidity_roll_amihud.png": (1038, 1377),
        "fig_realized_volatility.png": (1053, 934),
        "fig_stablecoin_peg.png": (1047, 1231),
        "fig_stablecoin_substitution_scatter.png": (1047, 1166),
        "fig_svb_crisis_zoom.png": (1041, 1384),
        "fig_tail_blowout_kde.png": (1054, 985),
        "fig_two_layer_persistence.png": (1038, 917),
        "fig_var_irf.png": (1064, 1046),
        "fig_volume_share.png": (1053, 937),
    }
)

EXPECTED_TABLES = frozenset(
    {
        "arbitrage_compact.tex",
        "arbitrage_summary.csv",
        "cointegration_johansen.csv",
        "cointegration_vecm_merged.tex",
        "contagion_intensity.csv",
        "contagion_intensity.tex",
        "data_coverage_core.csv",
        "data_coverage_core.tex",
        "depth_proxy_table.csv",
        "depth_proxy_table.tex",
        "dispersion_adjusted_identity_check.csv",
        "dispersion_adjusted_stats.csv",
        "dispersion_adjusted_stats.tex",
        "distributional_robustness.csv",
        "distributional_robustness.tex",
        "ff_sensitivity_core.csv",
        "genius_counterfactual.csv",
        "genius_counterfactual.tex",
        "granger_causality.csv",
        "granger_causality_fdr.csv",
        "hac_headline_metrics.csv",
        "half_life_robustness.csv",
        "half_life_sanity_grid.csv",
        "hasbrouck_is.csv",
        "liquidity_spread_table.tex",
        "ou_basis_stats.csv",
        "peg_recovery_thresholds.csv",
        "price_discovery_metrics.csv",
        "realized_vol_headline.csv",
        "realized_vol_regime_means.csv",
        "regression_hac.tex",
        "regression_results.txt",
        "regression_usdc.txt",
        "regression_usdt.txt",
        "volume_share_pair_regime.csv",
        "volume_share_quote_regime.csv",
    }
)


@dataclass(frozen=True)
class RepoPaths:
    root: Path
    raw: Path
    processed: Path
    tables: Path
    figures: Path
    paper_tex: Path
    paper_pdf: Path

    @classmethod
    def from_root(cls, root: Path) -> RepoPaths:
        root = Path(root)
        return cls(
            root=root,
            raw=root / "data_raw",
            processed=root / "data_processed",
            tables=root / "tables",
            figures=root / "figures_col",
            paper_tex=root / "IAQF_column_Final.tex",
            paper_pdf=root / "IAQF_column_Final.pdf",
        )
