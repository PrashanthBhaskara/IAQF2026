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
        "IAQF_Inefficient_Markets_2026.tex": (
            "8f645f8d7d97e0ef0042f4a1d10e206fb9c76be967224a9194e340b778973335"
        ),
        "IAQF_Inefficient_Markets_2026.pdf": (
            "b583b7051b00604af3da42559c287865335ea51b7afb2b590d60ff37f454025b"
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

EXPECTED_FIGURE_SHA256 = MappingProxyType(
    {
        "fig_arbitrage_after_fees.png": "4f87c06e684a718a766d885659dc35f55de519517b4d3c1ec0669b39f77eb10f",
        "fig_correlation_regime_heatmap.png": "4a3f1fc52ded4ba935a9d122f28e501e0cc7eb769991c219005547a404d5b10f",
        "fig_cross_exchange_basis.png": "a25cc496713dfa37e9c0bf5619294c74484add017bc6d0e0da501749198c7ee3",
        "fig_dispersion_vs_adjusted_kraken.png": "818fea10a6deb17802b7cd916c8f98fd5ff3575be051adddacdb83bb2e7fe7a6",
        "fig_half_life_robustness.png": "d391d170e82a54a073fa87a0c3aa4b96927042f8b96794f47b27f63f37e158a7",
        "fig_liquidity_roll_amihud.png": "1d8da0f9c57ce8aace03833232417e718dc536df79f62f55368fde598f148c24",
        "fig_realized_volatility.png": "0cef020b13c2528f2925ea82ca40d88f8fe90e16d9589c4e681ac01fe36f2196",
        "fig_stablecoin_peg.png": "8c3457dca2ee36121aeb391bb41282e01391fd17e255315c942f605e94e77fe3",
        "fig_stablecoin_substitution_scatter.png": "289e4f91d4fa58746c1b0b49911d6a31040a007d8f0991e8d68375fdf9ff8810",
        "fig_svb_crisis_zoom.png": "b06ee3c7b581bb726e828d412552a4ed110270f65c97c8ab17509901dd8d06a2",
        "fig_tail_blowout_kde.png": "1712fbf73e080fa4dbc0e476754f829c4dfb0f3c7b0b284799c2c3870248f919",
        "fig_two_layer_persistence.png": "4fee442e257fc3a9cc3cf0618c7f455644e30afce12298de5b87a83171989b7d",
        "fig_var_irf.png": "36f7a14ec142cbb2eb08b231a41fa1eb39482b0900528cbe4d31139b003cbd9b",
        "fig_volume_share.png": "e2819db6bea28237f22d887a665f5ca92746b440c25e6b018195ea1c98c331b7",
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
        "arbitrage_fee_sensitivity.csv",
        "arbitrage_summary.csv",
        "cointegration_johansen.csv",
        "cointegration_vecm_merged.tex",
        "contagion_intensity.csv",
        "contagion_intensity.tex",
        "contagion_robustness.csv",
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
        "half_life_ratio_bootstrap.csv",
        "half_life_sanity_grid.csv",
        "hasbrouck_is.csv",
        "liquidity_spread_table.tex",
        "normality_diagnostics.csv",
        "ou_basis_stats.csv",
        "paper_claim_audit.csv",
        "peg_recovery_thresholds.csv",
        "price_discovery_metrics.csv",
        "price_discovery_diagnostics.csv",
        "realized_vol_headline.csv",
        "realized_vol_regime_means.csv",
        "regression_hac.tex",
        "regression_results.txt",
        "regression_usdc.txt",
        "regression_usdt.txt",
        "structural_break_tests.csv",
        "var_irf_diagnostics.csv",
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
            paper_tex=root / "IAQF_Inefficient_Markets_2026.tex",
            paper_pdf=root / "IAQF_Inefficient_Markets_2026.pdf",
        )
