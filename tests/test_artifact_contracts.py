from __future__ import annotations

import hashlib
import importlib
import re
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

PAPER_SHA256 = {
    "IAQF_column_Final.tex": "de22c4fb09e64dbae89abfe340471d4eca68aa31746bd5ba7e47e28529f2e574",
    "IAQF_column_Final.pdf": "7c84ae52903ff8004c2fa8a8e772e2f082722157a70f94172d6a5d5586337258",
}

EXPECTED_FIGURES = {
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

EXPECTED_TABLES = {
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


def validation_module():
    return importlib.import_module("iaqf.validation")


def repo_paths(root: Path):
    return importlib.import_module("iaqf.config").RepoPaths.from_root(root)


def copy_artifact_surface(destination: Path) -> None:
    shutil.copytree(ROOT / "figures_col", destination / "figures_col")
    shutil.copytree(ROOT / "tables", destination / "tables")
    for paper_name in PAPER_SHA256:
        shutil.copy2(ROOT / paper_name, destination / paper_name)


def test_frozen_tex_and_pdf_sha256_are_immutable() -> None:
    observed = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in PAPER_SHA256
    }

    assert observed == PAPER_SHA256


def test_final_tex_references_exactly_the_fourteen_owned_pngs() -> None:
    tex = (ROOT / "IAQF_column_Final.tex").read_text(encoding="utf-8")
    referenced = {
        Path(token).name
        for token in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    }
    owned = {path.name for path in (ROOT / "figures_col").iterdir() if path.is_file()}

    assert referenced == EXPECTED_FIGURES
    assert owned == EXPECTED_FIGURES


def test_owned_table_artifact_set_is_exact() -> None:
    owned = {path.name for path in (ROOT / "tables").iterdir() if path.is_file()}

    assert owned == EXPECTED_TABLES


def test_validators_accept_the_committed_frozen_artifacts() -> None:
    paths = repo_paths(ROOT)

    assert validation_module().validate_frozen_paper(paths) is None
    assert validation_module().validate_artifacts(paths) is None


@pytest.mark.parametrize("mutation", ["missing", "stale"])
def test_artifact_validation_rejects_an_invalid_owned_set(
    tmp_path: Path, mutation: str
) -> None:
    copy_artifact_surface(tmp_path)
    if mutation == "missing":
        (tmp_path / "figures_col" / "fig_var_irf.png").unlink()
    else:
        (tmp_path / "figures_col" / "stale_preview.png").write_bytes(b"stale")

    with pytest.raises(ValueError, match=r"(?i)(artifact|figure|missing|unexpected)"):
        validation_module().validate_artifacts(repo_paths(tmp_path))


def test_frozen_paper_validation_rejects_tampering(tmp_path: Path) -> None:
    copy_artifact_surface(tmp_path)
    with (tmp_path / "IAQF_column_Final.tex").open("ab") as paper:
        paper.write(b"\n% tampered\n")

    with pytest.raises(ValueError, match=r"(?i)(sha|hash|frozen|paper)"):
        validation_module().validate_frozen_paper(repo_paths(tmp_path))


def test_legacy_bootstrap_claim_is_explicitly_uncomputed_provenance() -> None:
    assert validation_module().LEGACY_BOOTSTRAP_CLAIM == {
        "method": "moving-block bootstrap",
        "replications": 5_000,
        "block_length_minutes": 60,
        "median_ratio": 567,
        "ci_95_ratio": (216, 1_898),
        "p_value": "<1e-4",
        "provenance_exception": True,
        "computed": False,
    }
