from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from iaqf.config import RepoPaths
from iaqf.data import load_processed
from iaqf.tables import write_tables

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def analysis_data():
    return load_processed(RepoPaths.from_root(ROOT))


def test_recovered_parametric_bootstrap_reproduces_submitted_numbers(
    analysis_data,
) -> None:
    from iaqf.metrics import half_life_ratio_bootstrap

    result = half_life_ratio_bootstrap(analysis_data)

    assert result["method"] == "parametric_ar1_residual_sieve"
    assert result["replications"] == 10_000
    assert result["seed"] == 42
    assert result["point_ratio"] == pytest.approx(938.9842206184586, rel=1e-10)
    assert result["valid_replications"] == 9_994
    assert round(result["median_ratio"]) == 567
    assert round(result["ci_95_low"]) == 216
    assert round(result["ci_95_high"]) == 1_898
    assert result["p_value_r_le_1"] == 0.0


def test_committed_moving_block_bootstrap_exposes_method_mismatch(
    analysis_data,
) -> None:
    from iaqf.metrics import moving_block_half_life_ratio_bootstrap

    result = moving_block_half_life_ratio_bootstrap(analysis_data)

    assert result["method"] == "moving_block"
    assert result["replications"] == 5_000
    assert result["block_length_minutes"] == 60
    assert result["valid_replications"] == 5_000
    assert round(result["median_ratio"]) == 64
    assert round(result["ci_95_low"]) == 46
    assert round(result["ci_95_high"]) == 88
    assert result["p_value_r_le_1"] == 0.0


def test_structural_break_tests_reproduce_manuscript_values(analysis_data) -> None:
    from iaqf.metrics import structural_break_tests

    rows = structural_break_tests(analysis_data).set_index("Metric")

    assert round(rows.loc["USDC dispersion mean", "F statistic"]) == 2_089
    assert round(rows.loc["USDC adjusted residual mean", "F statistic"]) == 230
    assert round(rows.loc["USDC adjusted residual AR(1)", "F statistic"]) == 71
    assert (rows["p_value"] < 1e-16).all()


def test_normality_diagnostics_reproduce_crisis_jarque_bera(
    analysis_data,
) -> None:
    from iaqf.metrics import normality_diagnostics

    rows = normality_diagnostics(analysis_data).set_index(["Channel", "Regime"])
    crisis = rows.loc[("USDC", "Crisis")]

    assert round(crisis["JB statistic"]) == 30_498
    assert crisis["p_value"] < 1e-16
    assert crisis["N"] == 4_283


def test_fee_sensitivity_reproduces_manuscript_values(analysis_data) -> None:
    from iaqf.metrics import arbitrage_fee_sensitivity

    rows = arbitrage_fee_sensitivity(analysis_data).set_index(["Channel", "fee_bps"])

    assert rows.loc[
        ("USDC/USD (Kraken, 3-leg triangular)", 3.0),
        "pct_profitable_fee_slippage",
    ] == pytest.approx(14.032220406257295)
    assert rows.loc[
        ("USDC/USD (Kraken, 3-leg triangular)", 10.0),
        "pct_profitable_fee_slippage",
    ] == pytest.approx(0.9572729395283679)


def test_contagion_robustness_reproduces_no_ff_and_five_minute_rows(
    analysis_data,
) -> None:
    from iaqf.metrics import contagion_robustness

    rows = contagion_robustness(analysis_data).set_index("Specification")

    no_ff = rows.loc["Crisis no-FF"]
    assert no_ff["lambda"] == pytest.approx(-0.009237210644136962)
    assert no_ff["p_value"] == pytest.approx(0.0001323569241142272)
    assert no_ff["N"] == 3_001

    five_minute = rows.loc["Crisis 5-minute"]
    assert five_minute["lambda"] == pytest.approx(-0.011948234537673384)
    assert five_minute["p_value"] == pytest.approx(6.101494760718666e-06)
    assert five_minute["N"] == 861


def test_price_discovery_diagnostic_reproduces_residual_correlation(
    analysis_data,
) -> None:
    from iaqf.metrics import price_discovery_diagnostics

    result = price_discovery_diagnostics(analysis_data)

    assert result["channel"] == "Kraken BTC/USD vs BTC/USDT"
    assert result["n_obs_no_ff"] == 20_415
    assert result["k_ar_diff"] == 8
    assert result["residual_correlation"] == pytest.approx(0.8881645900548751)


def test_irf_diagnostics_surface_the_submitted_all_horizons_mismatch(
    analysis_data,
) -> None:
    from iaqf.metrics import var_irf_diagnostics

    rows = var_irf_diagnostics(analysis_data)
    reverse = rows.loc[rows["Direction"] == "BTC/USDC -> BTC/USD"].set_index("Horizon")
    forward = rows.loc[rows["Direction"] == "BTC/USD -> BTC/USDC"].set_index("Horizon")

    assert forward.loc[0, "Response (bps)"] == pytest.approx(6.132133239973696)
    assert reverse.loc[3, "CI low"] == pytest.approx(-0.23766451745769165)
    assert reverse.loc[3, "CI high"] == pytest.approx(-0.01046918440540151)
    assert bool(reverse.loc[3, "CI contains zero"]) is False
    assert bool(reverse["CI contains zero"].all()) is False


def test_table_writer_emits_recovered_diagnostics_and_claim_audit(
    analysis_data,
    tmp_path: Path,
) -> None:
    paths = RepoPaths.from_root(tmp_path)

    write_tables(analysis_data, paths)

    expected = {
        "arbitrage_fee_sensitivity.csv",
        "contagion_robustness.csv",
        "half_life_ratio_bootstrap.csv",
        "normality_diagnostics.csv",
        "paper_claim_audit.csv",
        "price_discovery_diagnostics.csv",
        "structural_break_tests.csv",
        "var_irf_diagnostics.csv",
    }
    assert expected <= {path.name for path in paths.tables.iterdir()}

    bootstrap_rows = pd.read_csv(paths.tables / "half_life_ratio_bootstrap.csv")
    bootstrap = bootstrap_rows.set_index("method").loc["parametric_ar1_residual_sieve"]
    assert round(bootstrap["median_ratio"]) == 567
    assert round(bootstrap["ci_95_low"]) == 216
    assert round(bootstrap["ci_95_high"]) == 1_898
    moving_block = bootstrap_rows.set_index("method").loc["moving_block"]
    assert round(moving_block["median_ratio"]) == 64
    assert round(moving_block["ci_95_low"]) == 46
    assert round(moving_block["ci_95_high"]) == 88

    audit = pd.read_csv(paths.tables / "paper_claim_audit.csv").set_index("claim_id")
    assert audit.loc["bootstrap_numeric_interval", "status"] == "verified"
    assert audit.loc["bootstrap_method", "status"] == "method_mismatch"
    assert audit.loc["reverse_irf_ci_all_horizons", "status"] == "contradicted"
