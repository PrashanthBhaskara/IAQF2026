from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def paper_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed = ROOT / "data_processed"
    return (
        pd.read_parquet(processed / "prices.parquet"),
        pd.read_parquet(processed / "basis.parquet"),
        pd.read_parquet(processed / "intraminute_ranges.parquet"),
        pd.read_parquet(processed / "price_ffill_flags.parquet"),
    )


def metrics_module():
    return importlib.import_module("iaqf.metrics")


@pytest.mark.parametrize(
    ("artifact", "selectors", "expected_cells"),
    [
        pytest.param(
            "data_coverage_core.csv",
            {"Pair": "Kraken BTC/USDC"},
            {
                "Coverage Pre-SVB (%)": (89.62, 2),
                "Forward-Fill Share Overall (%)": (42.07, 2),
            },
            id="data-coverage",
        ),
        pytest.param(
            "dispersion_adjusted_stats.csv",
            {"Regime": "Crisis", "Series": "USDC Kraken $D_t$ (Unadjusted)"},
            {"Mean (bps)": (319.97, 2), "Std (bps)": (322.87, 2), "N": (4_311, 0)},
            id="dispersion",
        ),
        pytest.param(
            "contagion_intensity.csv",
            {"Channel": "USDC", "Regime": "Crisis"},
            {
                "lambda": (-0.009, 3),
                "t_stat": (-4.77, 2),
                "R2": (0.351, 3),
                "N": (4_282, 0),
            },
            id="contagion",
        ),
        pytest.param(
            "depth_proxy_table.csv",
            {"Pair": "Coinbase BTC/USD", "Regime": "Post-SVB"},
            {"median_usd_mm": (664.46, 2)},
            id="depth",
        ),
        pytest.param(
            "distributional_robustness.csv",
            {"Channel": "USDC", "Regime": "Crisis"},
            {
                "Skewness": (1.01, 2),
                "Excess Kurtosis": (12.9, 1),
                "Min": (-165.2, 1),
                "Max": (180.6, 1),
            },
            id="distributional-robustness",
        ),
        pytest.param(
            "cointegration_johansen.csv",
            {"channel": "Kraken BTC/USD vs BTC/USDT"},
            {
                "rank_used": (1, 0),
                "k_ar_diff_used": (8, 0),
                "trace_stat_r0": (27.03, 2),
            },
            id="cointegration-rank",
        ),
        pytest.param(
            "price_discovery_metrics.csv",
            {"channel": "Kraken BTC/USD vs BTC/USDT"},
            {"alpha_market_1": (0.0079, 4), "alpha_market_2": (0.0118, 4)},
            id="vecm-adjustment",
        ),
        pytest.param(
            "hasbrouck_is.csv",
            {"Channel": "Kraken BTC/USD vs BTC/USDT"},
            {
                "IS_USD_lower": (0.69, 2),
                "IS_USD_upper": (0.77, 2),
                "IS_USD_mid": (0.73, 2),
            },
            id="price-discovery-share",
        ),
        pytest.param(
            "arbitrage_summary.csv",
            {
                "channel": "USDC/USD (Kraken, 3-leg triangular)",
                "regime": "Crisis",
                "cost_variant": "fee_plus_slippage_conservative",
            },
            {"pct_profitable": (6.72, 2), "avg_net_uncond_bps": (0.64, 2)},
            id="arbitrage",
        ),
        pytest.param(
            "genius_counterfactual.csv",
            {"Scenario": "Moderate mitigation"},
            {
                "Assumed mitigation of lock-up shock (%)": (50, 0),
                "Implied crisis D_t mean (bps)": (160.0, 1),
                "Reduction vs observed (bps)": (159.9, 1),
            },
            id="genius",
        ),
    ],
)
def test_generated_csv_cells_match_inline_manuscript_literals(
    artifact: str,
    selectors: dict[str, str],
    expected_cells: dict[str, tuple[float, int]],
) -> None:
    table = pd.read_csv(ROOT / "tables" / artifact)
    selected = table
    for column, value in selectors.items():
        selected = selected.loc[selected[column] == value]

    assert len(selected) == 1
    row = selected.iloc[0]
    for column, (expected, decimal_places) in expected_cells.items():
        assert round(float(row[column]), decimal_places) == expected


def test_liquidity_generated_row_matches_inline_manuscript_literals() -> None:
    artifact = (ROOT / "tables" / "liquidity_spread_table.tex").read_text(
        encoding="utf-8"
    )
    row = next(
        line for line in artifact.splitlines() if line.startswith("Kraken BTC/USDC &")
    )
    cells = [cell.strip().removesuffix(r"\\").strip() for cell in row.split("&")]

    assert cells == [
        "Kraken BTC/USDC",
        "1.100",
        "25.640",
        "2.070",
        "2",
        "2",
        "2",
        "31.970",
        "13.282",
        "23.530",
    ]


def test_global_complete_case_returns_preserves_frozen_full_panel_sample(
    paper_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    prices, _, _, _ = paper_inputs

    returns = metrics_module().global_complete_case_returns(prices)

    assert returns.shape == (26_548, 14)
    assert returns.index[0] == pd.Timestamp("2023-03-01 00:02:00+00:00")
    assert returns.index[-1] == pd.Timestamp("2023-03-21 23:59:00+00:00")
    assert not returns.isna().any(axis=None)
    assert returns.loc[returns.index[0], "kraken_btcusd"] == pytest.approx(
        0.0005314391631776605
    )


@pytest.mark.parametrize(
    (
        "stablecoin",
        "constant",
        "crisis",
        "realized_vol_60m",
        "range_proxy",
        "r_squared",
        "n_obs",
        "crisis_p_value",
    ),
    [
        (
            "USDC",
            -0.07883410982985607,
            4.295405618088111,
            0.004376241404463097,
            0.0480220729776178,
            0.04055920703458038,
            26_489,
            1.5396035216605183e-07,
        ),
        (
            "USDT",
            -0.6562442794874738,
            1.6546631806684189,
            -0.029419308377120226,
            -0.012527108040280586,
            0.012045007517985096,
            26_489,
            1.303215299763057e-05,
        ),
    ],
)
def test_hac_regressions_match_frozen_paper_rows(
    paper_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    stablecoin: str,
    constant: float,
    crisis: float,
    realized_vol_60m: float,
    range_proxy: float,
    r_squared: float,
    n_obs: int,
    crisis_p_value: float,
) -> None:
    prices, basis, ranges, _ = paper_inputs

    rows = metrics_module().hac_regressions(prices, basis, ranges)
    row = rows.set_index("stablecoin").loc[stablecoin]

    assert row["constant"] == pytest.approx(constant, abs=1e-12)
    assert row["crisis"] == pytest.approx(crisis, abs=1e-12)
    assert row["realized_vol_60m"] == pytest.approx(realized_vol_60m, abs=1e-12)
    assert row["range_proxy"] == pytest.approx(range_proxy, abs=1e-12)
    assert row["r_squared"] == pytest.approx(r_squared, abs=1e-12)
    assert row["n_obs"] == n_obs
    assert row["crisis_p_value"] == pytest.approx(crisis_p_value, rel=1e-10)


def test_reverse_granger_q_value_rounds_to_paper_display(
    paper_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    prices, _, _, _ = paper_inputs

    rows = metrics_module().granger_causality(prices)
    reverse = rows.set_index("test").loc["BTC/USDC -> BTC/USD (Kraken)"]

    assert len(rows) == 8
    assert reverse["var_lags"] == 10
    assert reverse["q_value"] == pytest.approx(0.12789750300468838, abs=1e-12)
    assert round(float(reverse["q_value"]), 2) == 0.13
    assert bool(reverse["significant_fdr"]) is False
