from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iaqf.config import EXPECTED_TABLES, RepoPaths
from iaqf.data import AnalysisData
from iaqf.metrics import (
    arbitrage_after_costs,
    daily_amihud_illiquidity,
    daily_regime_means,
    daily_roll_spread,
    daily_volume_shares,
    half_life_robustness,
    realized_volatility,
    var_irf,
)
from iaqf.tables import write_tables

ROOT = Path(__file__).resolve().parents[1]

CORRECTED_SHA256 = {
    "granger_causality.csv": (
        "81be76b984ae4799269feb8c16dad51c48cbd1e6adf4a964ff516eccb3cb88f9"
    ),
    "granger_causality_fdr.csv": (
        "20e59419fb8d193a19bd9bdcc0164445cce921052bad1a2ce57121faf67e319f"
    ),
    "regression_hac.tex": (
        "b425bc73ec233d35ca5f4112fdd39795c57f074a907030e5477368d815d1607b"
    ),
    "regression_results.txt": (
        "6eac7c566f5f2d89e4cf968dd7b96d9048be5df3d85413592c5c87e98eccf89a"
    ),
    "regression_usdc.txt": (
        "87aaf13bbb62f2a88da9e91bf6c74f1a6a61fa2f1668adad54848ec7d2266f2a"
    ),
    "regression_usdt.txt": (
        "1194109ff9785eab21dcaf3866345695ee85dfe45fc966b27f2f185d3d2d005a"
    ),
}


def committed_analysis_data() -> AnalysisData:
    processed = ROOT / "data_processed"
    return AnalysisData(
        prices=pd.read_parquet(processed / "prices.parquet"),
        price_ffill_flags=pd.read_parquet(processed / "price_ffill_flags.parquet"),
        ranges=pd.read_parquet(processed / "intraminute_ranges.parquet"),
        volumes=pd.read_parquet(processed / "volumes.parquet"),
        basis=pd.read_parquet(processed / "basis.parquet"),
        basis_ffill_flags=pd.read_parquet(processed / "basis_ffill_flags.parquet"),
    )


def test_daily_roll_spread_preserves_legacy_day_grouping_and_scaling() -> None:
    index = pd.date_range("2023-03-01", periods=17, freq="1min", tz="UTC")
    alternating_returns = [0.01, -0.01] * 8
    log_prices = np.concatenate(
        [[math.log(100.0)], math.log(100.0) + np.cumsum(alternating_returns)]
    )
    prices = pd.Series(np.exp(log_prices), index=index)

    result = daily_roll_spread(prices)

    assert result.index.tolist() == [pd.Timestamp("2023-03-01", tz="UTC")]
    assert result.iloc[0] == pytest.approx(206.5591117977289)


def test_daily_amihud_illiquidity_uses_close_times_volume_and_one_million_scale() -> (
    None
):
    index = pd.date_range("2023-03-01", periods=3, freq="1min", tz="UTC")
    prices = pd.Series([100.0, 110.0, 121.0], index=index)
    volumes = pd.Series([1.0, 2.0, 3.0], index=index)
    expected = math.log(1.1) * (1 / 220.0 + 1 / 363.0) / 2 * 1_000_000

    result = daily_amihud_illiquidity(prices, volumes)

    assert result.index.tolist() == [pd.Timestamp("2023-03-01", tz="UTC")]
    assert result.iloc[0] == pytest.approx(expected)


def test_arbitrage_after_costs_aligns_legs_and_clips_both_net_bounds() -> None:
    index = pd.date_range("2023-03-01", periods=3, freq="1min", tz="UTC")
    basis = pd.Series([-40.0, 12.0, 8.0], index=index)
    ranges = pd.DataFrame(
        {
            "leg_a": [0.001, np.nan, 0.0004],
            "leg_b": [0.002, 0.001, 0.0006],
        },
        index=index,
    )

    result = arbitrage_after_costs(
        basis,
        ranges,
        ("leg_a", "leg_b"),
        n_legs=2,
        fee_bps=5.0,
    )

    expected = pd.DataFrame(
        {
            "abs_basis": [40.0, 8.0],
            "fee": [10.0, 10.0],
            "slippage": [15.0, 5.0],
            "net_fee": [30.0, 0.0],
            "net_fee_slippage": [15.0, 0.0],
        },
        index=index[[0, 2]],
    )
    pd.testing.assert_frame_equal(result, expected, check_freq=False)


@pytest.mark.golden
def test_half_life_robustness_is_the_exact_shared_table_and_figure_frame() -> None:
    result = half_life_robustness(committed_analysis_data())

    assert list(result.columns) == [
        "series",
        "regime",
        "freq",
        "ff_filter",
        "rho_est",
        "half_life_min",
        "n_obs",
        "warning",
    ]
    assert (
        result.to_csv(index=False).encode()
        == (ROOT / "tables" / "half_life_robustness.csv").read_bytes()
    )


def test_var_irf_returns_paper_exact_orthogonalized_response_payload() -> None:
    result = var_irf(committed_analysis_data())

    np.testing.assert_array_equal(result["horizons"], np.arange(11))
    assert result["values"].shape == (11, 2, 2)
    assert result["selected_lags"] == 10
    assert result["n_obs"] == 26_548
    assert result["values"][1, 1, 1] == pytest.approx(-3.7446405630784927)
    assert result["lower"][10, 0, 1] == pytest.approx(-0.0704995473379884)
    assert result["upper"][10, 0, 1] == pytest.approx(0.13875851670984107)


def test_daily_regime_means_preserves_label_and_fixed_regime_order() -> None:
    series = {
        "Pair B": pd.Series(
            [np.nan, np.nan, np.nan],
            index=pd.to_datetime(
                ["2023-03-09", "2023-03-10", "2023-03-13"],
                utc=True,
            ),
        ),
        "Pair A": pd.Series(
            [1.2344, 2.3456, 3.4567],
            index=pd.to_datetime(
                ["2023-03-09", "2023-03-10", "2023-03-13"],
                utc=True,
            ),
        ),
    }

    result = daily_regime_means(series, ("Pair B", "Pair A"))

    assert result[["Pair", "Regime", "N"]].values.tolist() == [
        ["Pair B", "Pre-SVB", 0],
        ["Pair B", "Crisis", 0],
        ["Pair B", "Post-SVB", 0],
        ["Pair A", "Pre-SVB", 1],
        ["Pair A", "Crisis", 1],
        ["Pair A", "Post-SVB", 1],
    ]
    assert result.loc[result["Pair"] == "Pair A", "mean"].tolist() == [
        1.234,
        2.346,
        3.457,
    ]


def test_realized_volatility_uses_simple_returns_and_sixty_minute_window() -> None:
    index = pd.date_range("2023-03-01", periods=61, freq="1min", tz="UTC")
    prices = pd.DataFrame(
        {"market": 100.0 * np.power(1.01, np.arange(61))},
        index=index,
    )

    result = realized_volatility(prices, ("market",))

    assert result.columns.tolist() == ["market"]
    assert result.iloc[:60, 0].isna().all()
    assert result.iloc[60, 0] == pytest.approx(0.0, abs=1e-10)


def test_daily_volume_shares_uses_daily_sums_and_row_percentages() -> None:
    index = pd.date_range("2023-03-01", periods=2, freq="1min", tz="UTC")
    volumes = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 4.0]}, index=index)

    result = daily_volume_shares(volumes, ("A", "B"))

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {"A": [40.0], "B": [60.0]},
            index=pd.DatetimeIndex(["2023-03-01"], tz="UTC"),
        ),
        check_freq=False,
    )


@pytest.mark.golden
def test_write_tables_reproduces_frozen_surface_with_six_corrected_hashes(
    tmp_path: Path,
) -> None:
    paths = RepoPaths.from_root(tmp_path)
    paths.tables.mkdir(parents=True)
    (paths.tables / "stale.preview").write_bytes(b"stale")

    write_tables(committed_analysis_data(), paths)

    generated = {path.name: path for path in paths.tables.iterdir() if path.is_file()}
    assert set(generated) == set(EXPECTED_TABLES)
    for name, generated_path in generated.items():
        if name in CORRECTED_SHA256:
            digest = hashlib.sha256(generated_path.read_bytes()).hexdigest()
            assert digest == CORRECTED_SHA256[name]
        else:
            assert generated_path.read_bytes() == (ROOT / "tables" / name).read_bytes()
