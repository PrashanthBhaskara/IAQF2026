from __future__ import annotations

import importlib
import math

import pandas as pd
import pytest


def metric(name: str):
    try:
        module = importlib.import_module("iaqf.metrics")
    except (ImportError, ModuleNotFoundError):
        pytest.fail("planned iaqf.metrics API is missing")
    assert hasattr(module, name), f"planned metric is missing: {name}"
    return getattr(module, name)


def test_half_life_from_rho_uses_exact_ar1_mapping() -> None:
    half_life_from_rho = metric("half_life_from_rho")

    assert half_life_from_rho(0.5, dt_minutes=5.0) == pytest.approx(5.0)


@pytest.mark.parametrize("rho", [-0.2, 0.0, 1.0, 1.1, math.nan])
def test_half_life_from_rho_is_undefined_outside_mean_reverting_domain(
    rho: float,
) -> None:
    half_life_from_rho = metric("half_life_from_rho")

    assert math.isnan(half_life_from_rho(rho, dt_minutes=1.0))


def test_half_life_from_rho_rejects_nonpositive_sampling_interval() -> None:
    half_life_from_rho = metric("half_life_from_rho")

    with pytest.raises(ValueError, match="dt_minutes"):
        half_life_from_rho(0.5, dt_minutes=0.0)


def test_ecm_half_life_removes_filled_current_and_lagged_observations() -> None:
    estimate_half_life_from_ecm = metric("estimate_half_life_from_ecm")
    index = pd.date_range("2023-03-01", periods=7, freq="1min", tz="UTC")
    series = pd.Series(
        [0.0, 1.0, 1.5, 1.75, 1.875, 1.9375, 1.96875],
        index=index,
    )
    filled = pd.Series(
        [False, False, False, True, False, False, False],
        index=index,
    )

    result = estimate_half_life_from_ecm(
        series,
        dt_minutes=1.0,
        ff_mask=filled,
        min_obs=4,
    )

    assert result["rho_est"] == pytest.approx(0.5)
    assert result["half_life_min"] == pytest.approx(1.0)
    assert result["n_obs"] == 4
    assert result["warning"] == ""


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("2023-03-01 00:00:00+00:00", "Pre-SVB"),
        ("2023-03-09 23:59:00+00:00", "Pre-SVB"),
        ("2023-03-10 00:00:00+00:00", "Crisis"),
        ("2023-03-12 23:59:00+00:00", "Crisis"),
        ("2023-03-13 00:00:00+00:00", "Post-SVB"),
        ("2023-03-21 23:59:00+00:00", "Post-SVB"),
    ],
)
def test_study_regime_uses_fixed_half_open_crisis_boundaries(
    timestamp: str, expected: str
) -> None:
    study_regime = metric("study_regime")

    assert study_regime(pd.Timestamp(timestamp)) == expected


@pytest.mark.parametrize(
    "timestamp",
    ["2023-02-28 23:59:00+00:00", "2023-03-22 00:00:00+00:00"],
)
def test_study_regime_rejects_timestamps_outside_fixed_study(timestamp: str) -> None:
    study_regime = metric("study_regime")

    with pytest.raises(ValueError, match="study"):
        study_regime(pd.Timestamp(timestamp))
