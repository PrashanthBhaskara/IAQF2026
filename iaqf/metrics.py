"""Pure paper-exact statistical computations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR

from iaqf.config import CRISIS_END, CRISIS_START, STUDY_END, STUDY_START

_GRANGER_PAIRS = [
    ("kraken_btcusd", "kraken_btcusdc", "BTC/USDC -> BTC/USD (Kraken)"),
    ("kraken_btcusdc", "kraken_btcusd", "BTC/USD -> BTC/USDC (Kraken)"),
    ("kraken_btcusd", "kraken_btcusdt", "BTC/USDT -> BTC/USD (Kraken)"),
    ("kraken_btcusdt", "kraken_btcusd", "BTC/USD -> BTC/USDT (Kraken)"),
    ("binance_btcusdt", "kraken_btcusdt", "Kraken USDT -> Binance USDT"),
    ("kraken_btcusdt", "binance_btcusdt", "Binance USDT -> Kraken USDT"),
    ("coinbase_btcusd", "kraken_btcusd", "Kraken USD -> Coinbase USD"),
    ("kraken_btcusd", "coinbase_btcusd", "Coinbase USD -> Kraken USD"),
]


def global_complete_case_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Return one-minute simple returns on the common complete-case sample."""
    return prices.pct_change(fill_method=None).dropna()


def study_regime(timestamp: pd.Timestamp) -> str:
    """Classify a UTC timestamp within the fixed study window."""
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        raise ValueError("study timestamp must be timezone-aware")
    value = value.tz_convert("UTC")
    if value < STUDY_START or value > STUDY_END:
        raise ValueError(f"timestamp outside fixed study window: {value}")
    if value < CRISIS_START:
        return "Pre-SVB"
    if value < CRISIS_END:
        return "Crisis"
    return "Post-SVB"


def hac_regressions(
    prices: pd.DataFrame,
    basis: pd.DataFrame,
    ranges: pd.DataFrame,
) -> pd.DataFrame:
    """Fit the two paper HAC regressions using the frozen specification."""
    returns = global_complete_case_returns(prices)
    specifications = [
        ("USDC", "basis_usdc_kraken", "kraken_btcusdc"),
        ("USDT", "basis_usdt_kraken", "kraken_btcusdt"),
    ]
    rows = []
    for stablecoin, basis_column, market_column in specifications:
        regression = pd.DataFrame(
            {
                "Basis": basis[basis_column],
                "Crisis": (
                    (basis.index >= CRISIS_START) & (basis.index < CRISIS_END)
                ).astype(int),
                "RealizedVol60m": returns[market_column].rolling(60).std() * 10_000,
                "RangeProxy": ranges[market_column] * 10_000,
            },
            index=basis.index,
        ).dropna()
        design = sm.add_constant(regression[["Crisis", "RealizedVol60m", "RangeProxy"]])
        model = sm.OLS(regression["Basis"], design).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 60},
        )
        rows.append(
            {
                "stablecoin": stablecoin,
                "constant": float(model.params["const"]),
                "crisis": float(model.params["Crisis"]),
                "realized_vol_60m": float(model.params["RealizedVol60m"]),
                "range_proxy": float(model.params["RangeProxy"]),
                "r_squared": float(model.rsquared),
                "n_obs": int(model.nobs),
                "crisis_p_value": float(model.pvalues["Crisis"]),
            }
        )
    return pd.DataFrame(rows)


def granger_causality(prices: pd.DataFrame) -> pd.DataFrame:
    """Run the eight frozen VAR causality tests and BH/FDR correction."""
    returns = global_complete_case_returns(prices)
    rows = []
    for dependent, independent, label in _GRANGER_PAIRS:
        sample = returns[[dependent, independent]].dropna() * 10_000
        if len(sample) < 200:
            continue
        result = VAR(sample.to_numpy()).fit(maxlags=10, ic="aic")
        test = result.test_causality(0, 1, kind="f")
        rows.append(
            {
                "test": label,
                "var_lags": int(result.k_ar),
                "f_stat": float(test.test_statistic),
                "p_value": float(test.pvalue),
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        output["q_value"] = pd.Series(dtype=float)
        output["significant_fdr"] = pd.Series(dtype=bool)
        return output
    _, q_values, _, _ = multipletests(output["p_value"], method="fdr_bh")
    output["q_value"] = q_values
    output["significant_fdr"] = output["q_value"] < 0.05
    return output


def half_life_from_rho(rho: float, dt_minutes: float) -> float:
    """Map a levels AR(1) coefficient to its exact half-life in minutes."""
    if dt_minutes <= 0:
        raise ValueError(f"dt_minutes must be positive; got {dt_minutes}")
    if not np.isfinite(rho) or rho <= 0.0 or rho >= 1.0:
        return float("nan")
    return float((np.log(2.0) * dt_minutes) / -np.log(rho))


def estimate_half_life_from_ecm(
    series: pd.Series,
    dt_minutes: float,
    ff_mask: pd.Series | None = None,
    min_obs: int = 100,
) -> dict[str, float | int | str]:
    """Estimate the implied AR(1) half-life from an ECM regression."""
    clean = series.dropna().astype(float)
    if len(clean) < min_obs:
        return _insufficient_half_life(len(clean), "obs_too_few")

    regression = pd.DataFrame({"x_t": clean})
    regression["x_lag"] = regression["x_t"].shift(1)
    regression["dx_t"] = regression["x_t"] - regression["x_lag"]
    if ff_mask is None:
        regression["ff_t"] = False
        regression["ff_lag"] = False
    else:
        filled = ff_mask.reindex(regression.index).fillna(False).astype(bool)
        regression["ff_t"] = filled
        regression["ff_lag"] = filled.shift(1, fill_value=False).astype(bool)

    regression = regression.dropna(subset=["dx_t", "x_lag"])
    regression = regression.loc[~(regression["ff_t"] | regression["ff_lag"])]
    if len(regression) < min_obs:
        return _insufficient_half_life(len(regression), "obs_too_few_no_ff")

    design = sm.add_constant(regression["x_lag"])
    model = sm.OLS(regression["dx_t"], design).fit()
    beta = float(model.params.iloc[1])
    rho = 1.0 + beta
    warning = "" if 0.0 < rho < 1.0 else "rho_invalid"
    return {
        "estimation_form": "ECM (implied AR1)",
        "beta_est": beta,
        "kappa_est": -beta,
        "rho_est": rho,
        "half_life_min": half_life_from_rho(rho, dt_minutes),
        "n_obs": int(len(regression)),
        "warning": warning,
    }


def _insufficient_half_life(
    n_obs: int,
    warning: str,
) -> dict[str, float | int | str]:
    return {
        "estimation_form": "ECM (implied AR1)",
        "beta_est": float("nan"),
        "kappa_est": float("nan"),
        "rho_est": float("nan"),
        "half_life_min": float("nan"),
        "n_obs": n_obs,
        "warning": warning,
    }
