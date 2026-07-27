"""Pure paper-exact statistical computations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order

from iaqf.config import CRISIS_END, CRISIS_START, STUDY_END, STUDY_START

if TYPE_CHECKING:
    from iaqf.data import AnalysisData

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


def daily_roll_spread(prices: pd.Series) -> pd.Series:
    """Estimate daily Roll (1984) effective spread in basis points."""
    clean = prices.dropna()
    log_returns = np.log(clean / clean.shift(1))
    rows = []
    for date, group in log_returns.groupby(log_returns.index.date):
        values = group.dropna().to_numpy()
        if len(values) < 15:
            continue
        covariance = np.cov(values[1:], values[:-1])[0, 1]
        spread = 2.0 * np.sqrt(-covariance) * 10_000 if covariance < 0 else np.nan
        rows.append({"date": pd.Timestamp(date), "roll_bps": spread})
    if not rows:
        return pd.Series(dtype=float)
    result = pd.DataFrame(rows).set_index("date")["roll_bps"]
    result.index = pd.DatetimeIndex(result.index).tz_localize("UTC")
    return result


def daily_amihud_illiquidity(
    prices: pd.Series,
    volumes: pd.Series,
) -> pd.Series:
    """Calculate daily mean Amihud illiquidity, scaled by one million."""
    absolute_returns = np.log(prices / prices.shift(1)).abs()
    dollar_volume = volumes * prices
    aligned = pd.concat(
        [absolute_returns, dollar_volume],
        axis=1,
        keys=["abs_ret", "dvol"],
    ).dropna()
    aligned = aligned.loc[aligned["dvol"] > 1.0]
    illiquidity = aligned["abs_ret"] / aligned["dvol"]
    daily = illiquidity.groupby(illiquidity.index.date).mean() * 1_000_000
    daily.index = pd.DatetimeIndex(daily.index).tz_localize("UTC")
    return daily


def daily_dollar_volume(prices: pd.Series, volumes: pd.Series) -> pd.Series:
    """Aggregate close-times-volume to UTC daily totals."""
    values = (prices * volumes).dropna()
    daily = values.groupby(values.index.date).sum()
    daily.index = pd.DatetimeIndex(daily.index).tz_localize("UTC")
    return daily


def daily_regime_means(
    series_by_label: Mapping[str, pd.Series],
    label_order: tuple[str, ...],
) -> pd.DataFrame:
    """Summarize daily series in fixed regime and caller-provided label order."""
    rows = []
    for label in label_order:
        series = series_by_label[label]
        for regime, mask in [
            ("Pre-SVB", series.index < CRISIS_START),
            (
                "Crisis",
                (series.index >= CRISIS_START) & (series.index < CRISIS_END),
            ),
            ("Post-SVB", series.index >= CRISIS_END),
        ]:
            values = series.loc[mask].dropna()
            rows.append(
                {
                    "Pair": label,
                    "Regime": regime,
                    "mean": round(values.mean(), 3) if len(values) else np.nan,
                    "N": len(values),
                }
            )
    return pd.DataFrame(rows)


def realized_volatility(
    prices: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Return 60-minute rolling simple-return volatility in bps per hour."""
    return (
        prices[list(columns)].pct_change(fill_method=None).rolling(60).std()
        * 10_000
        * np.sqrt(60)
    )


def daily_volume_shares(
    volumes: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Return each selected market's percent share of daily volume."""
    daily = volumes[list(columns)].resample("D").sum()
    return daily.div(daily.sum(axis=1), axis=0) * 100


def arbitrage_after_costs(
    basis: pd.Series,
    ranges: pd.DataFrame,
    range_columns: tuple[str, ...],
    n_legs: int,
    fee_bps: float = 5.0,
) -> pd.DataFrame:
    """Calculate aligned fee-only and fee-plus-slippage arbitrage bounds."""
    aligned = pd.DataFrame({"abs_basis": basis.abs()})
    leg_columns = []
    for position, column in enumerate(range_columns):
        leg_column = f"_range_{position}"
        aligned[leg_column] = ranges[column] * 10_000
        leg_columns.append(leg_column)
    aligned = aligned.dropna()
    fee = n_legs * fee_bps
    slippage = 0.5 * aligned[leg_columns].sum(axis=1)
    return pd.DataFrame(
        {
            "abs_basis": aligned["abs_basis"],
            "fee": fee,
            "slippage": slippage,
            "net_fee": (aligned["abs_basis"] - fee).clip(lower=0.0),
            "net_fee_slippage": (aligned["abs_basis"] - (fee + slippage)).clip(
                lower=0.0
            ),
        },
        index=aligned.index,
    )


def hac_regressions(
    prices: pd.DataFrame,
    basis: pd.DataFrame,
    ranges: pd.DataFrame,
) -> pd.DataFrame:
    """Fit the two paper HAC regressions using the frozen specification."""
    rows = []
    for stablecoin, model in _fit_hac_models(prices, basis, ranges).items():
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


def fit_hac_regression_models(data: AnalysisData) -> dict[str, object]:
    """Fit the two paper HAC models for deterministic summary serialization."""
    return _fit_hac_models(data.prices, data.basis, data.ranges)


def _fit_hac_models(
    prices: pd.DataFrame,
    basis: pd.DataFrame,
    ranges: pd.DataFrame,
) -> dict[str, object]:
    returns = global_complete_case_returns(prices)
    models = {}
    for label, basis_column, market_column in [
        ("USDC", "basis_usdc_kraken", "kraken_btcusdc"),
        ("USDT", "basis_usdt_kraken", "kraken_btcusdt"),
    ]:
        frame = pd.DataFrame(
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
        design = sm.add_constant(frame[["Crisis", "RealizedVol60m", "RangeProxy"]])
        models[label] = sm.OLS(frame["Basis"], design).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 60},
        )
    return models


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


def var_irf(data: AnalysisData) -> dict[str, np.ndarray | int]:
    """Return the paper's orthogonalized Kraken BTC/USD-USDC VAR IRF payload."""
    returns = global_complete_case_returns(data.prices)
    values = returns[["kraken_btcusd", "kraken_btcusdc"]].to_numpy() * 10_000
    fit = VAR(values).fit(maxlags=10, ic="aic")
    result = fit.irf(10)
    responses = result.orth_irfs
    standard_errors = result.stderr(orth=True)
    return {
        "horizons": np.arange(len(responses)),
        "values": responses,
        "lower": responses - 1.96 * standard_errors,
        "upper": responses + 1.96 * standard_errors,
        "selected_lags": int(fit.k_ar),
        "n_obs": len(values),
    }


def ou_regime_statistics(data: AnalysisData) -> pd.DataFrame:
    """Compute the fixed-regime ECM half-life and ADF statistics."""
    rows = []
    for regime, start, end in _study_regimes(data.basis.index):
        mask = (data.basis.index >= start) & (data.basis.index < end)
        for column in data.basis:
            series = data.basis.loc[mask, column]
            clean = series.dropna()
            if len(clean) < 100:
                continue
            estimate = estimate_half_life_from_ecm(series, 1.0, min_obs=100)
            adf_stat, adf_p = adfuller(clean, maxlag=5)[:2]
            rows.append(
                {
                    "Regime": regime,
                    "Basis": column,
                    "Mean (bps)": round(clean.mean(), 2),
                    "Std (bps)": round(clean.std(), 2),
                    "Estimation Form": estimate["estimation_form"],
                    "rho_est": (
                        round(float(estimate["rho_est"]), 6)
                        if np.isfinite(estimate["rho_est"])
                        else np.nan
                    ),
                    "Half-Life (min)": (
                        round(float(estimate["half_life_min"]), 2)
                        if np.isfinite(estimate["half_life_min"])
                        else np.nan
                    ),
                    "ADF Stat": round(adf_stat, 2),
                    "ADF p-value": f"{adf_p:.4f}",
                    "N": len(clean),
                    "HL Warning": estimate["warning"],
                }
            )
    return pd.DataFrame(rows)


def hac_headline_metrics(data: AnalysisData) -> pd.DataFrame:
    """Compute HAC confidence intervals for the six headline series."""
    index = data.prices.index
    crisis = (index >= CRISIS_START) & (index < CRISIS_END)
    calm = (index < CRISIS_START) | (index >= CRISIS_END)
    specifications = [
        (
            "USDC dispersion $D_t$ (Kraken, crisis)",
            data.basis.loc[crisis, "dispersion_usdc_kraken"],
        ),
        (
            "USDC adjusted residual $B_t$ (Kraken, crisis)",
            data.basis.loc[crisis, "basis_usdc_kraken"],
        ),
        (
            "USDT premium to USD (Kraken, crisis)",
            (data.prices.loc[crisis, "kraken_usdtusd"] - 1) * 10_000,
        ),
        (
            "USDT premium to USD (Coinbase, crisis)",
            (data.prices.loc[crisis, "coinbase_usdtusd"] - 1) * 10_000,
        ),
        (
            "BTC/USDT cross-exchange basis (Binance-Kraken, calm)",
            data.basis.loc[calm, "xbasis_btcusdt_binance_kraken"],
        ),
        (
            "BTC/USD cross-exchange basis (Coinbase-Kraken, full)",
            data.basis["xbasis_btcusd_coinbase_kraken"],
        ),
    ]
    rows = []
    for name, series in specifications:
        values = series.dropna()
        model = sm.OLS(values.to_numpy(), np.ones((len(values), 1))).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": min(60, max(1, len(values) // 10))},
        )
        mean, standard_error = float(model.params[0]), float(model.bse[0])
        rows.append(
            {
                "Metric": name,
                "Mean (bps)": mean,
                "HAC SE (bps)": standard_error,
                "95% CI low": mean - 1.96 * standard_error,
                "95% CI high": mean + 1.96 * standard_error,
                "N": len(values),
            }
        )
    return pd.DataFrame(rows).round(
        {
            "Mean (bps)": 3,
            "HAC SE (bps)": 3,
            "95% CI low": 3,
            "95% CI high": 3,
        }
    )


def distributional_moments(data: AnalysisData) -> pd.DataFrame:
    """Compute fixed-regime higher moments of the two adjusted bases."""
    rows = []
    regimes = [
        ("Pre-SVB", data.basis.index < CRISIS_START),
        (
            "Crisis",
            (data.basis.index >= CRISIS_START) & (data.basis.index < CRISIS_END),
        ),
        ("Post-SVB", data.basis.index >= CRISIS_END),
    ]
    for channel, column in [
        ("USDC", "basis_usdc_kraken"),
        ("USDT", "basis_usdt_kraken"),
    ]:
        for regime, mask in regimes:
            values = data.basis.loc[mask, column].dropna()
            rows.append(
                {
                    "Channel": channel,
                    "Regime": regime,
                    "N": len(values),
                    "Mean": float(values.mean()),
                    "Std": float(values.std()),
                    "Skewness": float(scipy_stats.skew(values)),
                    "Excess Kurtosis": float(scipy_stats.kurtosis(values)),
                    "Min": float(values.min()),
                    "Max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def contagion_intensity(data: AnalysisData) -> pd.DataFrame:
    """Estimate the paper's fixed-regime coupled-OU contagion rows."""
    rows = []
    for channel, basis_column, peg_column in [
        ("USDC", "basis_usdc_kraken", "usdc_peg_dev_kraken"),
        ("USDT", "basis_usdt_kraken", "usdt_peg_dev_kraken"),
    ]:
        for regime, start, end in _study_regimes(data.basis.index):
            mask = (data.basis.index >= start) & (data.basis.index < end)
            frame = pd.DataFrame(
                {
                    "B": data.basis.loc[mask, basis_column],
                    "S": data.basis.loc[mask, peg_column],
                }
            ).dropna()
            frame["B_lag"] = frame["B"].shift(1)
            frame["S_lag"] = frame["S"].shift(1)
            frame["dB"] = frame["B"] - frame["B_lag"]
            frame = frame.dropna()
            design = sm.add_constant(frame[["B_lag", "S_lag"]])
            model = sm.OLS(frame["dB"], design).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": 60},
            )
            coefficient = float(model.params["S_lag"])
            standard_error = float(model.bse["S_lag"])
            rows.append(
                {
                    "Channel": channel,
                    "Regime": regime,
                    "lambda": coefficient,
                    "SE": standard_error,
                    "t_stat": coefficient / standard_error,
                    "p_value": float(model.pvalues["S_lag"]),
                    "R2": float(model.rsquared),
                    "N": int(model.nobs),
                }
            )
    return pd.DataFrame(rows)


def price_discovery(data: AnalysisData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the paper's Johansen and VECM price-discovery frames."""
    johansen_rows, discovery_rows = [], []
    channels = [
        (
            "Kraken BTC/USD vs BTC/USDC",
            "Kraken BTC/USD",
            "Kraken BTC/USDC",
            "kraken_btcusd",
            "kraken_btcusdc",
        ),
        (
            "Kraken BTC/USD vs BTC/USDT",
            "Kraken BTC/USD",
            "Kraken BTC/USDT",
            "kraken_btcusd",
            "kraken_btcusdt",
        ),
    ]
    for channel, market_1, market_2, column_1, column_2 in channels:
        levels = pd.DataFrame(
            {
                "p1": np.log(data.prices[column_1]),
                "p2": np.log(data.prices[column_2]),
            }
        )
        flags = data.price_ffill_flags[[column_1, column_2]].any(axis=1)
        levels = levels.loc[~flags].dropna()
        values = levels.to_numpy()
        selection = select_order(values, maxlags=10, deterministic="ci")
        aic, bic = (
            selection.selected_orders.get("aic"),
            selection.selected_orders.get("bic"),
        )
        selected = int(max(bic if bic is not None else (aic or 2), 1))
        lag = max(selected - 1, 0)
        base = _johansen(values, lag)
        minus = _johansen(values, max(lag - 1, 0))
        plus = _johansen(values, lag + 1)
        rank = min(int(base["rank_95"]), 1)
        johansen_rows.append(
            {
                "channel": channel,
                "market_1": market_1,
                "market_2": market_2,
                "n_obs_no_ff": int(len(levels)),
                "deterministic": "constant_in_cointegration",
                "lag_rule": "BIC (fallback AIC)",
                "selected_p_aic": int(aic) if aic is not None else np.nan,
                "selected_p_bic": int(bic) if bic is not None else np.nan,
                "selected_p_used": selected,
                "k_ar_diff_used": lag,
                "trace_stat_r0": base["trace_stat_r0"],
                "trace_crit95_r0": base["trace_crit95_r0"],
                "reject_r0_95": base["reject_r0_95"],
                "trace_stat_r1": base["trace_stat_r1"],
                "trace_crit95_r1": base["trace_crit95_r1"],
                "reject_r1_95": base["reject_r1_95"],
                "rank_95": base["rank_95"],
                "rank_used": rank,
                "rank_95_lag_minus1": minus["rank_95"],
                "rank_95_lag_plus1": plus["rank_95"],
            }
        )
        empty = {
            "channel": channel,
            "market_1": market_1,
            "market_2": market_2,
            "k_ar_diff_used": lag,
            "rank_used": rank,
        }
        if rank < 1:
            discovery_rows.append(
                empty
                | {
                    key: value
                    for key, value in [
                        ("alpha_market_1", np.nan),
                        ("alpha_market_2", np.nan),
                        ("abs_alpha_market_1", np.nan),
                        ("abs_alpha_market_2", np.nan),
                        ("gg_share_market_1", np.nan),
                        ("gg_share_market_2", np.nan),
                        ("leader_by_adjustment", "undetermined_no_cointegration"),
                        ("follower_by_adjustment", "undetermined_no_cointegration"),
                        ("gg_warning", "no_rank1_cointegration"),
                        ("leader_stable_lag_pm1", np.nan),
                        ("alpha_m1_lag_minus1", np.nan),
                        ("alpha_m1_lag_plus1", np.nan),
                        ("alpha_m2_lag_minus1", np.nan),
                        ("alpha_m2_lag_plus1", np.nan),
                    ]
                }
            )
            continue
        alpha = (
            VECM(
                values,
                k_ar_diff=lag,
                coint_rank=rank,
                deterministic="ci",
            )
            .fit()
            .alpha[:, 0]
        )
        first, second = float(alpha[0]), float(alpha[1])
        if np.isclose(abs(first), abs(second)):
            leader = follower = "co-adjusting"
        elif abs(first) < abs(second):
            leader, follower = market_1, market_2
        else:
            leader, follower = market_2, market_1
        shares = _gg_shares(alpha)
        alternate, alternate_ranks = {}, {}
        for delta in (-1, 1):
            alt_lag = max(lag + delta, 0)
            alt_rank = int(_johansen(values, alt_lag)["rank_95"])
            alternate_ranks[delta] = alt_rank
            alternate[delta] = (
                np.array([np.nan, np.nan])
                if alt_rank < rank
                else VECM(
                    values,
                    k_ar_diff=alt_lag,
                    coint_rank=rank,
                    deterministic="ci",
                )
                .fit()
                .alpha[:, 0]
            )

        def leader_for(
            values_: np.ndarray,
            market_1_: str = market_1,
            market_2_: str = market_2,
        ) -> str:
            if not np.isfinite(values_).all():
                return "undetermined_rank_change"
            if np.isclose(abs(values_[0]), abs(values_[1])):
                return "co-adjusting"
            return market_1_ if abs(values_[0]) < abs(values_[1]) else market_2_

        warning = shares[2]
        if alternate_ranks[-1] < rank or alternate_ranks[1] < rank:
            warning = ";".join(
                value for value in [warning, "rank_not_stable_lag_pm1"] if value
            )
        discovery_rows.append(
            empty
            | {
                "alpha_market_1": first,
                "alpha_market_2": second,
                "abs_alpha_market_1": abs(first),
                "abs_alpha_market_2": abs(second),
                "gg_share_market_1": shares[0],
                "gg_share_market_2": shares[1],
                "leader_by_adjustment": leader,
                "follower_by_adjustment": follower,
                "gg_warning": warning,
                "leader_stable_lag_pm1": (
                    leader_for(alternate[-1]) == leader
                    and leader_for(alternate[1]) == leader
                ),
                "alpha_m1_lag_minus1": float(alternate[-1][0]),
                "alpha_m1_lag_plus1": float(alternate[1][0]),
                "alpha_m2_lag_minus1": float(alternate[-1][1]),
                "alpha_m2_lag_plus1": float(alternate[1][1]),
            }
        )
    return pd.DataFrame(johansen_rows), pd.DataFrame(discovery_rows)


def information_shares(data: AnalysisData) -> pd.DataFrame:
    """Compute Hasbrouck information-share bounds for Kraken BTC/USD-USDT."""
    columns = ["kraken_btcusd", "kraken_btcusdt"]
    flags = data.price_ffill_flags[columns].any(axis=1)
    levels = np.log(data.prices[columns]).loc[~flags].dropna()
    values = levels.to_numpy()
    selection = select_order(values, maxlags=15, deterministic="ci")
    selected = selection.bic
    if selected is None:
        selected = selection.aic if selection.aic is not None else 2
    lag = max(1, int(selected)) - 1
    rank = min(
        int(
            np.sum(
                coint_johansen(values, det_order=0, k_ar_diff=lag).lr1
                > coint_johansen(values, det_order=0, k_ar_diff=lag).cvt[:, 1]
            )
        ),
        1,
    )
    fit = VECM(
        values,
        k_ar_diff=lag,
        coint_rank=rank,
        deterministic="ci",
    ).fit()
    alpha, covariance = fit.alpha[:, 0].astype(float), fit.sigma_u.astype(float)
    weights, results = np.array([-alpha[1], alpha[0]]), []
    for indices in ([0, 1], [1, 0]):
        factor = weights[indices] @ np.linalg.cholesky(
            covariance[np.ix_(indices, indices)]
        )
        shares = np.square(factor) / float(factor @ factor)
        original = np.zeros(2)
        original[indices] = shares
        results.append(original)
    results = np.asarray(results)
    low, high = results.min(axis=0), results.max(axis=0)
    return pd.DataFrame(
        [
            {
                "Channel": "Kraken BTC/USD vs BTC/USDT",
                "Rank": rank,
                "k_diff": lag,
                "IS_USD_lower": round(low[0], 3),
                "IS_USD_upper": round(high[0], 3),
                "IS_USD_mid": round((low[0] + high[0]) / 2, 3),
                "IS_other_lower": round(low[1], 3),
                "IS_other_upper": round(high[1], 3),
                "IS_other_mid": round((low[1] + high[1]) / 2, 3),
                "alpha_USD": round(float(alpha[0]), 5),
                "alpha_USDT": round(float(alpha[1]), 5),
                "note": "",
            }
        ]
    )


def _johansen(values: np.ndarray, lag: int) -> dict[str, float | int | bool]:
    result = coint_johansen(values, det_order=0, k_ar_diff=lag)
    trace = result.lr1.astype(float)
    critical = result.cvt[:, 1].astype(float)
    reject = trace > critical
    return {
        "trace_stat_r0": trace[0],
        "trace_stat_r1": trace[1],
        "trace_crit95_r0": critical[0],
        "trace_crit95_r1": critical[1],
        "reject_r0_95": bool(reject[0]),
        "reject_r1_95": bool(reject[1]),
        "rank_95": int(reject.sum()),
    }


def _gg_shares(alpha: np.ndarray) -> tuple[float, float, str]:
    denominator = float(alpha[1] - alpha[0])
    if np.isclose(denominator, 0):
        return np.nan, np.nan, "gg_denominator_near_zero"
    first, second = float(alpha[1] / denominator), float(-alpha[0] / denominator)
    warning = ""
    if not np.isfinite([first, second]).all():
        warning = "gg_non_finite"
    elif not (0 <= first <= 1 and 0 <= second <= 1):
        warning = "gg_non_convex_share"
    return first, second, warning


def _study_regimes(
    index: pd.DatetimeIndex,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    return [
        ("Pre-SVB", index.min(), CRISIS_START),
        ("Crisis", CRISIS_START, CRISIS_END),
        ("Post-SVB", CRISIS_END, index.max()),
    ]


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


def half_life_robustness(data: AnalysisData) -> pd.DataFrame:
    """Return the shared paper robustness frame for tables and figures."""
    regimes = {
        "Pre-SVB": (data.basis.index.min(), CRISIS_START),
        "Crisis": (CRISIS_START, CRISIS_END),
        "Post-SVB": (CRISIS_END, data.basis.index.max()),
    }
    rows = []
    for regime, (start, end) in regimes.items():
        mask = (data.basis.index >= start) & (data.basis.index < end)
        for column, label in [
            ("basis_usdc_kraken", "USDC/USD $B_t$ (Kraken)"),
            ("basis_usdt_kraken", "USDT/USD $B_t$ (Kraken)"),
        ]:
            one_minute = data.basis.loc[mask, column]
            one_minute_flags = data.basis_ffill_flags.loc[mask, column]
            frequencies = [
                ("1m", 1.0, one_minute, one_minute_flags),
                (
                    "5m",
                    5.0,
                    one_minute.resample("5min").last(),
                    one_minute_flags.astype(float)
                    .resample("5min")
                    .last()
                    .fillna(0.0)
                    .astype(bool),
                ),
            ]
            for frequency, minutes, series, flags in frequencies:
                for ff_filter, ff_mask in [("all", None), ("no_ff", flags)]:
                    estimate = estimate_half_life_from_ecm(
                        series,
                        minutes,
                        ff_mask=ff_mask,
                        min_obs=80,
                    )
                    rows.append(
                        {
                            "series": label,
                            "regime": regime,
                            "freq": frequency,
                            "ff_filter": ff_filter,
                            "rho_est": estimate["rho_est"],
                            "half_life_min": estimate["half_life_min"],
                            "n_obs": estimate["n_obs"],
                            "warning": estimate["warning"],
                        }
                    )
    return pd.DataFrame(rows)[
        [
            "series",
            "regime",
            "freq",
            "ff_filter",
            "rho_est",
            "half_life_min",
            "n_obs",
            "warning",
        ]
    ]


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
