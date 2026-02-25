import numpy as np
import pandas as pd
import statsmodels.api as sm


def half_life_from_rho(rho: float, dt_minutes: float) -> float:
    """
    Exact half-life mapping for a levels AR(1):
      X_t = c + rho * X_{t-1} + eps_t

    half_life = (ln(2) * dt_minutes) / (-ln(rho))
    Valid mean-reverting domain: 0 < rho < 1.
    """
    if dt_minutes <= 0:
        raise ValueError(f"dt_minutes must be positive; got {dt_minutes}")
    if not np.isfinite(rho) or rho <= 0.0 or rho >= 1.0:
        return np.nan
    return float((np.log(2.0) * dt_minutes) / (-np.log(rho)))


def estimate_half_life_from_ecm(
    series: pd.Series,
    dt_minutes: float,
    ff_mask: pd.Series | None = None,
    min_obs: int = 100,
) -> dict:
    """
    Estimate half-life via ECM regression:
      ΔX_t = a + beta * X_{t-1} + eps_t
    which implies levels AR(1):
      X_t = c + rho * X_{t-1} + eps_t,  rho = 1 + beta

    If ff_mask is provided, any observation where X_t or X_{t-1}
    is forward-filled is removed from the regression sample.
    """
    s = series.dropna().astype(float)
    if len(s) < min_obs:
        return {
            'estimation_form': 'ECM (implied AR1)',
            'beta_est': np.nan,
            'kappa_est': np.nan,
            'rho_est': np.nan,
            'half_life_min': np.nan,
            'n_obs': len(s),
            'warning': 'insufficient_observations',
        }

    reg = pd.DataFrame({'x_t': s})
    reg['x_lag'] = reg['x_t'].shift(1)
    reg['dx_t'] = reg['x_t'] - reg['x_lag']

    if ff_mask is not None:
        ff = ff_mask.reindex(reg.index).fillna(False).astype(bool)
        reg['ff_t'] = ff
        reg['ff_lag'] = ff.shift(1, fill_value=False).astype(bool)
    else:
        reg['ff_t'] = False
        reg['ff_lag'] = False

    reg = reg.dropna(subset=['dx_t', 'x_lag'])
    reg = reg[~(reg['ff_t'] | reg['ff_lag'])]

    if len(reg) < min_obs:
        return {
            'estimation_form': 'ECM (implied AR1)',
            'beta_est': np.nan,
            'kappa_est': np.nan,
            'rho_est': np.nan,
            'half_life_min': np.nan,
            'n_obs': len(reg),
            'warning': 'insufficient_observations_after_ff_filter',
        }

    X = sm.add_constant(reg['x_lag'])
    model = sm.OLS(reg['dx_t'], X).fit()
    beta = float(model.params.iloc[1])
    kappa = float(-beta)
    rho = float(1.0 + beta)
    hl = half_life_from_rho(rho, dt_minutes=dt_minutes)

    warning = ''
    if not (0.0 < rho < 1.0):
        warning = 'rho_outside_(0,1)_half_life_undefined'

    return {
        'estimation_form': 'ECM (implied AR1)',
        'beta_est': beta,
        'kappa_est': kappa,
        'rho_est': rho,
        'half_life_min': hl,
        'n_obs': int(len(reg)),
        'warning': warning,
    }


def run_half_life_sanity_tests(dt_minutes: float = 1.0) -> pd.DataFrame:
    """
    Unit-style checks:
    - monotonicity in rho grid
    - undefined outside (0,1)
    """
    rho_grid = [0.5, 0.8, 0.9, 0.95, 0.99]
    hl_grid = [half_life_from_rho(rho, dt_minutes=dt_minutes) for rho in rho_grid]

    if not all(np.isfinite(v) for v in hl_grid):
        raise AssertionError(f"Expected finite half-lives on rho grid; got {hl_grid}")
    if not all(hl_grid[i] < hl_grid[i + 1] for i in range(len(hl_grid) - 1)):
        raise AssertionError(f"Half-life is not strictly increasing in rho: {hl_grid}")
    if hl_grid[-1] <= 10.0 * hl_grid[0]:
        raise AssertionError(
            "Half-life does not expand enough near rho -> 1; expected blow-up behavior."
        )

    invalid_rhos = [-0.2, 0.0, 1.0, 1.1]
    invalid_hl = [half_life_from_rho(rho, dt_minutes=dt_minutes) for rho in invalid_rhos]
    if not all(np.isnan(v) for v in invalid_hl):
        raise AssertionError(f"Invalid rhos should return NaN half-life; got {invalid_hl}")

    out = pd.DataFrame({
        'rho': rho_grid,
        'dt_minutes': dt_minutes,
        'half_life_min': hl_grid,
    })
    return out
