import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass, field


ASSET_NAMES = ["gold", "equity", "cash"]

# Proxy annualized returns and volatility for equity and cash
# (since the dataset only covers gold, we use historical benchmarks)
EQUITY_ANNUAL_RETURN = 0.10   # S&P 500 long-run average
EQUITY_ANNUAL_VOL = 0.18      # S&P 500 long-run volatility
CASH_ANNUAL_RETURN = 0.045    # Current money market / T-bill rate
CASH_ANNUAL_VOL = 0.005       # Near zero


@dataclass
class OptimalAllocation:
    weights: dict[str, float]           # optimal weights
    expected_return: float              # portfolio expected return (annualized)
    expected_volatility: float          # portfolio std dev (annualized)
    sharpe_ratio: float
    efficient_frontier: list[dict] = field(default_factory=list)  # list of {return, risk, weights}


def _portfolio_stats(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> tuple[float, float]:
    port_return = weights @ mean_returns
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    return float(port_return), float(port_vol)


def _neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    ret, vol = _portfolio_stats(weights, mean_returns, cov_matrix)
    if vol < 1e-10:
        return 0.0
    return -(ret - risk_free_rate) / vol


def _portfolio_variance(weights, mean_returns, cov_matrix):
    return weights @ cov_matrix @ weights


def optimize_portfolio(
    gold_annual_return: float,
    gold_annual_vol: float,
    gold_equity_corr: float,
    risk_params: dict,
    current_weights: dict[str, float],
) -> OptimalAllocation:
    # Build expected returns vector and covariance matrix
    mean_returns = np.array([gold_annual_return, EQUITY_ANNUAL_RETURN, CASH_ANNUAL_RETURN])
    vols = np.array([gold_annual_vol, EQUITY_ANNUAL_VOL, CASH_ANNUAL_VOL])

    # Correlation matrix
    corr = np.array([
        [1.0, gold_equity_corr, 0.0],
        [gold_equity_corr, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    cov_matrix = np.diag(vols) @ corr @ np.diag(vols)

    # Constraints
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Bounds based on risk profile
    max_gold = risk_params["max_gold_weight"]
    min_cash = risk_params["min_cash_weight"]
    bounds = [
        (0.0, max_gold),          # gold
        (0.0, 1.0 - min_cash),    # equity
        (min_cash, 1.0),          # cash
    ]

    # Maximize Sharpe ratio
    risk_free = CASH_ANNUAL_RETURN
    x0 = np.array([1 / 3, 1 / 3, 1 / 3])

    result = minimize(
        _neg_sharpe,
        x0,
        args=(mean_returns, cov_matrix, risk_free),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    opt_weights = result.x
    opt_ret, opt_vol = _portfolio_stats(opt_weights, mean_returns, cov_matrix)
    opt_sharpe = (opt_ret - risk_free) / opt_vol if opt_vol > 1e-10 else 0.0

    # Compute efficient frontier
    # Find min and max achievable returns
    min_ret = min(mean_returns)
    max_ret = max(mean_returns)
    target_returns = np.linspace(min_ret * 0.8, max_ret * 1.1, 50)

    frontier = []
    for target in target_returns:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mean_returns - t},
        ]
        res = minimize(
            _portfolio_variance,
            x0,
            args=(mean_returns, cov_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000},
        )
        if res.success:
            fr, fv = _portfolio_stats(res.x, mean_returns, cov_matrix)
            frontier.append({
                "return": fr,
                "risk": fv,
                "weights": {name: float(w) for name, w in zip(ASSET_NAMES, res.x)},
            })

    # Also compute current portfolio stats for comparison
    current_w = np.array([current_weights.get(a, 0) for a in ASSET_NAMES])

    return OptimalAllocation(
        weights={name: float(w) for name, w in zip(ASSET_NAMES, opt_weights)},
        expected_return=opt_ret,
        expected_volatility=opt_vol,
        sharpe_ratio=opt_sharpe,
        efficient_frontier=frontier,
    )


def compute_portfolio_stats_for_weights(
    weights: dict[str, float],
    gold_annual_return: float,
    gold_annual_vol: float,
    gold_equity_corr: float,
) -> tuple[float, float, float]:
    mean_returns = np.array([gold_annual_return, EQUITY_ANNUAL_RETURN, CASH_ANNUAL_RETURN])
    vols = np.array([gold_annual_vol, EQUITY_ANNUAL_VOL, CASH_ANNUAL_VOL])
    corr = np.array([
        [1.0, gold_equity_corr, 0.0],
        [gold_equity_corr, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    cov_matrix = np.diag(vols) @ corr @ np.diag(vols)
    w = np.array([weights.get(a, 0) for a in ASSET_NAMES])
    ret, vol = _portfolio_stats(w, mean_returns, cov_matrix)
    sharpe = (ret - CASH_ANNUAL_RETURN) / vol if vol > 1e-10 else 0.0
    return ret, vol, sharpe
