import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class SimulationResult:
    terminal_values: np.ndarray           # shape (n_simulations,)
    paths: np.ndarray                      # shape (n_simulations, horizon+1)
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    probability_of_loss: float
    expected_terminal_value: float
    median_terminal_value: float
    max_drawdowns: np.ndarray              # shape (n_simulations,)
    regime_params: dict = field(default_factory=dict)


def _identify_regimes(returns: pd.Series, threshold: float = 0.0) -> dict:
    """Simple regime identification using rolling mean of returns."""
    rolling_mean = returns.rolling(30).mean()
    bull_mask = rolling_mean > threshold
    bear_mask = rolling_mean <= threshold

    bull_returns = returns[bull_mask].dropna()
    bear_returns = returns[bear_mask].dropna()

    n_total = len(rolling_mean.dropna())
    n_bull = bull_mask.sum()
    n_bear = bear_mask.sum()

    # Transition probabilities (simplified)
    regime = bull_mask.astype(int)  # 1=bull, 0=bear
    transitions = pd.Series(regime.values[1:]) != pd.Series(regime.values[:-1])
    n_transitions = transitions.sum()
    stay_prob = 1 - (n_transitions / max(len(transitions), 1))

    params = {
        "bull": {
            "mu": float(bull_returns.mean()) if len(bull_returns) > 0 else 0.0005,
            "sigma": float(bull_returns.std()) if len(bull_returns) > 0 else 0.01,
            "probability": n_bull / max(n_total, 1),
        },
        "bear": {
            "mu": float(bear_returns.mean()) if len(bear_returns) > 0 else -0.0005,
            "sigma": float(bear_returns.std()) if len(bear_returns) > 0 else 0.015,
            "probability": n_bear / max(n_total, 1),
        },
        "transition_matrix": {
            "bull_to_bull": float(stay_prob),
            "bull_to_bear": float(1 - stay_prob),
            "bear_to_bear": float(stay_prob),
            "bear_to_bull": float(1 - stay_prob),
        },
    }
    return params


def _max_drawdown(path: np.ndarray) -> float:
    peak = np.maximum.accumulate(path)
    drawdown = (peak - path) / peak
    return float(np.max(drawdown))


def run_simulation(
    returns: pd.Series,
    portfolio_weights: dict[str, float],
    capital: float,
    horizon_days: int,
    gold_annual_return: float,
    gold_annual_vol: float,
    n_simulations: int = 10000,
    random_seed: int = 42,
) -> SimulationResult:
    rng = np.random.default_rng(random_seed)

    # Identify regimes from historical gold returns
    regime_params = _identify_regimes(returns)

    bull = regime_params["bull"]
    bear = regime_params["bear"]
    trans = regime_params["transition_matrix"]

    gold_weight = portfolio_weights.get("gold", 0)
    equity_weight = portfolio_weights.get("equity", 0)
    cash_weight = portfolio_weights.get("cash", 0)

    # Equity proxy parameters (daily)
    equity_daily_mu = 0.10 / 252
    equity_daily_sigma = 0.18 / np.sqrt(252)
    cash_daily_return = 0.045 / 365

    paths = np.zeros((n_simulations, horizon_days + 1))
    paths[:, 0] = capital

    for sim in range(n_simulations):
        # Start in a random regime based on overall probability
        in_bull = rng.random() < bull["probability"]

        for t in range(1, horizon_days + 1):
            # Regime-switching: determine current regime parameters
            if in_bull:
                mu_gold = bull["mu"]
                sigma_gold = bull["sigma"]
                # Transition
                if rng.random() > trans["bull_to_bull"]:
                    in_bull = False
            else:
                mu_gold = bear["mu"]
                sigma_gold = bear["sigma"]
                if rng.random() > trans["bear_to_bear"]:
                    in_bull = True

            # Generate daily returns for each asset
            gold_daily_return = rng.normal(mu_gold, max(sigma_gold, 1e-6))
            equity_daily_return = rng.normal(equity_daily_mu, equity_daily_sigma)

            # Portfolio return (weighted sum)
            portfolio_return = (
                gold_weight * gold_daily_return
                + equity_weight * equity_daily_return
                + cash_weight * cash_daily_return
            )

            paths[sim, t] = paths[sim, t - 1] * (1 + portfolio_return)

    terminal_values = paths[:, -1]
    returns_total = (terminal_values - capital) / capital

    # VaR and CVaR (on returns)
    var_95 = float(np.percentile(returns_total, 5))
    var_99 = float(np.percentile(returns_total, 1))
    cvar_95 = float(np.mean(returns_total[returns_total <= var_95]))
    cvar_99 = float(np.mean(returns_total[returns_total <= var_99]))

    # Max drawdown per simulation
    max_drawdowns = np.array([_max_drawdown(paths[i]) for i in range(n_simulations)])

    return SimulationResult(
        terminal_values=terminal_values,
        paths=paths,
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        probability_of_loss=float(np.mean(terminal_values < capital)),
        expected_terminal_value=float(np.mean(terminal_values)),
        median_terminal_value=float(np.median(terminal_values)),
        max_drawdowns=max_drawdowns,
        regime_params=regime_params,
    )
