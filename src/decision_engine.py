import numpy as np
from dataclasses import dataclass, field
from .simulation import run_simulation, SimulationResult
from .user_profile import UserProfile


SCENARIOS = ["Bull", "Bear", "Sideways"]
STRATEGY_NAMES = ["Keep Current", "Conservative", "Moderate", "Aggressive"]


@dataclass
class DecisionResult:
    decision_matrix: np.ndarray             # shape (n_strategies, n_scenarios)
    strategy_names: list[str]
    scenario_names: list[str]
    scenario_probabilities: np.ndarray

    # Criterion results: each maps strategy name -> value
    expected_value: dict[str, float]
    maximin: dict[str, float]
    minimax_regret: dict[str, float]
    hurwicz: dict[str, float]

    recommended_strategy: str
    recommended_weights: dict[str, float]
    sensitivity: dict[str, str]             # criterion -> winning strategy


def _build_strategy_weights(
    current_weights: dict[str, float],
    risk_params_by_level: dict,
) -> dict[str, dict[str, float]]:
    strategies = {
        "Keep Current": current_weights.copy(),
        "Conservative": {"gold": 0.10, "equity": 0.50, "cash": 0.40},
        "Moderate": {"gold": 0.25, "equity": 0.55, "cash": 0.20},
        "Aggressive": {"gold": 0.45, "equity": 0.45, "cash": 0.10},
    }
    return strategies


def _simulate_strategy_scenario(
    returns_data,
    weights: dict[str, float],
    capital: float,
    horizon: int,
    scenario_mu_override: float,
    scenario_sigma_override: float,
    n_sims: int = 2000,
    seed: int = 42,
) -> float:
    """Run a smaller simulation with scenario-specific parameters and return mean portfolio return."""
    rng = np.random.default_rng(seed)

    gold_w = weights.get("gold", 0)
    equity_w = weights.get("equity", 0)
    cash_w = weights.get("cash", 0)

    equity_daily_mu = 0.10 / 252
    equity_daily_sigma = 0.18 / np.sqrt(252)
    cash_daily_r = 0.045 / 365

    terminal_returns = []
    for _ in range(n_sims):
        value = capital
        for _ in range(horizon):
            gr = rng.normal(scenario_mu_override, max(scenario_sigma_override, 1e-6))
            er = rng.normal(equity_daily_mu, equity_daily_sigma)
            port_r = gold_w * gr + equity_w * er + cash_w * cash_daily_r
            value *= (1 + port_r)
        terminal_returns.append((value - capital) / capital)

    return float(np.mean(terminal_returns))


def run_decision_analysis(
    returns_data,
    user_profile: UserProfile,
    regime_params: dict,
) -> DecisionResult:
    capital = user_profile.capital
    horizon = user_profile.investment_horizon_days
    hurwicz_alpha = user_profile.risk_params["hurwicz_alpha"]

    # Define strategies
    strategies = _build_strategy_weights(user_profile.current_weights, {})
    strategy_names = list(strategies.keys())

    # Define scenario parameters
    bull = regime_params["bull"]
    bear = regime_params["bear"]
    scenario_params = {
        "Bull": {"mu": bull["mu"], "sigma": bull["sigma"]},
        "Bear": {"mu": bear["mu"], "sigma": bear["sigma"]},
        "Sideways": {
            "mu": (bull["mu"] + bear["mu"]) / 2,
            "sigma": (bull["sigma"] + bear["sigma"]) / 2,
        },
    }

    # Scenario probabilities from regime analysis
    bull_prob = regime_params["bull"]["probability"]
    bear_prob = regime_params["bear"]["probability"]
    sideways_prob = 0.15  # assumed small probability for sideways
    total = bull_prob + bear_prob + sideways_prob
    scenario_probs = np.array([bull_prob / total, bear_prob / total, sideways_prob / total])

    # Build decision matrix: expected portfolio return for each (strategy, scenario)
    n_strat = len(strategy_names)
    n_scen = len(SCENARIOS)
    matrix = np.zeros((n_strat, n_scen))

    for i, sname in enumerate(strategy_names):
        for j, scenario in enumerate(SCENARIOS):
            sp = scenario_params[scenario]
            matrix[i, j] = _simulate_strategy_scenario(
                returns_data,
                strategies[sname],
                capital,
                horizon,
                sp["mu"],
                sp["sigma"],
                n_sims=2000,
                seed=42 + i * 100 + j,
            )

    # --- Decision Criteria ---

    # 1. Expected Value
    ev = {sname: float(matrix[i] @ scenario_probs) for i, sname in enumerate(strategy_names)}

    # 2. Maximin (best worst-case)
    maximin = {sname: float(np.min(matrix[i])) for i, sname in enumerate(strategy_names)}

    # 3. Minimax Regret
    max_per_scenario = np.max(matrix, axis=0)  # best payoff per scenario
    regret_matrix = max_per_scenario - matrix
    max_regret = {sname: float(np.max(regret_matrix[i])) for i, sname in enumerate(strategy_names)}

    # 4. Hurwicz
    hurwicz = {
        sname: float(hurwicz_alpha * np.max(matrix[i]) + (1 - hurwicz_alpha) * np.min(matrix[i]))
        for i, sname in enumerate(strategy_names)
    }

    # Determine winner for each criterion
    ev_winner = max(ev, key=ev.get)
    maximin_winner = max(maximin, key=maximin.get)
    regret_winner = min(max_regret, key=max_regret.get)  # minimize max regret
    hurwicz_winner = max(hurwicz, key=hurwicz.get)

    sensitivity = {
        "Expected Value": ev_winner,
        "Maximin": maximin_winner,
        "Minimax Regret": regret_winner,
        "Hurwicz": hurwicz_winner,
    }

    # Primary recommendation: Expected Value criterion
    recommended = ev_winner
    recommended_weights = strategies[recommended]

    return DecisionResult(
        decision_matrix=matrix,
        strategy_names=strategy_names,
        scenario_names=SCENARIOS,
        scenario_probabilities=scenario_probs,
        expected_value=ev,
        maximin=maximin,
        minimax_regret=max_regret,
        hurwicz=hurwicz,
        recommended_strategy=recommended,
        recommended_weights=recommended_weights,
        sensitivity=sensitivity,
    )
