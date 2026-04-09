import numpy as np
from dataclasses import dataclass


TRANSACTION_COST_PCT = 0.0005  # 0.05% bid-ask spread for gold futures
RISK_FREE_RATE_ANNUAL = 0.045  # ~4.5% annual risk-free rate (approximate)


@dataclass
class RebalancingAnalysis:
    transaction_cost_usd: float
    expected_incremental_return_annual: float
    expected_incremental_return_usd: float
    npv_of_rebalancing: float
    break_even_days: int
    opportunity_cost_usd: float
    worth_rebalancing: bool
    explanation: str


def compute_rebalancing_economics(
    capital: float,
    current_weights: dict[str, float],
    optimal_weights: dict[str, float],
    expected_returns: dict[str, float],
    investment_horizon_days: int,
) -> RebalancingAnalysis:
    # Transaction cost: proportional to the total weight change in gold and equity
    # (cash rebalancing is free)
    total_turnover = sum(
        abs(optimal_weights.get(asset, 0) - current_weights.get(asset, 0))
        for asset in ["gold", "equity"]
    )
    transaction_cost = capital * total_turnover * TRANSACTION_COST_PCT

    # Expected portfolio returns (annualized)
    current_exp_return = sum(
        current_weights.get(a, 0) * expected_returns.get(a, 0)
        for a in expected_returns
    )
    optimal_exp_return = sum(
        optimal_weights.get(a, 0) * expected_returns.get(a, 0)
        for a in expected_returns
    )
    incremental_return_annual = optimal_exp_return - current_exp_return

    # Scale to investment horizon
    horizon_fraction = investment_horizon_days / 365.0
    incremental_return_usd = capital * incremental_return_annual * horizon_fraction

    # NPV: discount incremental return at risk-free rate, subtract transaction cost
    discount_factor = 1 / (1 + RISK_FREE_RATE_ANNUAL * horizon_fraction)
    npv = incremental_return_usd * discount_factor - transaction_cost

    # Break-even: days until daily incremental return covers transaction cost
    daily_incremental = capital * incremental_return_annual / 365.0
    if daily_incremental > 0:
        break_even = int(np.ceil(transaction_cost / daily_incremental))
    else:
        break_even = 9999  # never breaks even

    # Opportunity cost of NOT rebalancing
    opportunity_cost = max(0, incremental_return_usd)

    worth_it = npv > 0 and break_even < investment_horizon_days

    if worth_it:
        explanation = (
            f"Rebalancing is recommended. The expected incremental gain of "
            f"${incremental_return_usd:,.2f} over {investment_horizon_days} days "
            f"exceeds the transaction cost of ${transaction_cost:,.2f}. "
            f"NPV = ${npv:,.2f}. Break-even in {break_even} days."
        )
    elif incremental_return_annual <= 0:
        explanation = (
            f"Rebalancing is not recommended. The optimal portfolio does not offer "
            f"a higher expected return than your current allocation."
        )
    else:
        explanation = (
            f"Rebalancing is not recommended for your {investment_horizon_days}-day horizon. "
            f"The transaction cost of ${transaction_cost:,.2f} is too high relative to "
            f"the expected incremental gain of ${incremental_return_usd:,.2f}. "
            f"Break-even would take {break_even} days."
        )

    return RebalancingAnalysis(
        transaction_cost_usd=transaction_cost,
        expected_incremental_return_annual=incremental_return_annual,
        expected_incremental_return_usd=incremental_return_usd,
        npv_of_rebalancing=npv,
        break_even_days=break_even,
        opportunity_cost_usd=opportunity_cost,
        worth_rebalancing=worth_it,
        explanation=explanation,
    )
