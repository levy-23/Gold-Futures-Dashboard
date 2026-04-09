from pydantic import BaseModel, model_validator
from enum import Enum


class RiskTolerance(str, Enum):
    CONSERVATIVE = "Conservative"
    MODERATE = "Moderate"
    AGGRESSIVE = "Aggressive"


class UserProfile(BaseModel):
    capital: float
    current_gold_pct: float
    current_equity_pct: float
    current_cash_pct: float
    risk_tolerance: RiskTolerance
    investment_horizon_days: int

    @model_validator(mode="after")
    def validate_allocations(self):
        total = self.current_gold_pct + self.current_equity_pct + self.current_cash_pct
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Allocations must sum to 100%, got {total:.2f}%")
        if self.capital <= 0:
            raise ValueError("Capital must be positive")
        if self.investment_horizon_days not in (30, 90, 180, 365):
            raise ValueError("Horizon must be 30, 90, 180, or 365 days")
        return self

    @property
    def current_weights(self) -> dict[str, float]:
        return {
            "gold": self.current_gold_pct / 100.0,
            "equity": self.current_equity_pct / 100.0,
            "cash": self.current_cash_pct / 100.0,
        }

    @property
    def risk_params(self) -> dict:
        params = {
            RiskTolerance.CONSERVATIVE: {
                "target_return_percentile": 0.25,
                "max_drawdown_tolerance": 0.05,
                "hurwicz_alpha": 0.3,
                "min_cash_weight": 0.20,
                "max_gold_weight": 0.30,
            },
            RiskTolerance.MODERATE: {
                "target_return_percentile": 0.50,
                "max_drawdown_tolerance": 0.10,
                "hurwicz_alpha": 0.5,
                "min_cash_weight": 0.10,
                "max_gold_weight": 0.50,
            },
            RiskTolerance.AGGRESSIVE: {
                "target_return_percentile": 0.75,
                "max_drawdown_tolerance": 0.20,
                "hurwicz_alpha": 0.7,
                "min_cash_weight": 0.05,
                "max_gold_weight": 0.70,
            },
        }
        return params[self.risk_tolerance]
