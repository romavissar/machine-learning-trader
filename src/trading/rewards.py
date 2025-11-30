"""Reward functions for DRL trading agents."""
import numpy as np
from typing import Sequence


def sharpe_reward(returns: Sequence[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio reward. Returns 0 if insufficient variance."""
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    excess = arr - risk_free_rate
    std = np.std(excess, ddof=1)
    return float(np.mean(excess) / std) if std > 1e-10 else 0.0


def pnl_reward(entry_price: float, exit_price: float, position_size: float) -> float:
    """Calculate normalized PnL reward. Returns 0 for invalid entry price."""
    if entry_price <= 1e-10:
        return 0.0
    return position_size * (exit_price - entry_price) / entry_price


def risk_adjusted_reward(pnl: float, volatility: float, max_drawdown: float) -> float:
    """Calculate risk-adjusted reward with volatility and drawdown penalties."""
    vol_factor = 1.0 / (1.0 + max(volatility, 0.0))
    dd_factor = 1.0 - min(abs(max_drawdown), 1.0)
    return pnl * vol_factor * dd_factor


class RewardCalculator:
    """Combines reward functions based on configurable weights."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {"sharpe": 0.4, "pnl": 0.4, "risk_adjusted": 0.2}

    def compute(self, returns: Sequence[float], entry_price: float, exit_price: float,
                position_size: float, volatility: float, max_drawdown: float,
                risk_free_rate: float = 0.0) -> float:
        """Compute weighted combined reward from all components."""
        pnl = pnl_reward(entry_price, exit_price, position_size)
        components = {
            "sharpe": sharpe_reward(returns, risk_free_rate),
            "pnl": pnl,
            "risk_adjusted": risk_adjusted_reward(pnl, volatility, max_drawdown),
        }
        return sum(self.weights.get(k, 0.0) * v for k, v in components.items())

