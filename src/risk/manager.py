"""Risk management module for MLT trading system."""
import numpy as np
from numpy.typing import ArrayLike


class RiskManager:
    """Handles position limits, stop-loss, volatility monitoring, and drawdown."""

    def check_position_limit(self, position: float, max_position: float) -> bool:
        """Return True if position is within allowed limit."""
        if max_position <= 0:
            return False  # Fail-safe: invalid limit blocks trading
        return abs(position) <= max_position

    def calculate_stop_loss(self, entry_price: float, atr: float, multiplier: float = 2.0) -> float:
        """Calculate stop-loss price using ATR-based trailing stop."""
        if entry_price <= 0 or atr < 0:
            return 0.0  # Fail-safe: return 0 on invalid inputs
        return entry_price - (atr * multiplier)

    def monitor_volatility(self, returns: ArrayLike, threshold: float) -> bool:
        """Return True (alert) if volatility exceeds threshold."""
        if threshold <= 0:
            return True  # Fail-safe: alert on invalid threshold
        arr = np.asarray(returns, dtype=np.float64)
        if len(arr) < 2:
            return False
        return float(np.std(arr, ddof=1)) > threshold

    def max_drawdown(self, equity_curve: ArrayLike) -> float:
        """Calculate maximum drawdown as a positive fraction."""
        equity = np.asarray(equity_curve, dtype=np.float64)
        if len(equity) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.where(peak > 0, peak, 1.0)
        return float(np.max(drawdown))

    def should_halt_trading(self, drawdown: float, vol: float, limits: dict) -> bool:
        """Return True if trading should halt based on risk limits."""
        max_dd = limits.get("max_drawdown", 0.0)  # Fail-safe: halt if not specified
        max_vol = limits.get("max_volatility", 0.0)
        return drawdown >= max_dd or vol >= max_vol

