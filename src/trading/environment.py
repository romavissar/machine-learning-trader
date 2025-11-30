"""Gymnasium trading environment for DRL agents."""

from typing import Literal
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    """Trading environment with configurable action/reward schemes."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        action_type: Literal["discrete", "continuous"] = "discrete",
        reward_type: Literal["pnl", "sharpe"] = "pnl",
        initial_balance: float = 10_000.0,
        transaction_cost: float = 0.001,
        window_size: int = 1,
    ) -> None:
        super().__init__()
        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.action_type = action_type
        self.reward_type = reward_type
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size

        n_features = features.shape[1] + 2  # +position, +balance_ratio
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, n_features), dtype=np.float32
        )
        if action_type == "discrete":
            self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._reset_state()

    def _reset_state(self) -> None:
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.trade_history = []
        self.returns_history = []

    def _get_obs(self) -> np.ndarray:
        start = self.current_step - self.window_size
        window = self.features[start:self.current_step]
        portfolio_value = self.balance + self.position * self.prices[self.current_step - 1]
        extra = np.array([[self.position, self.balance / self.initial_balance]] * self.window_size, dtype=np.float32)
        return np.hstack([window, extra])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), self._get_info()

    def step(self, action):
        price = self.prices[self.current_step]
        prev_value = self.balance + self.position * self.prices[self.current_step - 1]

        # Execute action
        if self.action_type == "discrete":
            target_pos = {0: self.position, 1: 1.0, 2: -1.0}.get(action, self.position)
        else:
            target_pos = float(np.clip(action, -1.0, 1.0))

        delta = target_pos - self.position
        if abs(delta) > 1e-6:
            cost = abs(delta) * price * self.transaction_cost
            self.balance -= delta * price + cost
            self.position = target_pos
            self.trade_history.append({"step": self.current_step, "action": delta, "price": price})

        # Calculate reward
        curr_value = self.balance + self.position * price
        pnl = curr_value - prev_value
        self.returns_history.append(pnl / prev_value if prev_value > 0 else 0.0)

        if self.reward_type == "sharpe" and len(self.returns_history) > 1:
            returns = np.array(self.returns_history[-20:])
            reward = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))
        else:
            reward = float(pnl)

        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        truncated = curr_value <= 0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_info(self) -> dict:
        curr_price = self.prices[min(self.current_step, len(self.prices) - 1)]
        return {
            "position": self.position,
            "portfolio_value": self.balance + self.position * curr_price,
            "trade_history": self.trade_history.copy(),
        }

