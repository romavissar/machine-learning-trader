"""PPO trading agent using Stable-Baselines3."""

from pathlib import Path
from typing import Sequence

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from src.trading.environment import TradingEnv


class PPOTrader:
    """Thin wrapper around SB3 PPO for trading."""

    def __init__(
        self,
        env: TradingEnv,
        hidden_layers: Sequence[int] = (64, 64),
        learning_rate: float = 3e-4,
        **ppo_kwargs,
    ) -> None:
        policy_kwargs = {"net_arch": list(hidden_layers)}
        self.model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            verbose=1,
            **ppo_kwargs,
        )

    def train(self, total_timesteps: int, eval_env: TradingEnv | None = None) -> None:
        callbacks = []
        if eval_env:
            callbacks.append(EvalCallback(eval_env, eval_freq=1000, verbose=1))
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks or None)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str | Path) -> None:
        self.model.save(path)

    def load(self, path: str | Path, env: TradingEnv | None = None) -> None:
        self.model = PPO.load(path, env=env)

