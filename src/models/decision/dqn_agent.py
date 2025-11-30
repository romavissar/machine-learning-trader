"""DQN trading agent built on Stable-Baselines3."""
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from src.trading.environment import TradingEnv


class DQNTrader:
    def __init__(
        self,
        env: TradingEnv,
        buffer_size: int = 250_000,
        exploration_fraction: float = 0.1,
        exploration_final_eps: float = 0.02,
        learning_rate: float = 1e-3,
        **dqn_kwargs,
    ) -> None:
        self.model = DQN(
            "MlpPolicy",
            env,
            buffer_size=buffer_size,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            learning_rate=learning_rate,
            verbose=1,
            **dqn_kwargs,
        )

    def train(self, total_timesteps: int, eval_env: TradingEnv | None = None) -> None:
        callback = EvalCallback(eval_env, eval_freq=1000, verbose=1) if eval_env else None
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self.model.predict(observation, deterministic=deterministic)[0]

    def save(self, path: str | Path) -> None:
        self.model.save(path)

    def load(self, path: str | Path, env: TradingEnv | None = None) -> None:
        self.model = DQN.load(path, env=env)

