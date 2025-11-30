import numpy as np
import pytest

from src.trading.environment import TradingEnv


@pytest.fixture
def env():
    features = np.column_stack([np.linspace(0, 1, 12)] * 3).astype(np.float32)
    prices = np.linspace(100, 111, 12).astype(np.float32)
    env = TradingEnv(features, prices, window_size=2)
    yield env
    env.close()


def test_reset_returns_valid_shapes(env):
    obs, info = env.reset()
    assert obs.shape == (2, 5)
    assert info["position"] == 0.0
    assert info["portfolio_value"] == env.initial_balance


def test_step_executes_trade(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == (2, 5)
    assert isinstance(reward, float)
    assert not terminated and not truncated
    assert info["trade_history"]

