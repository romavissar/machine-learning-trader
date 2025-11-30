"""Integration tests for MLT pipeline."""
import asyncio
import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.pipeline import TradingPipeline, PipelineConfig
from src.features.technical import TechnicalFeatures
from src.trading.portfolio import PortfolioOptimizer
from src.trading.execution import OrderExecutor
from src.risk.manager import RiskManager


@pytest.fixture
def sample_ohlcv():
    ts = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame({
        "timestamp": ts, "open": close * 0.99, "high": close * 1.01,
        "low": close * 0.98, "close": close, "volume": np.random.rand(100) * 1000,
    }).set_index("timestamp")

@pytest.fixture
def processed_features(sample_ohlcv):
    df = sample_ohlcv.copy()
    df["close_main"] = df["close"]
    df["ret_close_main"] = df["close"].pct_change()
    return df.dropna()

@pytest.fixture
def mock_pipeline(sample_ohlcv, processed_features):
    ingestion = MagicMock(fetch_ohlcv=AsyncMock(return_value=sample_ohlcv.reset_index()))
    preprocessor = MagicMock(transform=MagicMock(return_value=processed_features))
    predictor = MagicMock(predict=lambda x: np.array([2] * len(x)))
    sentiment = MagicMock(aggregate_sentiment=MagicMock(return_value=pd.Series([0.5])))
    client = MagicMock(create_order=AsyncMock(return_value={"id": "123", "status": "filled"}))
    return TradingPipeline(
        ingestion, preprocessor, TechnicalFeatures, sentiment, predictor,
        PortfolioOptimizer(), RiskManager(), OrderExecutor(client), PipelineConfig()
    )

class TestPipelineIntegration:
    def test_backtest_full_flow(self, mock_pipeline, sample_ohlcv):
        result = mock_pipeline.backtest(sample_ohlcv.reset_index())
        assert {"signal", "position", "pnl"} <= set(result.columns)
        assert len(result) > 0 and result["position"].abs().max() <= 1.0

    def test_run_once_executes_order(self, mock_pipeline):
        orders = asyncio.new_event_loop().run_until_complete(mock_pipeline.run_once())
        assert len(orders) == 1 and orders[0].status == "filled"

    def test_empty_data_returns_empty(self, mock_pipeline):
        mock_pipeline.preprocessor.transform.return_value = pd.DataFrame()
        result = mock_pipeline.backtest(pd.DataFrame())
        assert result.empty

    def test_extreme_prices_in_prediction(self, mock_pipeline, processed_features):
        processed_features.loc[processed_features.index[0], "close"] = 1e12
        mock_pipeline.preprocessor.transform.return_value = processed_features
        result = mock_pipeline.backtest(pd.DataFrame())  # input ignored by mock
        assert not result.empty  # pipeline handles extremes

class TestRiskIntegration:
    def test_position_limit_enforced(self):
        rm = RiskManager()
        assert rm.check_position_limit(0.5, 1.0) and not rm.check_position_limit(1.5, 1.0)

    def test_halt_on_high_drawdown(self):
        rm = RiskManager()
        assert rm.should_halt_trading(0.25, 0.01, {"max_drawdown": 0.2, "max_volatility": 0.5})

    def test_drawdown_calculation(self):
        rm = RiskManager()
        assert abs(rm.max_drawdown([100, 80, 90, 70]) - 0.30) < 0.01

class TestExecutionIntegration:
    def test_market_order_success(self):
        client = MagicMock(create_order=AsyncMock(return_value={"id": "X", "status": "ok"}))
        result = asyncio.new_event_loop().run_until_complete(
            OrderExecutor(client).market_order("BTC/USDT", "buy", 0.1))
        assert result.order_id == "X" and result.status == "ok"

    def test_order_failure_handled(self):
        client = MagicMock(create_order=AsyncMock(side_effect=Exception("API down")))
        result = asyncio.new_event_loop().run_until_complete(
            OrderExecutor(client).market_order("BTC/USDT", "buy", 0.1))
        assert "error" in result.status
