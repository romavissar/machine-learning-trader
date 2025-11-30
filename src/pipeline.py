"""Trading pipeline orchestrating data -> prediction -> execution flow."""
import asyncio
from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class Predictor(Protocol):
    def predict(self, X) -> list: ...


class Optimizer(Protocol):
    def mean_variance_optimize(self, returns, cov) -> list: ...


class RiskChecker(Protocol):
    def check_position_limit(self, pos: float, max_pos: float) -> bool: ...


@dataclass
class PipelineConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "hourly"
    max_position: float = 1.0


class TradingPipeline:
    def __init__(
        self,
        ingestion,
        preprocessor,
        technical_features_cls,
        sentiment_features,
        predictor: Predictor,
        optimizer: Optimizer,
        risk_manager: RiskChecker,
        executor,
        config: PipelineConfig = None,
    ):
        self.ingestion = ingestion
        self.preprocessor = preprocessor
        self.technical_cls = technical_features_cls
        self.sentiment = sentiment_features
        self.predictor = predictor
        self.optimizer = optimizer
        self.risk = risk_manager
        self.executor = executor
        self.config = config or PipelineConfig()

    async def run_once(self, news_df: pd.DataFrame = None) -> list:
        ohlcv = await self.ingestion.fetch_ohlcv(self.config.symbol, self.config.timeframe)
        features = self._build_features({"main": ohlcv}, news_df)
        signals = self.predictor.predict(features.values[-1:])
        positions = self._optimize_positions(features, signals)
        validated = [p for p in positions if self.risk.check_position_limit(p, self.config.max_position)]
        if validated:
            side = "buy" if validated[0] > 0 else "sell"
            return [await self.executor.market_order(self.config.symbol, side, abs(validated[0]))]
        return []

    async def run_loop(self, interval: float, news_df: pd.DataFrame = None):
        while True:
            await self.run_once(news_df)
            await asyncio.sleep(interval)

    def backtest(self, historical_data: pd.DataFrame, news_df: pd.DataFrame = None) -> pd.DataFrame:
        features = self._build_features({"main": historical_data}, news_df)
        signals = self.predictor.predict(features.values)
        positions = [1.0 if s == 2 else (-1.0 if s == 0 else 0.0) for s in signals]
        validated = [p if self.risk.check_position_limit(p, self.config.max_position) else 0.0 for p in positions]
        returns = features.filter(like="close").pct_change().mean(axis=1).iloc[-len(validated):]
        pnl = pd.Series(validated, index=returns.index) * returns.values
        return pd.DataFrame({"signal": signals, "position": validated, "pnl": pnl.values}, index=returns.index)

    def _build_features(self, frames: dict, news_df: pd.DataFrame = None) -> pd.DataFrame:
        processed = self.preprocessor.transform(frames)
        close_col = next((c for c in processed.columns if "close" in c.lower()), None)
        if close_col:
            tech_df = pd.DataFrame({"close": processed[close_col], "volume": processed.get(close_col.replace("close", "volume"), 0)})
            tech = self.technical_cls(tech_df)
            processed["rsi"], processed["macd"] = tech.rsi(), tech.macd()
        if news_df is not None and not news_df.empty:
            sentiment = self.sentiment.aggregate_sentiment("1h").reindex(processed.index, method="ffill").fillna(0)
            processed["sentiment"] = sentiment
        return processed.dropna()

    def _optimize_positions(self, features: pd.DataFrame, signals) -> list:
        returns = features.filter(like="ret_").mean()
        cov = features.filter(like="ret_").cov()
        if cov.empty or returns.empty:
            return [float(signals[0]) - 1.0]
        weights = self.optimizer.mean_variance_optimize(returns.values, cov.values)
        return [w * (1 if s == 2 else -1 if s == 0 else 0) for w, s in zip(weights, signals)]

