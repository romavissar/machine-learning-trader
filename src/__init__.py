"""MLT - Machine Learning Trader."""
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class StrategyType(str, Enum):
    ALPHA_GENERATION = "alpha_generation"
    OPTIMAL_EXECUTION = "optimal_execution"
    ARBITRAGE = "arbitrage"


class AssetClass(str, Enum):
    CRYPTO = "crypto"
    EQUITIES = "equities"
    FOREX = "forex"
    COMMODITIES = "commodities"
    FIXED_INCOME = "fixed_income"


class Timeframe(str, Enum):
    MILLISECONDS = "milliseconds"
    SECONDS = "seconds"
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"


class DataSource(str, Enum):
    OHLCV = "ohlcv"
    ORDER_BOOK = "order_book"
    NEWS_SENTIMENT = "news_sentiment"
    TECHNICAL_INDICATORS = "technical_indicators"


class StrategyConfig(BaseModel):
    type: StrategyType
    asset_class: AssetClass
    timeframe: Timeframe


class ModelsConfig(BaseModel):
    prediction: Literal["xgboost", "lstm", "cnn", "random_forest"] = "xgboost"
    decision: Literal["ppo", "dqn", "sac"] = "ppo"
    sentiment: Literal["finbert", "opt", "gpt4"] = "finbert"
    arbitrage: Literal["graphsage"] | None = None


class MLTConfig(BaseModel):
    """Root configuration for MLT system."""
    strategy: StrategyConfig
    models: ModelsConfig
    data_sources: list[DataSource] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MLTConfig":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

