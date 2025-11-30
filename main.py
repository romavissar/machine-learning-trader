# src/main.py
import asyncio
import yaml
import pandas as pd
import numpy as np

from src.data.ingestion import DataIngestion
from src.data.preprocessing import Preprocessor
from src.features.technical import TechnicalFeatures
from src.features.sentiment import SentimentFeatures
from src.models.prediction.xgboost_model import PriceDirectionModel
from src.trading.portfolio import PortfolioOptimizer
from src.trading.execution import OrderExecutor
from src.risk.manager import RiskManager
from src.pipeline import TradingPipeline, PipelineConfig

def load_config():
    with open("config/strategy.yaml") as f:
        return yaml.safe_load(f)

async def main():
    config = load_config()
    
    # Initialize components
    ingestion = DataIngestion(default_timeframe=config["strategy"]["timeframe"])
    preprocessor = Preprocessor()
    sentiment = SentimentFeatures()
    predictor = PriceDirectionModel()
    optimizer = PortfolioOptimizer()
    risk_manager = RiskManager()
    
    # For live trading, you need an exchange client
    # client = ccxt.kucoin({'apiKey': '...', 'secret': '...'})
    # executor = OrderExecutor(client)
    
    pipeline_config = PipelineConfig(
        symbol="BTC/USDT",
        timeframe=config["strategy"]["timeframe"]
    )
    
    # For backtesting (example):
    print("Loading historical data for backtest...")
    # You need to provide historical data here
    # historical_data = pd.read_csv("data/btc_usdt_hourly.csv")
    # results = pipeline.backtest(historical_data)
    
    print("Pipeline initialized successfully!")

if __name__ == "__main__":
    asyncio.run(main())