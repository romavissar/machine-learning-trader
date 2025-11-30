"""MLT - Machine Learning Trader entry point.

All APIs and platforms used are:
✓ FREE for paper trading (no paid API keys required)
✓ Available in Romania (EU/MiCA compliant or globally available)
✓ Open-source ML models (Hugging Face - no paid services)

Exchanges (all support Romania):
- Binance: MiCA compliant, testnet.binance.vision (free)
- Bybit: Global, testnet.bybit.com (free)  
- Kraken: EU regulated, fully available in Romania
- OKX: Global, sandbox mode (free)
- Bitget: Global, demo mode (free)

ML Models (all FREE, open-source):
- FinBERT: Financial sentiment analysis
- XGBoost/LSTM: Price prediction
- PPO/DQN: Reinforcement learning agents
"""
import asyncio
import yaml

from src.data.ingestion import DataIngestion
from src.data.preprocessing import Preprocessor
from src.features.technical import TechnicalFeatures
from src.features.sentiment import SentimentFeatures
from src.models.prediction.xgboost_model import PriceDirectionModel
from src.trading.portfolio import PortfolioOptimizer
from src.trading.paper import PaperExecutor
from src.risk.manager import RiskManager
from src.pipeline import TradingPipeline, PipelineConfig

PAPER_TRADING = True  # Set False for live trading (requires API keys)
INITIAL_CAPITAL = 10_000.0
EXCHANGE = "binance"  # Options: binance, bybit, kraken, okx, bitget (all support Romania)

def load_config():
    with open("config/strategy.yaml") as f:
        return yaml.safe_load(f)

async def main():
    config = load_config()
    symbol = "BTC/USDT"
    
    # Initialize paper executor with Romania-friendly exchange (FREE, no API key needed)
    executor = PaperExecutor(initial_cash=INITIAL_CAPITAL, exchange=EXCHANGE)
    
    print(f"{'📝 PAPER' if PAPER_TRADING else '🔴 LIVE'} TRADING MODE")
    print(f"Symbol: {symbol} | Capital: ${INITIAL_CAPITAL:,.2f} | Exchange: {EXCHANGE}")
    print("✓ All exchanges support Romania | ✓ FREE for paper trading")
    
    # Demo: execute a paper trade
    print("\n--- Executing paper trade ---")
    result = await executor.market_order(symbol, "buy", 0.01)
    print(f"Order: {result.side} {result.amount} {symbol} @ ${result.price:,.2f} | Status: {result.status}")
    print(f"Portfolio: {executor.summary()}")
    
    # Sell it back
    result = await executor.market_order(symbol, "sell", 0.01)
    print(f"Order: {result.side} {result.amount} {symbol} @ ${result.price:,.2f} | Status: {result.status}")
    print(f"Portfolio: {executor.summary()}")
    
    await executor.close()
    print("\n✅ Paper trading infrastructure ready!")

if __name__ == "__main__":
    asyncio.run(main())
