"""Trading pipeline orchestrating data -> prediction -> execution flow.

IMPORTANT: Backtesting uses walk-forward validation to prevent lookahead bias.
The model is retrained periodically using only data available at each point in time.
This ensures realistic backtest results.
"""
import asyncio
from dataclasses import dataclass
from typing import Protocol, Callable, Optional
import warnings

import numpy as np
import pandas as pd


class Predictor(Protocol):
    def predict(self, X) -> list: ...
    def fit(self, X, y) -> 'Predictor': ...


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

    def backtest(
        self,
        historical_data: pd.DataFrame,
        news_df: pd.DataFrame = None,
        train_ratio: float = 0.7,
        walk_forward: bool = True,
        retrain_frequency: int = 100,
        min_train_samples: int = 200,
    ) -> pd.DataFrame:
        """Run backtest with proper temporal validation.
        
        IMPORTANT: This backtest prevents lookahead bias by:
        1. Training only on data available at each point in time
        2. Never using future data for predictions
        3. Retraining the model periodically as new data becomes available
        
        Args:
            historical_data: OHLCV DataFrame with DatetimeIndex
            news_df: Optional news data for sentiment features
            train_ratio: Initial training set size (for first split)
            walk_forward: If True, use walk-forward validation (recommended)
                         If False, uses static train/test split (faster but less realistic)
            retrain_frequency: How often to retrain (in samples) during walk-forward
            min_train_samples: Minimum samples required before making predictions
            
        Returns:
            DataFrame with signals, positions, PnL, and metadata
        """
        features = self._build_features({"main": historical_data}, news_df)
        
        # Create target variable (next period return direction)
        close_col = next((c for c in features.columns if "close" in c.lower()), None)
        if close_col is None:
            raise ValueError("No close column found in features")
        
        returns = features[close_col].pct_change().shift(-1)  # NEXT period return
        targets = np.where(returns > 0.001, 2,  # Up
                  np.where(returns < -0.001, 0,  # Down
                           1))  # Hold
        
        # Remove last row (no target for it)
        features = features.iloc[:-1]
        targets = targets[:-1]
        returns = returns.iloc[:-1]
        
        n_samples = len(features)
        
        if not walk_forward:
            # Static split (faster but less realistic)
            warnings.warn(
                "Using static train/test split. Set walk_forward=True for more realistic results.",
                UserWarning
            )
            return self._static_backtest(features, targets, returns, train_ratio)
        
        # =======================================================================
        # WALK-FORWARD BACKTEST - Prevents lookahead bias
        # =======================================================================
        print(f"🔄 Running walk-forward backtest on {n_samples} samples...")
        print(f"   Retrain frequency: every {retrain_frequency} samples")
        print(f"   Minimum training samples: {min_train_samples}")
        
        results = []
        last_train_idx = 0
        
        # Start predicting after we have enough training data
        start_idx = max(min_train_samples, int(n_samples * train_ratio))
        
        for i in range(start_idx, n_samples):
            # Retrain if needed
            if i == start_idx or (i - last_train_idx) >= retrain_frequency:
                # Train ONLY on data up to this point (PAST data)
                X_train = features.iloc[:i].values
                y_train = targets[:i]
                
                self.predictor.fit(X_train, y_train)
                last_train_idx = i
                
            # Predict current sample (using model trained on PAST only)
            X_current = features.iloc[i:i+1].values
            signal = self.predictor.predict(X_current)[0]
            
            # Convert signal to position
            position = 1.0 if signal == 2 else (-1.0 if signal == 0 else 0.0)
            
            # Apply risk check
            if not self.risk.check_position_limit(position, self.config.max_position):
                position = 0.0
            
            # Record result
            results.append({
                'timestamp': features.index[i],
                'signal': signal,
                'position': position,
                'return': returns.iloc[i],
                'pnl': position * returns.iloc[i],
                'train_samples': i,  # How much data model was trained on
            })
        
        result_df = pd.DataFrame(results).set_index('timestamp')
        
        # Calculate performance metrics
        total_pnl = result_df['pnl'].sum()
        sharpe = result_df['pnl'].mean() / (result_df['pnl'].std() + 1e-8) * np.sqrt(252 * 24)
        
        print(f"\n📈 WALK-FORWARD BACKTEST RESULTS:")
        print(f"   Total PnL: {total_pnl:.4f}")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        print(f"   Trades: {(result_df['position'] != 0).sum()}")
        
        return result_df
    
    def _static_backtest(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        returns: pd.Series,
        train_ratio: float,
    ) -> pd.DataFrame:
        """Static train/test backtest (faster but less realistic).
        
        WARNING: This trains once on past data and tests on future.
        It's faster but doesn't simulate periodic retraining.
        """
        n = len(features)
        train_end = int(n * train_ratio)
        
        # Train on PAST only
        X_train = features.iloc[:train_end].values
        y_train = targets[:train_end]
        self.predictor.fit(X_train, y_train)
        
        # Test on FUTURE only
        X_test = features.iloc[train_end:].values
        signals = self.predictor.predict(X_test)
        
        positions = [1.0 if s == 2 else (-1.0 if s == 0 else 0.0) for s in signals]
        validated = [p if self.risk.check_position_limit(p, self.config.max_position) else 0.0 for p in positions]
        
        test_returns = returns.iloc[train_end:]
        pnl = pd.Series(validated, index=test_returns.index) * test_returns.values
        
        return pd.DataFrame({
            "signal": signals,
            "position": validated,
            "pnl": pnl.values,
        }, index=test_returns.index)

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

