"""Run backtest with trained XGBoost model to validate the full pipeline."""
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.models.prediction.xgboost_model import PriceDirectionModel
from src.risk.manager import RiskManager


def create_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Create features from OHLCV data (same as training)."""
    features = pd.DataFrame(index=df.index)
    
    # Price returns at different horizons
    features['return_1'] = df['close'].pct_change()
    features['return_5'] = df['close'].pct_change(5)
    features['return_20'] = df['close'].pct_change(lookback)
    
    # Volatility
    features['volatility'] = df['close'].pct_change().rolling(lookback).std()
    
    # Volume ratio (current vs moving average)
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(lookback).mean()
    
    # Price range
    features['high_low_range'] = (df['high'] - df['low']) / df['close']
    
    # Moving average crossover signal
    features['ma_ratio'] = df['close'] / df['close'].rolling(lookback).mean()
    
    return features


def main():
    print("=" * 60)
    print("MLT Pipeline Backtest")
    print("=" * 60)
    
    # Load data
    data_path = PROJECT_ROOT / "data" / "BTC_USDT_1h.csv"
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    print(f"Loaded {len(df)} candles")
    
    # Load trained model
    model_path = PROJECT_ROOT / "models" / "xgboost_btc.joblib"
    print(f"\nLoading model from {model_path}...")
    model = PriceDirectionModel.load(str(model_path))
    print("Model loaded successfully")
    
    # Create features (same as training)
    print("\nCreating features...")
    features = create_features(df)
    features = features.dropna()
    prices = df.loc[features.index, 'close']
    
    print(f"Features shape: {features.shape}")
    
    # Initialize risk manager
    risk_manager = RiskManager()
    max_position = 1.0
    
    # Run backtest
    print("\nRunning backtest simulation...")
    
    # Get predictions
    signals = model.predict(features.values)
    
    # Convert signals to positions (0=down->short, 1=hold->neutral, 2=up->long)
    positions = []
    for sig in signals:
        if sig == 2:  # Up
            pos = 1.0
        elif sig == 0:  # Down
            pos = -1.0
        else:  # Hold
            pos = 0.0
        
        # Apply risk check
        if risk_manager.check_position_limit(pos, max_position):
            positions.append(pos)
        else:
            positions.append(0.0)
    
    positions = np.array(positions)
    
    # Calculate returns
    returns = prices.pct_change().values
    
    # Calculate PnL (position * next period return, shifted)
    pnl = positions[:-1] * returns[1:]
    
    # Build results DataFrame
    results = pd.DataFrame({
        'signal': signals[:-1],
        'position': positions[:-1],
        'price': prices.values[:-1],
        'return': returns[1:],
        'pnl': pnl,
    }, index=features.index[:-1])
    
    # Calculate metrics
    total_pnl = results['pnl'].sum()
    avg_return = results['pnl'].mean()
    sharpe = avg_return / (results['pnl'].std() + 1e-10) * np.sqrt(252 * 24)
    
    # Calculate max drawdown
    cumulative_pnl = results['pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()
    
    # Count trades (position changes)
    trade_count = (results['position'].diff().abs() > 0).sum()
    
    # Win rate
    winning_trades = (results['pnl'] > 0).sum()
    total_trades_with_position = (results['position'] != 0).sum()
    win_rate = winning_trades / total_trades_with_position if total_trades_with_position > 0 else 0
    
    # Signal distribution
    signal_counts = results['signal'].value_counts().sort_index()
    
    # Calculate total return assuming $10,000 initial capital
    initial_capital = 10_000
    final_capital = initial_capital * (1 + cumulative_pnl.iloc[-1])
    total_return_pct = (final_capital - initial_capital) / initial_capital * 100
    
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"\nPeriod: {results.index[0]} to {results.index[-1]}")
    print(f"Total candles: {len(results)}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Initial Capital:  ${initial_capital:,.2f}")
    print(f"  Final Capital:    ${final_capital:,.2f}")
    print(f"  Total Return:     {total_return_pct:.2f}%")
    print(f"  Cumulative PnL:   {cumulative_pnl.iloc[-1]:.4f}")
    print(f"  Sharpe Ratio:     {sharpe:.4f}")
    print(f"  Max Drawdown:     {max_drawdown:.4f}")
    print(f"  Trade Count:      {trade_count}")
    print(f"  Win Rate:         {win_rate:.2%}")
    
    print(f"\nSignal Distribution:")
    for sig, count in signal_counts.items():
        label = {0: "Down (Short)", 1: "Hold", 2: "Up (Long)"}.get(sig, str(sig))
        print(f"  {label}: {count} ({count/len(results)*100:.1f}%)")
    
    print(f"\nPosition Summary:")
    pos_counts = results['position'].value_counts().sort_index()
    for pos, count in pos_counts.items():
        label = {-1.0: "Short", 0.0: "Neutral", 1.0: "Long"}.get(pos, str(pos))
        print(f"  {label}: {count} ({count/len(results)*100:.1f}%)")
    
    # Save results
    results_path = PROJECT_ROOT / "data" / "backtest_results.csv"
    results.to_csv(results_path)
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
