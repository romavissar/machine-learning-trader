"""Train PPO trading agent using Stable-Baselines3.

IMPORTANT: This script uses TEMPORAL train/test splits to prevent lookahead bias.
- Training data: First 70% of historical data (the PAST)
- Validation data: Next 15% (for hyperparameter tuning, optional)
- Test data: Last 15% (the FUTURE - never seen during training)

The model is trained ONLY on past data and evaluated on future data,
simulating realistic trading conditions.
"""
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Tuple
from src.trading.environment import TradingEnv
from src.models.decision.ppo_agent import PPOTrader
from src.validation.temporal import split_temporal, validate_no_lookahead
from src.features.advanced import create_features as create_advanced_features


def create_features(df: pd.DataFrame, use_advanced: bool = True) -> Tuple[np.ndarray, int]:
    """Create feature matrix for the trading environment.
    
    Args:
        df: OHLCV DataFrame
        use_advanced: If True, use 30+ advanced features
        
    Returns:
        Tuple of (features array, warmup period to skip)
        
    IMPORTANT: All features are calculated using ONLY past data.
    No feature should peek into the future.
    """
    if use_advanced:
        # Use advanced features (30+ indicators)
        features_df = create_advanced_features(df, include_lagged=False)  # No lag for RL
        # Replace inf and fill NaN
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        # Clip extreme values
        features_df = features_df.clip(-100, 100)
        # Return warmup period (skip first 250 rows)
        warmup = min(250, len(features_df) // 10)
        return features_df.values.astype(np.float32), warmup
    
    # Basic features (fallback)
    features = pd.DataFrame(index=df.index)
    features['return_1'] = df['close'].pct_change().fillna(0)
    features['return_5'] = df['close'].pct_change(5).fillna(0)
    features['volatility'] = df['close'].pct_change().rolling(20).std().fillna(0)
    features['volume_change'] = df['volume'].pct_change().fillna(0).clip(-10, 10)
    features['range'] = ((df['high'] - df['low']) / df['close']).fillna(0)
    features['momentum'] = (df['close'] / df['close'].shift(10) - 1).fillna(0)
    
    return features.values.astype(np.float32), 20


def main():
    print("=" * 60)
    print("PPO Trading Agent Training (with Temporal Validation)")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: Using proper temporal train/test split")
    print("   - Model trains on PAST data only")
    print("   - Model is tested on FUTURE data only")
    print("   - No lookahead bias / data leakage")
    
    # Load data
    data_path = PROJECT_ROOT / "data" / "BTC_USDT_1h.csv"
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()  # Ensure chronological order
    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Create features
    print("\nCreating features...")
    features, warmup = create_features(df)
    prices = df['close'].values.astype(np.float32)
    
    # Skip warmup period where features have insufficient data
    print(f"  Using {features.shape[1]} features, skipping {warmup} warmup rows")
    features = features[warmup:]
    prices = prices[warmup:]
    
    # ==========================================================================
    # TEMPORAL SPLIT - Critical for preventing lookahead bias
    # ==========================================================================
    # Split ratios: 70% train, 15% validation (unused here), 15% test
    train_ratio = 0.70
    test_ratio = 0.15
    gap = 1  # 1-hour embargo between train and test
    
    n_samples = len(features)
    train_end = int(n_samples * train_ratio)
    test_start = int(n_samples * (1 - test_ratio))
    
    # Training data: PAST only
    train_features = features[:train_end]
    train_prices = prices[:train_end]
    
    # Test data: FUTURE only (with gap to prevent any leakage)
    test_features = features[test_start:]
    test_prices = prices[test_start:]
    
    print(f"\n📊 TEMPORAL SPLIT:")
    print(f"   Training samples:   {len(train_features):,} (indices 0 to {train_end})")
    print(f"   Test samples:       {len(test_features):,} (indices {test_start} to {n_samples})")
    print(f"   Embargo gap:        {test_start - train_end} samples between train and test")
    print(f"\n   Training period:    {df.index[warmup]} to {df.index[warmup + train_end - 1]}")
    print(f"   Test period:        {df.index[warmup + test_start]} to {df.index[-1]}")
    
    # Verify no overlap
    if train_end >= test_start:
        raise ValueError("CRITICAL ERROR: Training and test sets overlap! This would cause lookahead bias.")
    
    # ==========================================================================
    # Train on PAST data only
    # ==========================================================================
    print("\n🎯 Training on PAST data only...")
    
    train_env = TradingEnv(
        features=train_features,
        prices=train_prices,
        action_type="discrete",
        reward_type="pnl",
        initial_balance=10_000,
        transaction_cost=0.001,
        window_size=1,
    )
    
    trader = PPOTrader(
        env=train_env,
        hidden_layers=(64, 64),
        learning_rate=3e-4,
    )
    
    total_timesteps = 20_000
    print(f"Training for {total_timesteps} timesteps on historical data...")
    print("(Model will NEVER see test data during training)\n")
    
    trader.train(total_timesteps=total_timesteps)
    
    # ==========================================================================
    # Test on FUTURE data only
    # ==========================================================================
    print("\n🔮 Testing on FUTURE data (never seen during training)...")
    
    test_env = TradingEnv(
        features=test_features,
        prices=test_prices,
        action_type="discrete",
        reward_type="pnl",
        initial_balance=10_000,
        transaction_cost=0.001,
        window_size=1,
    )
    
    obs, info = test_env.reset()
    total_reward = 0
    steps = 0
    
    while True:
        action = trader.predict(obs)
        action = int(np.asarray(action).flatten()[0])
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    
    print(f"\n📈 OUT-OF-SAMPLE TEST RESULTS (on data model never saw):")
    print(f"   Test period steps:    {steps}")
    print(f"   Total reward:         {total_reward:.2f}")
    print(f"   Final portfolio:      ${info['portfolio_value']:.2f}")
    print(f"   Total trades:         {len(info['trade_history'])}")
    
    initial_balance = 10_000
    final_value = info['portfolio_value']
    total_return = (final_value - initial_balance) / initial_balance * 100
    print(f"   Out-of-sample return: {total_return:.2f}%")
    
    # Compare to buy-and-hold benchmark
    buy_hold_return = (test_prices[-1] - test_prices[0]) / test_prices[0] * 100
    print(f"\n📊 BENCHMARKS:")
    print(f"   Buy-and-hold return:  {buy_hold_return:.2f}%")
    print(f"   Strategy alpha:       {total_return - buy_hold_return:.2f}%")
    
    # Save model
    model_path = PROJECT_ROOT / "models" / "ppo_btc"
    trader.save(str(model_path))
    print(f"\n💾 Model saved to {model_path}.zip")
    
    # Save training metadata
    metadata = {
        "training_samples": len(train_features),
        "test_samples": len(test_features),
        "train_period": f"{df.index[warmup]} to {df.index[warmup + train_end - 1]}",
        "test_period": f"{df.index[warmup + test_start]} to {df.index[-1]}",
        "out_of_sample_return": total_return,
        "buy_hold_return": buy_hold_return,
    }
    print(f"\n📋 Training metadata: {metadata}")
    
    print("\n" + "=" * 60)
    print("✅ Training complete with proper temporal validation!")
    print("   Model was trained on PAST data and tested on FUTURE data.")
    print("   This simulates realistic trading conditions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
