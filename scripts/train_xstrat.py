"""Train XStrat V1 model with temporal validation.

Uses proper walk-forward split to prevent lookahead bias:
- Train: 70% (PAST)
- Validation: 15%
- Test: 15% (FUTURE)
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from datetime import datetime

from src.models.xstrat_v1 import XStratV1
from src.features.technical import TechnicalFeatures


def load_data(path: Path) -> pd.DataFrame:
    """Load and prepare OHLCV data."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()
    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df


def create_features(df: pd.DataFrame) -> tuple:
    """Create features and targets."""
    tech = TechnicalFeatures(df)
    features = tech.all_features()
    
    # Add custom features
    features['return_1'] = df['close'].pct_change().fillna(0)
    features['return_5'] = df['close'].pct_change(5).fillna(0)
    features['sma200'] = df['close'].rolling(200).mean()
    features['price_vs_sma200'] = (df['close'] - features['sma200']) / features['sma200']
    
    # Get indicators for regime detection
    adx = tech.adx()
    atr = tech.atr()
    volatility = tech.historical_volatility()
    
    # Target: next period direction
    returns = df['close'].pct_change().shift(-1)
    targets = np.where(returns > 0.001, 2, np.where(returns < -0.001, 0, 1))
    
    # Skip warmup period
    warmup = 200
    features = features.iloc[warmup:-1]
    prices = df['close'].values[warmup:-1]
    adx = adx.iloc[warmup:-1].values
    atr = atr.iloc[warmup:-1].values
    volatility = volatility.iloc[warmup:-1].values
    targets = targets[warmup:-1]
    timestamps = df.index[warmup:-1]
    
    # Clean NaN/Inf
    X = np.nan_to_num(features.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    adx = np.nan_to_num(adx.astype(np.float32), nan=15.0)
    atr = np.nan_to_num(atr.astype(np.float32), nan=0.0)
    volatility = np.nan_to_num(volatility.astype(np.float32), nan=0.02)
    
    print(f"Features: {X.shape}, Targets: {np.bincount(targets)}")
    return X, targets, prices, adx, atr, volatility, timestamps


def main():
    print("=" * 60)
    print("XStrat V1 Training (Temporal Validation)")
    print("=" * 60)
    
    # Load data
    data_path = PROJECT_ROOT / "data" / "BTC_USDT_1h.csv"
    if not data_path.exists():
        print(f"[ERROR] Data not found: {data_path}")
        return
    
    df = load_data(data_path)
    X, y, prices, adx, atr, vol, timestamps = create_features(df)
    
    # Temporal split
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"\n[SPLIT] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train
    print("\n[TRAIN] Training XStrat V1...")
    strat = XStratV1(str(PROJECT_ROOT / "config" / "xstrat_v1.yaml"))
    strat.fit(X_train, y_train)
    
    # Validate
    val_preds = strat.model.predict(X_val)
    val_acc = (val_preds == y_val).mean()
    print(f"[VAL] Accuracy: {val_acc:.2%}")
    
    # Test
    test_preds = strat.model.predict(X_test)
    test_acc = (test_preds == y_test).mean()
    print(f"[TEST] Accuracy: {test_acc:.2%}")
    
    # Save
    output_path = PROJECT_ROOT / "models" / "xstrat_v1.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strat.save(str(output_path))
    
    print("\n" + "=" * 60)
    print("[OK] Training complete!")
    print("=" * 60)
    
    return strat


if __name__ == "__main__":
    main()

