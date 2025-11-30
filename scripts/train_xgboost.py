"""Train XGBoost price direction model on historical data.

IMPORTANT: This script uses TEMPORAL train/test splits to prevent lookahead bias.
- Training data: First 70% of historical data (the PAST)
- Test data: Last 30% (the FUTURE - never seen during training)
- Optional: Walk-forward validation with multiple retraining periods

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
from src.models.prediction.xgboost_model import PriceDirectionModel
from src.validation.temporal import WalkForwardValidator, split_temporal
from src.features.advanced import create_features as create_advanced_features


def create_features(df: pd.DataFrame, use_advanced: bool = True) -> pd.DataFrame:
    """Create features from OHLCV data.
    
    Args:
        df: OHLCV DataFrame
        use_advanced: If True, use 30+ advanced features. If False, use basic features.
    
    IMPORTANT: All features are calculated using ONLY past data.
    """
    if use_advanced:
        # Use advanced feature engineering (30+ features)
        features = create_advanced_features(df, include_lagged=True)
        # Replace inf with NaN, then fill NaN with 0 (needed for indicators that need warmup)
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        # Skip first 250 rows where most indicators have insufficient data
        warmup = min(250, len(features) // 10)
        return features.iloc[warmup:]
    
    # Basic features (fallback)
    features = pd.DataFrame(index=df.index)
    features['return_1'] = df['close'].pct_change()
    features['return_5'] = df['close'].pct_change(5)
    features['return_20'] = df['close'].pct_change(20)
    features['volatility'] = df['close'].pct_change().rolling(20).std()
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['high_low_range'] = (df['high'] - df['low']) / df['close']
    features['ma_ratio'] = df['close'] / df['close'].rolling(20).mean()
    
    return features.dropna()


def create_labels(df: pd.DataFrame, threshold: float = 0.001) -> np.ndarray:
    """Create labels: 0=down, 1=hold, 2=up based on NEXT candle's return.
    
    Note: shift(-1) is used here because we're predicting the FUTURE.
    This is correct - the label IS the future we're trying to predict.
    The key is that FEATURES must not contain this future information.
    """
    returns = df['close'].pct_change().shift(-1)  # NEXT period return (what we predict)
    labels = np.where(returns > threshold, 2,      # Up
              np.where(returns < -threshold, 0,    # Down
                       1))                          # Hold
    return labels


def run_walk_forward_validation(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5):
    """Run walk-forward validation to get robust out-of-sample performance.
    
    This retrains the model multiple times, each time using only data
    available up to that point in time.
    """
    print("\n" + "=" * 60)
    print("📊 WALK-FORWARD VALIDATION")
    print("=" * 60)
    
    validator = WalkForwardValidator(
        n_splits=n_splits,
        train_size=0.6,  # Start with 60% for first training
        test_size=0.1,   # Test on 10% at a time
        gap=1,           # 1-period embargo
        expanding=True,  # Training set grows (more realistic)
    )
    
    all_predictions = []
    all_actuals = []
    split_results = []
    
    for i, split in enumerate(validator.split(len(X))):
        # Training data: ONLY PAST
        X_train = X.iloc[split.train_slice].values
        y_train = y[split.train_slice]
        
        # Test data: FUTURE (relative to training)
        X_test = X.iloc[split.test_slice].values
        y_test = y[split.test_slice]
        
        # Train fresh model
        model = PriceDirectionModel()
        model.fit(X_train, y_train)
        
        # Predict on future data
        preds = model.predict(X_test)
        accuracy = (preds == y_test).mean()
        
        all_predictions.extend(preds)
        all_actuals.extend(y_test)
        
        split_results.append({
            'split': i + 1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': accuracy,
        })
        
        print(f"   Split {i+1}: Train={len(X_train):,}, Test={len(X_test):,}, Accuracy={accuracy:.2%}")
    
    # Overall out-of-sample accuracy
    overall_accuracy = np.mean(np.array(all_predictions) == np.array(all_actuals))
    print(f"\n   📈 Overall out-of-sample accuracy: {overall_accuracy:.2%}")
    
    return split_results, overall_accuracy


def main():
    print("=" * 60)
    print("XGBoost Price Direction Model Training")
    print("(with Temporal Validation - No Lookahead Bias)")
    print("=" * 60)
    
    print("\n⚠️  IMPORTANT: Using proper temporal train/test split")
    print("   - Features calculated from PAST data only")
    print("   - Labels are FUTURE returns (what we predict)")
    print("   - Model trained on PAST, tested on FUTURE")
    
    # Load data
    data_path = PROJECT_ROOT / "data" / "BTC_USDT_1h.csv"
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()  # Ensure chronological order
    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Create features and labels
    print("\nCreating features (from PAST data only)...")
    X = create_features(df)
    y_full = create_labels(df)
    
    print(f"  Advanced features created: {X.shape[1]} features")
    
    # Align labels with feature indices
    # Get the positions of feature indices in the original df
    feature_mask = df.index.isin(X.index)
    y = y_full[feature_mask]
    
    # Remove last row (no label for it - can't know future of last candle)
    X = X.iloc[:-1]
    y = y[:-1]
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: down={np.sum(y==0)}, hold={np.sum(y==1)}, up={np.sum(y==2)}")
    
    # ==========================================================================
    # TEMPORAL SPLIT - Critical for preventing lookahead bias
    # ==========================================================================
    train_ratio = 0.70
    gap = 1  # 1-period embargo
    
    split_idx = int(len(X) * train_ratio)
    test_start = split_idx + gap
    
    # Training data: PAST only
    X_train, y_train = X.iloc[:split_idx], y[:split_idx]
    
    # Test data: FUTURE only
    X_test, y_test = X.iloc[test_start:], y[test_start:]
    
    print(f"\n📊 TEMPORAL SPLIT:")
    print(f"   Training samples:   {len(X_train):,} (first {train_ratio:.0%} of data)")
    print(f"   Test samples:       {len(X_test):,} (last {1-train_ratio:.0%} of data)")
    print(f"   Embargo gap:        {gap} samples between train and test")
    print(f"\n   Training period:    {X_train.index[0]} to {X_train.index[-1]}")
    print(f"   Test period:        {X_test.index[0]} to {X_test.index[-1]}")
    
    # Verify no data leakage
    if X_train.index.max() >= X_test.index.min():
        raise ValueError("CRITICAL: Training data overlaps with test data! Lookahead bias detected.")
    
    # ==========================================================================
    # Train on PAST data only
    # ==========================================================================
    print("\n🎯 Training on PAST data only...")
    model = PriceDirectionModel()
    model.fit(X_train.values, y_train)
    
    # ==========================================================================
    # Evaluate on FUTURE data only
    # ==========================================================================
    print("\n🔮 Evaluating on FUTURE data (never seen during training)...")
    
    train_preds = model.predict(X_train.values)
    test_preds = model.predict(X_test.values)
    
    train_acc = (train_preds == y_train).mean()
    test_acc = (test_preds == y_test).mean()
    
    print(f"\n📈 RESULTS:")
    print(f"   Train accuracy (in-sample):     {train_acc:.2%}")
    print(f"   Test accuracy (out-of-sample):  {test_acc:.2%}")
    
    # Check for overfitting
    overfit_gap = train_acc - test_acc
    if overfit_gap > 0.1:
        print(f"\n   ⚠️  Warning: Large gap between train/test ({overfit_gap:.2%})")
        print(f"      This may indicate overfitting. Consider regularization.")
    
    # Show prediction distribution
    print(f"\n   Test predictions: down={np.sum(test_preds==0)}, hold={np.sum(test_preds==1)}, up={np.sum(test_preds==2)}")
    print(f"   Test actuals:     down={np.sum(y_test==0)}, hold={np.sum(y_test==1)}, up={np.sum(y_test==2)}")
    
    # ==========================================================================
    # Optional: Walk-Forward Validation
    # ==========================================================================
    run_walk_forward = True
    if run_walk_forward:
        run_walk_forward_validation(X, y, n_splits=5)
    
    # ==========================================================================
    # Save model with metadata
    # ==========================================================================
    model_path = PROJECT_ROOT / "models" / "xgboost_btc.joblib"
    model.save(str(model_path))
    print(f"\n💾 Model saved to {model_path}")
    
    # Log training info
    print(f"\n📋 Training metadata:")
    print(f"   - Model trained on data up to: {X_train.index[-1]}")
    print(f"   - Out-of-sample accuracy: {test_acc:.2%}")
    print(f"   - DO NOT use this model on data before: {X_test.index[0]}")
    
    print("\n" + "=" * 60)
    print("✅ Training complete with proper temporal validation!")
    print("   Model was trained on PAST data and tested on FUTURE data.")
    print("   This simulates realistic trading conditions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
