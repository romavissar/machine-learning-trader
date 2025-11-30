"""Train the Model Orchestrator with walk-forward validation.

IMPORTANT: Uses proper temporal splits to prevent lookahead bias.
- Training data: First 70% of historical data (the PAST)
- Validation data: Next 15% (for hyperparameter tuning)
- Test data: Last 15% (the FUTURE - never seen during training)

The orchestrator combines XGBoost, LSTM, and Ensemble models,
dynamically selecting based on market regime.
"""
import sys
from pathlib import Path
import warnings

# Suppress sklearn feature name warnings (we use numpy arrays consistently)
warnings.filterwarnings('ignore', message='.*feature names.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from tqdm import tqdm

from src.models.orchestrator import ModelOrchestrator, OrchestratorConfig
from src.models.prediction.xgboost_model import PriceDirectionModel
from src.models.prediction.lstm_model import LSTMPredictor
from src.features.technical import TechnicalFeatures
from src.validation.temporal import split_temporal, WalkForwardValidator


def load_and_prepare_data(data_path: Path) -> tuple:
    """Load data and create features with no lookahead bias.
    
    Returns:
        Tuple of (features_df, prices, high, low, targets)
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()
    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Extract OHLCV
    prices = df['close'].values.astype(np.float32)
    high = df['high'].values.astype(np.float32)
    low = df['low'].values.astype(np.float32)
    volume = df['volume'].values.astype(np.float32)
    
    # Create technical features
    print("Creating technical features...")
    tech = TechnicalFeatures(df)
    features_df = tech.all_features()
    
    # Add price-based features
    features_df['close'] = prices
    features_df['return_1'] = df['close'].pct_change().fillna(0)
    features_df['return_5'] = df['close'].pct_change(5).fillna(0)
    features_df['volume_change'] = df['volume'].pct_change().fillna(0).clip(-10, 10)
    features_df['range'] = ((df['high'] - df['low']) / df['close']).fillna(0)
    features_df['momentum'] = (df['close'] / df['close'].shift(10) - 1).fillna(0)
    
    # Create target: next period return direction
    returns = df['close'].pct_change().shift(-1)  # NEXT period return
    targets = np.where(returns > 0.001, 2,  # Up
              np.where(returns < -0.001, 0,  # Down
                       1))  # Hold
    
    # Drop NaN rows (from rolling features)
    warmup = 50
    features_df = features_df.iloc[warmup:-1]  # -1 because last target is NaN
    prices = prices[warmup:-1]
    high = high[warmup:-1]
    low = low[warmup:-1]
    targets = targets[warmup:-1]
    
    print(f"Features shape: {features_df.shape}")
    print(f"Target distribution: {np.bincount(targets)}")
    
    return features_df, prices, high, low, targets


def create_lstm_sequences(features: np.ndarray, seq_len: int = 20) -> np.ndarray:
    """Create sequence data for LSTM.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        seq_len: Sequence length
        
    Returns:
        Sequence array (n_samples, seq_len, n_features)
    """
    n_samples, n_features = features.shape
    sequences = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
    
    for i in range(n_samples):
        start_idx = max(0, i - seq_len + 1)
        actual_len = i - start_idx + 1
        sequences[i, seq_len - actual_len:] = features[start_idx:i + 1]
    
    return sequences


def train_orchestrator(
    data_path: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    embargo_gap: int = 6,  # hours
):
    """Train the model orchestrator with temporal validation.
    
    Args:
        data_path: Path to OHLCV CSV
        output_dir: Directory to save trained model
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        embargo_gap: Gap between train and test sets (in samples/hours)
    """
    print("=" * 60)
    print("Model Orchestrator Training (with Temporal Validation)")
    print("=" * 60)
    print("\n[!] IMPORTANT: Using proper temporal train/test split")
    print("    - Model trains on PAST data only")
    print("    - Model is tested on FUTURE data only")
    print("    - No lookahead bias / data leakage")
    
    # Load and prepare data
    features_df, prices, high, low, targets = load_and_prepare_data(data_path)
    
    # Convert to numpy
    features = features_df.values.astype(np.float32)
    
    # Handle NaN/Inf in features
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    n_samples = len(features)
    
    # ==========================================================================
    # TEMPORAL SPLIT - Critical for preventing lookahead bias
    # ==========================================================================
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    test_start = val_end + embargo_gap
    
    # Training data: PAST only
    X_train = features[:train_end]
    y_train = targets[:train_end]
    prices_train = prices[:train_end]
    high_train = high[:train_end]
    low_train = low[:train_end]
    
    # Validation data
    X_val = features[train_end + embargo_gap:val_end]
    y_val = targets[train_end + embargo_gap:val_end]
    
    # Test data: FUTURE only
    X_test = features[test_start:]
    y_test = targets[test_start:]
    prices_test = prices[test_start:]
    high_test = high[test_start:]
    low_test = low[test_start:]
    
    print(f"\n[DATA] TEMPORAL SPLIT:")
    print(f"   Training samples:   {len(X_train):,} (indices 0 to {train_end})")
    print(f"   Validation samples: {len(X_val):,} (indices {train_end + embargo_gap} to {val_end})")
    print(f"   Test samples:       {len(X_test):,} (indices {test_start} to {n_samples})")
    print(f"   Embargo gap:        {embargo_gap} samples between sets")
    
    # Verify no overlap
    if train_end >= test_start:
        raise ValueError("CRITICAL ERROR: Training and test sets overlap!")
    
    # ==========================================================================
    # Create and configure orchestrator
    # ==========================================================================
    print("\n[CONFIG] Configuring Model Orchestrator...")
    
    config = OrchestratorConfig(
        adx_trending_threshold=25.0,
        adx_ranging_threshold=20.0,
        volatility_high_multiplier=2.0,
        single_model_confidence=0.75,
        ensemble_min_agreement=0.67,
        max_position=1.0,
        max_drawdown=0.15,
    )
    
    # Initialize LSTM
    n_features = X_train.shape[1]
    lstm_model = LSTMPredictor(
        input_dim=n_features,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        output_dim=3,  # 3 classes: down, hold, up
    )
    
    orchestrator = ModelOrchestrator(
        config=config,
        lstm_model=lstm_model,
    )
    
    # ==========================================================================
    # Train on PAST data only
    # ==========================================================================
    print("\n[TRAIN] Training on PAST data only...")
    
    # Create LSTM sequences for training
    X_train_seq = create_lstm_sequences(X_train, seq_len=20)
    
    # Train orchestrator (trains XGBoost, Ensemble, and LSTM)
    orchestrator.fit(X_train, y_train, X_seq=X_train_seq)
    
    # ==========================================================================
    # Quick validation using XGBoost directly (faster than full orchestrator)
    # ==========================================================================
    print("\n[VALIDATE] Quick validation on hold-out set...")
    
    # Use XGBoost predictions directly for speed
    val_preds = orchestrator.xgboost.predict(X_val)
    val_correct = (val_preds == y_val).sum()
    val_accuracy = val_correct / len(X_val) if len(X_val) > 0 else 0
    print(f"   Validation accuracy (XGBoost): {val_accuracy:.2%}")
    
    # ==========================================================================
    # Test on FUTURE data only
    # ==========================================================================
    print("\n[TEST] Testing on FUTURE data (never seen during training)...")
    
    test_results = []
    test_correct = 0
    
    # Sample subset for regime-aware testing (full test takes too long)
    test_sample_size = min(500, len(X_test))
    sample_indices = np.linspace(0, len(X_test) - 1, test_sample_size, dtype=int)
    
    for idx in tqdm(sample_indices, desc="Testing"):
        i = int(idx)
        # Use historical prices up to this point for regime detection
        if i < 50:
            price_history = np.concatenate([prices_train[-(50-i):], prices_test[:i]]) if i > 0 else prices_train[-50:]
            high_history = np.concatenate([high_train[-(50-i):], high_test[:i]]) if i > 0 else high_train[-50:]
            low_history = np.concatenate([low_train[-(50-i):], low_test[:i]]) if i > 0 else low_train[-50:]
        else:
            price_history = prices_test[max(0, i-50):i]
            high_history = high_test[max(0, i-50):i]
            low_history = low_test[max(0, i-50):i]
        
        result = orchestrator.generate_signal(
            X=X_test[i:i + 1],
            prices=price_history,
            high=high_history,
            low=low_history,
        )
        
        test_results.append({
            'signal': result.signal,
            'confidence': result.confidence,
            'regime': result.regime,
            'selected_models': result.selected_models,
            'actual': y_test[i],
            'position': result.risk_adjusted_position,
        })
        
        if result.signal == y_test[i]:
            test_correct += 1
    
    test_accuracy = test_correct / len(test_results) if len(test_results) > 0 else 0
    
    # Analyze regime distribution
    regimes = [r['regime'] for r in test_results]
    regime_counts = pd.Series(regimes).value_counts()
    
    print(f"\n[RESULTS] OUT-OF-SAMPLE TEST RESULTS:")
    print(f"   Test accuracy:        {test_accuracy:.2%}")
    print(f"   Test samples (sampled): {len(test_results)} of {len(X_test)}")
    print(f"   Regime distribution:")
    for regime, count in regime_counts.items():
        print(f"      {regime}: {count} ({count/len(test_results):.1%})")
    
    # Calculate simple backtest return (on sampled data)
    # Get prices at sampled indices
    sampled_prices = prices_test[sample_indices]
    if len(sampled_prices) > 1:
        sampled_returns = np.diff(sampled_prices) / sampled_prices[:-1]
        positions = [r['position'] for r in test_results[:-1]]  # One less than prices
        strategy_returns = np.array(positions) * sampled_returns
        total_return = (1 + strategy_returns).prod() - 1
    else:
        total_return = 0.0
    
    print(f"\n   Strategy return (sampled):  {total_return:.2%}")
    print(f"   Buy-and-hold return:        {(prices_test[-1]/prices_test[0] - 1):.2%}")
    
    # ==========================================================================
    # Save model
    # ==========================================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "orchestrator.joblib"
    orchestrator.save(str(model_path))
    
    # Save training metadata
    metadata = {
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'train_accuracy': 'N/A (not evaluated on train set)',
        'validation_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'strategy_return': total_return,
        'regime_distribution': regime_counts.to_dict(),
        'trained_at': datetime.now().isoformat(),
        'config': {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'embargo_gap': embargo_gap,
        }
    }
    
    metadata_path = output_dir / "training_metadata.joblib"
    joblib.dump(metadata, metadata_path)
    
    print(f"\n[SAVE] Model saved to {model_path}")
    print(f"[SAVE] Metadata saved to {metadata_path}")
    
    print("\n" + "=" * 60)
    print("[OK] Training complete with proper temporal validation!")
    print("   Model was trained on PAST data and tested on FUTURE data.")
    print("=" * 60)
    
    return orchestrator, metadata


def main():
    """Main entry point for training."""
    data_path = PROJECT_ROOT / "data" / "BTC_USDT_1h.csv"
    output_dir = PROJECT_ROOT / "models"
    
    if not data_path.exists():
        print(f"[ERROR] Data file not found: {data_path}")
        print("   Please run data/scripts/download_data.py first")
        return
    
    orchestrator, metadata = train_orchestrator(
        data_path=data_path,
        output_dir=output_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        embargo_gap=6,
    )
    
    return orchestrator


if __name__ == "__main__":
    main()

