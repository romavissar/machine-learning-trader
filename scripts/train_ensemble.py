"""Train ensemble model with proper temporal validation.

Uses 2 years of historical data and 30+ advanced features.
Trains XGBoost, LightGBM, and RandomForest as base models,
then combines them using voting, stacking, or confidence weighting.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

from src.features.advanced import create_features
from src.models.ensemble import (
    VotingEnsemble, StackingEnsemble, ConfidenceEnsemble,
    create_ensemble
)
from src.validation.temporal import WalkForwardValidator


def load_and_prepare_data():
    """Load data and create advanced features."""
    data_path = PROJECT_ROOT / "data" / "BTC_USDT_1h.csv"
    print(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()
    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Create advanced features
    print("\nCreating advanced features (30+ indicators)...")
    features = create_features(df, include_lagged=True)
    
    # Replace inf with NaN, then fill NaN with 0
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    
    # Skip warmup period (first 250 rows where indicators have insufficient data)
    warmup = min(250, len(features) // 10)
    features = features.iloc[warmup:]
    
    # Create target (next period return direction)
    returns = df['close'].pct_change().shift(-1)
    targets = np.where(returns > 0.001, 2,  # Up
              np.where(returns < -0.001, 0,  # Down
                       1))  # Hold
    
    # Align targets with features
    feature_mask = df.index.isin(features.index)
    targets = targets[feature_mask]
    
    # Remove last row (no label for it)
    features = features.iloc[:-1]
    targets = targets[:-1]
    
    print(f"\nData prepared:")
    print(f"  Features shape: {features.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Feature count: {len(features.columns)}")
    print(f"  Label distribution: down={np.sum(targets==0)}, hold={np.sum(targets==1)}, up={np.sum(targets==2)}")
    
    return features, targets


def evaluate_ensemble(ensemble, X_test, y_test, name: str):
    """Evaluate ensemble performance."""
    preds = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    print(f"\n{name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Predictions: down={np.sum(preds==0)}, hold={np.sum(preds==1)}, up={np.sum(preds==2)}")
    print(f"  Actuals:     down={np.sum(y_test==0)}, hold={np.sum(y_test==1)}, up={np.sum(y_test==2)}")
    
    return accuracy


def run_walk_forward_evaluation(ensemble_type: str, X: np.ndarray, y: np.ndarray):
    """Run walk-forward validation for an ensemble type."""
    print(f"\n{'='*60}")
    print(f"Walk-Forward Validation: {ensemble_type.upper()} Ensemble")
    print(f"{'='*60}")
    
    validator = WalkForwardValidator(
        n_splits=5,
        train_size=0.6,
        test_size=0.1,
        gap=1,
        expanding=True,
    )
    
    scores = []
    for i, split in enumerate(validator.split(len(X))):
        X_train = X[split.train_slice]
        y_train = y[split.train_slice]
        X_test = X[split.test_slice]
        y_test = y[split.test_slice]
        
        if len(X_train) < 500 or len(X_test) < 50:
            continue
        
        # Create and train ensemble
        ensemble = create_ensemble(ensemble_type)
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        preds = ensemble.predict(X_test)
        score = accuracy_score(y_test, preds)
        scores.append(score)
        
        print(f"  Split {i+1}: Train={len(X_train):,}, Test={len(X_test):,}, Accuracy={score:.4f}")
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"\n  📊 Walk-Forward Accuracy: {mean_score:.4f} ± {std_score:.4f}")
    
    return mean_score


def main():
    print("=" * 60)
    print("Ensemble Model Training")
    print("Using 2 years of data and 30+ advanced features")
    print("=" * 60)
    
    # Load data
    features, targets = load_and_prepare_data()
    X = features.values
    y = targets
    feature_names = features.columns.tolist()
    
    # Temporal split (70% train, 30% test)
    train_ratio = 0.70
    split_idx = int(len(X) * train_ratio)
    gap = 1
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx + gap:], y[split_idx + gap:]
    
    print(f"\n📊 TEMPORAL SPLIT:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples:     {len(X_test):,}")
    print(f"  Training period:  {features.index[0]} to {features.index[split_idx-1]}")
    print(f"  Test period:      {features.index[split_idx+gap]} to {features.index[-1]}")
    
    # Walk-forward validation for each ensemble type
    ensemble_types = ['voting', 'stacking', 'confidence']
    wf_scores = {}
    
    for etype in ensemble_types:
        wf_scores[etype] = run_walk_forward_evaluation(etype, X, y)
    
    # Select best ensemble type
    best_type = max(wf_scores, key=wf_scores.get)
    print(f"\n🏆 Best ensemble type: {best_type.upper()} (WF accuracy: {wf_scores[best_type]:.4f})")
    
    # Train final ensemble on full training data
    print(f"\n{'='*60}")
    print(f"Training Final {best_type.upper()} Ensemble")
    print(f"{'='*60}")
    
    final_ensemble = create_ensemble(best_type)
    final_ensemble.fit(X_train, y_train)
    
    # Final evaluation
    train_acc = accuracy_score(y_train, final_ensemble.predict(X_train))
    test_acc = accuracy_score(y_test, final_ensemble.predict(X_test))
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    
    # Individual member performance
    print(f"\n📊 Individual Member Performance (on test set):")
    for member in final_ensemble.members if hasattr(final_ensemble, 'members') else final_ensemble.base_models:
        member_preds = member.predict(X_test)
        member_acc = accuracy_score(y_test, member_preds)
        print(f"  {member.name}: {member_acc:.4f}")
    
    # Save ensemble
    model_path = PROJECT_ROOT / "models" / "ensemble_btc.joblib"
    final_ensemble.save(str(model_path))
    print(f"\n💾 Ensemble saved to {model_path}")
    
    # Save metadata
    metadata = {
        'ensemble_type': best_type,
        'walk_forward_scores': wf_scores,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'feature_count': len(feature_names),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'train_period': f"{features.index[0]} to {features.index[split_idx-1]}",
        'test_period': f"{features.index[split_idx+gap]} to {features.index[-1]}",
    }
    
    metadata_path = PROJECT_ROOT / "models" / "ensemble_metadata.joblib"
    joblib.dump(metadata, metadata_path)
    print(f"📋 Metadata saved to {metadata_path}")
    
    # Compare all approaches
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\nWalk-Forward Validation Scores:")
    for etype, score in sorted(wf_scores.items(), key=lambda x: -x[1]):
        marker = "🏆" if etype == best_type else "  "
        print(f"  {marker} {etype.upper():12} {score:.4f}")
    
    print(f"\n✅ Ensemble training complete!")
    print(f"   Best model: {best_type.upper()}")
    print(f"   Out-of-sample accuracy: {test_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

