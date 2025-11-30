"""Hyperparameter optimization using Optuna with walk-forward validation.

Optimizes XGBoost and LightGBM hyperparameters using temporal cross-validation
to prevent overfitting and lookahead bias.

Usage:
    python scripts/optimize_hyperparams.py --model xgboost --trials 50
    python scripts/optimize_hyperparams.py --model lightgbm --trials 50
    python scripts/optimize_hyperparams.py --model all --trials 30
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import joblib
import json
from datetime import datetime

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.features.advanced import create_features
from src.validation.temporal import WalkForwardValidator


def load_data():
    """Load and prepare data with advanced features."""
    data_path = PROJECT_ROOT / "data" / "BTC_USDT_1h.csv"
    print(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()
    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Create advanced features
    print("Creating advanced features...")
    features = create_features(df, include_lagged=True)
    
    # Create target (next period return direction)
    returns = df['close'].pct_change().shift(-1)
    targets = np.where(returns > 0.001, 2,  # Up
              np.where(returns < -0.001, 0,  # Down
                       1))  # Hold
    
    # Align features and targets
    targets = targets[:len(features)]
    
    # Remove last row and NaN rows
    valid_mask = ~features.isna().any(axis=1)
    features = features[valid_mask].iloc[:-1]
    targets = targets[valid_mask.values][:-1]
    
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    
    return features, targets


def create_xgboost_objective(X: np.ndarray, y: np.ndarray, validator: WalkForwardValidator):
    """Create Optuna objective function for XGBoost."""
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'verbosity': 0,
        }
        
        scores = []
        for split in validator.split(len(X)):
            X_train = X[split.train_slice]
            y_train = y[split.train_slice]
            X_test = X[split.test_slice]
            y_test = y[split.test_slice]
            
            if len(X_train) < 100 or len(X_test) < 10:
                continue
            
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    return objective


def create_lightgbm_objective(X: np.ndarray, y: np.ndarray, validator: WalkForwardValidator):
    """Create Optuna objective function for LightGBM."""
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'objective': 'multiclass',
            'num_class': 3,
            'verbosity': -1,
        }
        
        scores = []
        for split in validator.split(len(X)):
            X_train = X[split.train_slice]
            y_train = y[split.train_slice]
            X_test = X[split.test_slice]
            y_test = y[split.test_slice]
            
            if len(X_train) < 100 or len(X_test) < 10:
                continue
            
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    return objective


def optimize_model(model_type: str, X: np.ndarray, y: np.ndarray, n_trials: int = 50):
    """Run hyperparameter optimization for a model."""
    print(f"\n{'='*60}")
    print(f"Optimizing {model_type.upper()} hyperparameters")
    print(f"{'='*60}")
    
    # Create walk-forward validator
    validator = WalkForwardValidator(
        n_splits=5,
        train_size=0.6,
        test_size=0.1,
        gap=1,
        expanding=True,
    )
    
    # Create study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    # Create objective function
    if model_type == 'xgboost':
        objective = create_xgboost_objective(X, y, validator)
    elif model_type == 'lightgbm':
        objective = create_lightgbm_objective(X, y, validator)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Optimize
    print(f"\nRunning {n_trials} optimization trials...")
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[lambda study, trial: print(f"  Trial {trial.number}: {trial.value:.4f}") if trial.number % 10 == 0 else None]
    )
    
    # Results
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION RESULTS for {model_type.upper()}")
    print(f"{'='*60}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params, study.best_value


def train_final_model(model_type: str, params: dict, X: np.ndarray, y: np.ndarray):
    """Train final model with optimized hyperparameters."""
    print(f"\nTraining final {model_type} model with optimized params...")
    
    # Use 80% for training
    train_end = int(len(X) * 0.8)
    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[train_end:], y[train_end:]
    
    if model_type == 'xgboost':
        params.update({
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
        })
        model = XGBClassifier(**params)
    else:
        params.update({
            'objective': 'multiclass',
            'num_class': 3,
            'verbosity': -1,
        })
        model = LGBMClassifier(**params)
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    
    return model, {'train_acc': train_acc, 'test_acc': test_acc}


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization')
    parser.add_argument('--model', type=str, default='all', choices=['xgboost', 'lightgbm', 'all'])
    parser.add_argument('--trials', type=int, default=50)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hyperparameter Optimization with Walk-Forward Validation")
    print("=" * 60)
    
    # Load data
    features, targets = load_data()
    X = features.values
    y = targets
    
    # Install dependencies if needed
    try:
        import optuna
        import lightgbm
    except ImportError:
        print("\nInstalling required packages...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'optuna', 'lightgbm', '-q'])
    
    # Models to optimize
    models = ['xgboost', 'lightgbm'] if args.model == 'all' else [args.model]
    
    results = {}
    for model_type in models:
        # Optimize
        best_params, best_score = optimize_model(model_type, X, y, args.trials)
        
        # Train final model
        model, metrics = train_final_model(model_type, best_params.copy(), X, y)
        
        # Save results
        results[model_type] = {
            'best_params': best_params,
            'cv_score': best_score,
            'metrics': metrics,
        }
        
        # Save model
        model_path = PROJECT_ROOT / "models" / f"{model_type}_optimized.joblib"
        joblib.dump({
            'model': model,
            'params': best_params,
            'metrics': metrics,
            'feature_names': features.columns.tolist(),
        }, model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save best params as JSON
        params_path = PROJECT_ROOT / "config" / f"{model_type}_best_params.json"
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Best params saved to {params_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    for model_type, result in results.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Walk-forward CV accuracy: {result['cv_score']:.4f}")
        print(f"  Final test accuracy:      {result['metrics']['test_acc']:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Optimization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

