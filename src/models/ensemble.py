"""Ensemble model combining multiple base models for robust predictions.

Architecture:
    ┌─────────────┐
    │  XGBoost    │──┐
    └─────────────┘  │
    ┌─────────────┐  │    ┌──────────────┐    ┌──────────┐
    │    LSTM     │──┼───>│  Meta-Model  │───>│  Signal  │
    └─────────────┘  │    │  (Stacking)  │    └──────────┘
    ┌─────────────┐  │    └──────────────┘
    │  LightGBM   │──┘
    └─────────────┘

Ensemble methods:
- VotingEnsemble: Simple majority/weighted voting
- StackingEnsemble: Train meta-model on base model predictions
- ConfidenceEnsemble: Weight by model confidence/probability
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import joblib

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@dataclass
class ModelPrediction:
    """Container for model predictions with confidence."""
    model_name: str
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None


class BaseModel:
    """Base class for ensemble members."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class XGBoostMember(BaseModel):
    """XGBoost ensemble member."""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("xgboost")
        default_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'verbosity': 0,
        }
        if params:
            default_params.update(params)
        self.model = XGBClassifier(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostMember':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class LightGBMMember(BaseModel):
    """LightGBM ensemble member."""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("lightgbm")
        default_params = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multiclass',
            'num_class': 3,
            'verbosity': -1,
        }
        if params:
            default_params.update(params)
        self.model = LGBMClassifier(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LightGBMMember':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class RandomForestMember(BaseModel):
    """Random Forest ensemble member for diversity."""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("random_forest")
        default_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'n_jobs': -1,
        }
        if params:
            default_params.update(params)
        self.model = RandomForestClassifier(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestMember':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class VotingEnsemble:
    """Ensemble using voting strategy (hard or soft voting)."""
    
    def __init__(
        self,
        members: Optional[List[BaseModel]] = None,
        voting: str = 'soft',
        weights: Optional[List[float]] = None,
    ):
        """Initialize voting ensemble.
        
        Args:
            members: List of base models (default: XGBoost, LightGBM, RandomForest)
            voting: 'hard' for majority vote, 'soft' for probability averaging
            weights: Optional weights for each model
        """
        self.members = members or [
            XGBoostMember(),
            LightGBMMember(),
            RandomForestMember(),
        ]
        self.voting = voting
        self.weights = weights or [1.0] * len(self.members)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VotingEnsemble':
        """Train all base models."""
        print(f"Training {len(self.members)} ensemble members...")
        for i, member in enumerate(self.members):
            print(f"  [{i+1}/{len(self.members)}] Training {member.name}...")
            member.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using voting."""
        if self.voting == 'soft':
            return self._soft_vote(X)
        else:
            return self._hard_vote(X)
    
    def _hard_vote(self, X: np.ndarray) -> np.ndarray:
        """Majority voting."""
        predictions = np.array([m.predict(X) for m in self.members])
        # Weighted voting
        votes = np.zeros((len(X), 3))
        for i, (preds, weight) in enumerate(zip(predictions, self.weights)):
            for j, pred in enumerate(preds):
                votes[j, int(pred)] += weight
        return np.argmax(votes, axis=1)
    
    def _soft_vote(self, X: np.ndarray) -> np.ndarray:
        """Probability-weighted voting."""
        probas = np.array([m.predict_proba(X) for m in self.members])
        # Weight and average probabilities
        weighted_probas = np.zeros_like(probas[0])
        for proba, weight in zip(probas, self.weights):
            weighted_probas += proba * weight
        weighted_probas /= sum(self.weights)
        return np.argmax(weighted_probas, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get averaged probabilities."""
        probas = np.array([m.predict_proba(X) for m in self.members])
        weighted_probas = np.zeros_like(probas[0])
        for proba, weight in zip(probas, self.weights):
            weighted_probas += proba * weight
        weighted_probas /= sum(self.weights)
        return weighted_probas
    
    def save(self, path: str):
        """Save ensemble to file."""
        joblib.dump({
            'members': self.members,
            'voting': self.voting,
            'weights': self.weights,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'VotingEnsemble':
        """Load ensemble from file."""
        data = joblib.load(path)
        ensemble = cls(
            members=data['members'],
            voting=data['voting'],
            weights=data['weights'],
        )
        ensemble.is_fitted = True
        return ensemble


class StackingEnsemble:
    """Ensemble using stacking (meta-learning) strategy.
    
    Base models' predictions become features for a meta-model.
    """
    
    def __init__(
        self,
        base_models: Optional[List[BaseModel]] = None,
        meta_model: Optional[Any] = None,
        use_probas: bool = True,
        passthrough: bool = False,
    ):
        """Initialize stacking ensemble.
        
        Args:
            base_models: List of base models
            meta_model: Meta-learner (default: LogisticRegression)
            use_probas: Use probabilities instead of predictions as meta-features
            passthrough: Include original features in meta-model input
        """
        self.base_models = base_models or [
            XGBoostMember(),
            LightGBMMember(),
            RandomForestMember(),
        ]
        self.meta_model = meta_model or LogisticRegression(max_iter=1000)
        self.use_probas = use_probas
        self.passthrough = passthrough
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """Train base models and meta-model.
        
        Uses temporal split to prevent data leakage:
        - First 70% trains base models
        - Last 30% generates meta-features and trains meta-model
        """
        # Temporal split
        split_idx = int(len(X) * 0.7)
        X_base, y_base = X[:split_idx], y[:split_idx]
        X_meta, y_meta = X[split_idx:], y[split_idx:]
        
        # Train base models on first portion
        print(f"Training {len(self.base_models)} base models...")
        for i, model in enumerate(self.base_models):
            print(f"  [{i+1}/{len(self.base_models)}] Training {model.name}...")
            model.fit(X_base, y_base)
        
        # Generate meta-features from second portion
        print("Generating meta-features...")
        meta_features = self._generate_meta_features(X_meta)
        
        # Train meta-model
        print("Training meta-model...")
        self.meta_model.fit(meta_features, y_meta)
        
        # Retrain base models on full data for final predictions
        print("Retraining base models on full data...")
        for model in self.base_models:
            model.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base model predictions."""
        if self.use_probas:
            # Use class probabilities
            features = [m.predict_proba(X) for m in self.base_models]
            meta_features = np.hstack(features)
        else:
            # Use predictions
            features = [m.predict(X).reshape(-1, 1) for m in self.base_models]
            meta_features = np.hstack(features)
        
        if self.passthrough:
            meta_features = np.hstack([meta_features, X])
        
        return meta_features
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using stacked model."""
        meta_features = self._generate_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        meta_features = self._generate_meta_features(X)
        return self.meta_model.predict_proba(meta_features)
    
    def save(self, path: str):
        """Save ensemble to file."""
        joblib.dump({
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'use_probas': self.use_probas,
            'passthrough': self.passthrough,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'StackingEnsemble':
        """Load ensemble from file."""
        data = joblib.load(path)
        ensemble = cls(
            base_models=data['base_models'],
            meta_model=data['meta_model'],
            use_probas=data['use_probas'],
            passthrough=data['passthrough'],
        )
        ensemble.is_fitted = True
        return ensemble


class ConfidenceEnsemble:
    """Ensemble that weights predictions by model confidence.
    
    Models with higher prediction confidence get more weight.
    """
    
    def __init__(
        self,
        members: Optional[List[BaseModel]] = None,
        confidence_threshold: float = 0.5,
    ):
        self.members = members or [
            XGBoostMember(),
            LightGBMMember(),
            RandomForestMember(),
        ]
        self.confidence_threshold = confidence_threshold
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ConfidenceEnsemble':
        """Train all members."""
        print(f"Training {len(self.members)} ensemble members...")
        for i, member in enumerate(self.members):
            print(f"  [{i+1}/{len(self.members)}] Training {member.name}...")
            member.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using confidence-weighted voting."""
        n_samples = len(X)
        votes = np.zeros((n_samples, 3))
        
        for member in self.members:
            probas = member.predict_proba(X)
            confidences = np.max(probas, axis=1)
            preds = np.argmax(probas, axis=1)
            
            for i in range(n_samples):
                if confidences[i] >= self.confidence_threshold:
                    votes[i, preds[i]] += confidences[i]
        
        # If no confident predictions, fall back to majority vote
        no_votes = votes.sum(axis=1) == 0
        if np.any(no_votes):
            for member in self.members:
                preds = member.predict(X[no_votes])
                for i, pred in enumerate(preds):
                    idx = np.where(no_votes)[0][i]
                    votes[idx, pred] += 1
        
        return np.argmax(votes, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get averaged probabilities weighted by confidence."""
        probas = np.array([m.predict_proba(X) for m in self.members])
        confidences = np.max(probas, axis=2)  # (n_members, n_samples)
        
        # Weight probabilities by confidence
        weighted_probas = np.zeros_like(probas[0])
        for proba, conf in zip(probas, confidences):
            weighted_probas += proba * conf[:, np.newaxis]
        
        # Normalize
        weighted_probas /= weighted_probas.sum(axis=1, keepdims=True) + 1e-8
        return weighted_probas
    
    def save(self, path: str):
        """Save ensemble to file."""
        joblib.dump({
            'members': self.members,
            'confidence_threshold': self.confidence_threshold,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'ConfidenceEnsemble':
        """Load ensemble from file."""
        data = joblib.load(path)
        ensemble = cls(
            members=data['members'],
            confidence_threshold=data['confidence_threshold'],
        )
        ensemble.is_fitted = True
        return ensemble


def create_ensemble(
    ensemble_type: str = 'voting',
    xgb_params: Optional[Dict] = None,
    lgb_params: Optional[Dict] = None,
    rf_params: Optional[Dict] = None,
    **kwargs,
) -> VotingEnsemble | StackingEnsemble | ConfidenceEnsemble:
    """Factory function to create ensemble models.
    
    Args:
        ensemble_type: 'voting', 'stacking', or 'confidence'
        xgb_params: XGBoost hyperparameters
        lgb_params: LightGBM hyperparameters
        rf_params: RandomForest hyperparameters
        **kwargs: Additional ensemble-specific parameters
        
    Returns:
        Ensemble model instance
    """
    members = [
        XGBoostMember(xgb_params),
        LightGBMMember(lgb_params),
        RandomForestMember(rf_params),
    ]
    
    if ensemble_type == 'voting':
        return VotingEnsemble(members=members, **kwargs)
    elif ensemble_type == 'stacking':
        return StackingEnsemble(base_models=members, **kwargs)
    elif ensemble_type == 'confidence':
        return ConfidenceEnsemble(members=members, **kwargs)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

