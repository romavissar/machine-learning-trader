"""Dynamic Model Orchestrator for multi-model trading strategy.

Implements the strategy.md logic for automatic model selection based on market regime.
Uses ADX for trend detection and volatility for regime classification.

Model Selection Logic (strategy.md compliant):
- ADX > 25 (trending): XGBoost single model (confidence threshold 0.7)
- ADX < 20 (ranging): ConfidenceEnsemble with 2/3 agreement
- Volatility > 2σ (volatile): LSTM + tighter risk
- Sentiment confidence > 0.8: FinBERT override (60% sentiment, 40% prediction)
- All models < 60% confidence: HOLD / Reduce Exposure

Best-Fit Scoring:
    best_fit_score = (
        historical_accuracy * 0.30 +
        regime_match * 0.25 +
        confidence_level * 0.20 +
        recency_weight * 0.15 +
        diversity_bonus * 0.10
    )

Risk Limits by Model Type:
    | Model Type      | Max Position | Stop-Loss ATR | Max Drawdown |
    |-----------------|-------------|---------------|--------------|
    | XGBoost         | 1.0         | 2x            | 15%          |
    | Ensemble        | 1.0         | 2x            | 15%          |
    | PPO (RL)        | 0.5         | 3x            | 10%          |
    | Sentiment-only  | 0.3         | 1.5x          | 5%           |
"""
import warnings
warnings.filterwarnings('ignore', message='.*feature names.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from collections import deque

from src.models.prediction.xgboost_model import PriceDirectionModel
from src.models.prediction.lstm_model import LSTMPredictor
from src.models.ensemble import VotingEnsemble, ConfidenceEnsemble, XGBoostMember, LightGBMMember, RandomForestMember
from src.risk.manager import RiskManager
from src.features.technical import TechnicalFeatures


@dataclass
class ModelRiskLimits:
    """Risk limits specific to each model type (from strategy.md)."""
    max_position: float
    stop_loss_atr_multiplier: float
    max_drawdown: float


@dataclass
class ModelPerformance:
    """Track recent model performance for health monitoring."""
    accuracy_history: List[float] = field(default_factory=list)
    returns_history: List[float] = field(default_factory=list)
    last_trained_step: int = 0
    
    def get_accuracy(self, window: int = 100) -> float:
        """Get recent accuracy."""
        if not self.accuracy_history:
            return 0.5  # Default neutral
        recent = self.accuracy_history[-window:]
        return sum(recent) / len(recent)
    
    def get_sharpe(self, window: int = 720) -> float:
        """Get recent Sharpe ratio (30 days at hourly = 720)."""
        if len(self.returns_history) < 2:
            return 0.0
        recent = self.returns_history[-window:]
        mean_ret = np.mean(recent)
        std_ret = np.std(recent) + 1e-8
        return float(mean_ret / std_ret * np.sqrt(8760))  # Annualized
    
    def get_max_drawdown(self, window: int = 720) -> float:
        """Get recent max drawdown."""
        if len(self.returns_history) < 2:
            return 0.0
        recent = self.returns_history[-window:]
        cumulative = np.cumprod(1 + np.array(recent))
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + 1e-8)
        return float(np.max(drawdown))


@dataclass
class OrchestratorConfig:
    """Configuration for the model orchestrator (strategy.md compliant)."""
    # Regime detection thresholds
    adx_trending_threshold: float = 25.0
    adx_ranging_threshold: float = 20.0
    volatility_high_multiplier: float = 2.0
    
    # Confidence thresholds (from strategy.md)
    single_model_confidence: float = 0.75  # XGBoost alone needs 75%
    ensemble_min_agreement: float = 0.67   # 2/3 models must agree
    sentiment_override_confidence: float = 0.8
    min_confidence_threshold: float = 0.60  # Below this -> HOLD
    position_scaling_threshold: float = 0.60  # Scale position below this
    
    # Global risk limits (defaults)
    max_position: float = 1.0
    max_drawdown: float = 0.15
    max_volatility: float = 0.05
    stop_loss_atr_multiplier: float = 2.0
    
    # Per-model risk limits (from strategy.md table)
    model_risk_limits: Dict[str, ModelRiskLimits] = field(default_factory=lambda: {
        'xgboost': ModelRiskLimits(max_position=1.0, stop_loss_atr_multiplier=2.0, max_drawdown=0.15),
        'ensemble': ModelRiskLimits(max_position=1.0, stop_loss_atr_multiplier=2.0, max_drawdown=0.15),
        'ppo': ModelRiskLimits(max_position=0.5, stop_loss_atr_multiplier=3.0, max_drawdown=0.10),
        'sentiment': ModelRiskLimits(max_position=0.3, stop_loss_atr_multiplier=1.5, max_drawdown=0.05),
        'lstm': ModelRiskLimits(max_position=0.8, stop_loss_atr_multiplier=2.5, max_drawdown=0.12),
    })
    
    # Best-fit scoring weights (from strategy.md)
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        'historical_accuracy': 0.30,
        'regime_match': 0.25,
        'confidence_level': 0.20,
        'recency_weight': 0.15,
        'diversity_bonus': 0.10,
    })
    
    # Model weights by regime
    trending_weights: Dict[str, float] = field(default_factory=lambda: {
        'xgboost': 0.7, 'lstm': 0.2, 'sentiment': 0.1
    })
    ranging_weights: Dict[str, float] = field(default_factory=lambda: {
        'xgboost': 0.33, 'lightgbm': 0.33, 'random_forest': 0.34
    })
    volatile_weights: Dict[str, float] = field(default_factory=lambda: {
        'lstm': 0.6, 'xgboost': 0.3, 'sentiment': 0.1
    })
    news_event_weights: Dict[str, float] = field(default_factory=lambda: {
        'sentiment': 0.6, 'xgboost': 0.3, 'lstm': 0.1
    })
    
    # PPO configuration
    ppo_enabled: bool = False
    ppo_model_path: Optional[str] = None


@dataclass
class PredictionResult:
    """Container for orchestrator prediction output."""
    signal: int  # 0=down, 1=hold, 2=up
    confidence: float
    regime: str
    selected_models: List[str]
    model_predictions: Dict[str, int]
    model_confidences: Dict[str, float]
    risk_adjusted_position: float
    stop_loss_price: Optional[float] = None
    model_scores: Optional[Dict[str, float]] = None


class ModelOrchestrator:
    """Dynamic model orchestrator implementing strategy.md logic.
    
    Automatically selects and combines models based on market regime:
    - Trending markets: XGBoost single model (confidence > 0.75)
    - Ranging markets: ConfidenceEnsemble (2/3 agreement)
    - High volatility: LSTM with tighter risk
    - News events: FinBERT sentiment override (60% weight)
    - Low confidence: HOLD and reduce exposure to 0
    
    Features:
    - Best-fit scoring system for model selection
    - Model health monitoring (disable underperforming models)
    - Per-model risk limits
    - ATR-based stop-loss calculation
    - Optional PPO position sizing (Layer 2)
    """
    
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        xgboost_model: Optional[PriceDirectionModel] = None,
        lstm_model: Optional[LSTMPredictor] = None,
        sentiment_analyzer: Optional[Any] = None,
        risk_manager: Optional[RiskManager] = None,
        ppo_trader: Optional[Any] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.risk_manager = risk_manager or RiskManager()
        
        # Initialize models
        self.xgboost = xgboost_model or PriceDirectionModel()
        self.lstm = lstm_model
        self.sentiment_analyzer = sentiment_analyzer
        self.ppo_trader = ppo_trader
        
        # Ensemble for ranging markets - use ConfidenceEnsemble per strategy.md
        self.confidence_ensemble = ConfidenceEnsemble(
            members=[XGBoostMember(), LightGBMMember(), RandomForestMember()],
            confidence_threshold=0.5,
        )
        
        # Also keep VotingEnsemble for comparison/fallback
        self.voting_ensemble = VotingEnsemble(
            members=[XGBoostMember(), LightGBMMember(), RandomForestMember()],
            voting='soft'
        )
        
        # Track model states
        self.models_fitted = {
            'xgboost': False,
            'lstm': False,
            'confidence_ensemble': False,
            'voting_ensemble': False,
            'ppo': False,
        }
        
        # Model performance tracking for health monitoring
        self.model_performance: Dict[str, ModelPerformance] = {
            'xgboost': ModelPerformance(),
            'lstm': ModelPerformance(),
            'ensemble': ModelPerformance(),
            'sentiment': ModelPerformance(),
            'ppo': ModelPerformance(),
        }
        
        # Track current step for recency weighting
        self._current_step: int = 0
        
        # Store historical volatility for regime detection
        self._vol_history: List[float] = []
        self._vol_median: float = 0.0
        
        # Store recent ATR for stop-loss calculation
        self._last_atr: float = 0.0
    
    def should_use_model(self, model_name: str) -> bool:
        """Check if model should be used based on recent performance.
        
        From strategy.md:
            if recent_performance['sharpe_30d'] < 0:
                return False  # Disable underperforming model
            if recent_performance['max_drawdown_30d'] > LIMITS['max_drawdown']:
                return False  # Risk limit breach
        """
        if model_name not in self.model_performance:
            return True
        
        perf = self.model_performance[model_name]
        limits = self.config.model_risk_limits.get(
            model_name, 
            ModelRiskLimits(max_position=1.0, stop_loss_atr_multiplier=2.0, max_drawdown=0.15)
        )
        
        # Check Sharpe ratio (30-day)
        sharpe = perf.get_sharpe(window=720)  # 30 days * 24 hours
        if sharpe < 0 and len(perf.returns_history) > 100:
            return False
        
        # Check max drawdown
        max_dd = perf.get_max_drawdown(window=720)
        if max_dd > limits.max_drawdown and len(perf.returns_history) > 100:
            return False
        
        return True
    
    def compute_model_score(
        self,
        model_name: str,
        regime: str,
        confidence: float,
    ) -> float:
        """Compute best-fit score for a model.
        
        From strategy.md:
            best_fit_score = (
                historical_accuracy * 0.30 +
                regime_match * 0.25 +
                confidence_level * 0.20 +
                recency_weight * 0.15 +
                diversity_bonus * 0.10
            )
        """
        weights = self.config.score_weights
        perf = self.model_performance.get(model_name, ModelPerformance())
        
        # Historical accuracy (0-1)
        historical_accuracy = perf.get_accuracy()
        
        # Regime match (1.0 if model matches regime, 0.5 otherwise)
        regime_match_map = {
            'trending': ['xgboost'],
            'ranging': ['ensemble', 'confidence_ensemble'],
            'volatile': ['lstm', 'xgboost'],
        }
        matching_models = regime_match_map.get(regime, [])
        regime_match = 1.0 if model_name in matching_models else 0.5
        
        # Confidence level (0-1)
        confidence_level = confidence
        
        # Recency weight (decay based on steps since last training)
        steps_since_training = self._current_step - perf.last_trained_step
        recency_weight = max(0.0, 1.0 - steps_since_training / 10000)
        
        # Diversity bonus (higher if model disagrees with others - adds value)
        # Simplified: give bonus to less-used models
        diversity_bonus = 0.5 if model_name not in ['xgboost', 'ensemble'] else 0.3
        
        score = (
            historical_accuracy * weights['historical_accuracy'] +
            regime_match * weights['regime_match'] +
            confidence_level * weights['confidence_level'] +
            recency_weight * weights['recency_weight'] +
            diversity_bonus * weights['diversity_bonus']
        )
        
        return score
    
    def update_model_performance(
        self,
        model_name: str,
        was_correct: bool,
        pnl: float,
    ) -> None:
        """Update model performance tracking after a prediction."""
        if model_name not in self.model_performance:
            self.model_performance[model_name] = ModelPerformance()
        
        perf = self.model_performance[model_name]
        perf.accuracy_history.append(1.0 if was_correct else 0.0)
        perf.returns_history.append(pnl)
        
        # Keep bounded history
        max_history = 2000
        if len(perf.accuracy_history) > max_history:
            perf.accuracy_history = perf.accuracy_history[-max_history:]
        if len(perf.returns_history) > max_history:
            perf.returns_history = perf.returns_history[-max_history:]
    
    def compute_regime(
        self,
        prices: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None,
    ) -> Tuple[str, float]:
        """Determine market regime based on ADX and volatility.
        
        Returns:
            Tuple of (regime, atr):
            - 'trending': ADX > 25
            - 'ranging': ADX < 20 and normal volatility
            - 'volatile': Volatility > 2x median
            - atr: Average True Range for stop-loss calculation
        """
        if len(prices) < 20:
            return 'ranging', 0.0  # Default when insufficient data
        
        # Create DataFrame for TechnicalFeatures
        df = pd.DataFrame({
            'close': prices,
            'high': high if high is not None else prices,
            'low': low if low is not None else prices,
            'volume': volumes if volumes is not None else np.ones_like(prices),
        })
        
        tech = TechnicalFeatures(df)
        
        # Get latest ADX value
        adx_series = tech.adx()
        adx = adx_series.iloc[-1] if not pd.isna(adx_series.iloc[-1]) else 15.0
        
        # Get ATR for stop-loss calculation
        atr_series = tech.atr()
        atr = atr_series.iloc[-1] if not pd.isna(atr_series.iloc[-1]) else 0.0
        self._last_atr = atr
        
        # Get volatility regime
        vol_series = tech.historical_volatility()
        current_vol = vol_series.iloc[-1] if not pd.isna(vol_series.iloc[-1]) else 0.0
        
        # Update volatility history
        if not pd.isna(current_vol) and current_vol > 0:
            self._vol_history.append(current_vol)
            if len(self._vol_history) > 100:
                self._vol_history = self._vol_history[-100:]
            self._vol_median = np.median(self._vol_history) if self._vol_history else current_vol
        
        # Determine regime
        if self._vol_median > 0 and current_vol > self.config.volatility_high_multiplier * self._vol_median:
            return 'volatile', atr
        elif adx > self.config.adx_trending_threshold:
            return 'trending', atr
        else:
            return 'ranging', atr
    
    def select_models(
        self, 
        regime: str, 
        sentiment_confidence: float = 0.0,
        use_scoring: bool = True,
    ) -> List[str]:
        """Select which models to use based on regime, sentiment, and health.
        
        From strategy.md:
        - Sentiment override if confidence > 0.8
        - Trending: XGBoost single model
        - Ranging: Ensemble with 2/3 agreement
        - Volatile: LSTM + XGBoost
        
        Also filters out unhealthy models.
        """
        # Sentiment override if very confident (strategy.md)
        if sentiment_confidence >= self.config.sentiment_override_confidence:
            return ['sentiment', 'xgboost']
        
        # Base selection by regime
        if regime == 'trending':
            candidates = ['xgboost']
        elif regime == 'volatile':
            candidates = ['lstm', 'xgboost'] if self.lstm is not None else ['xgboost']
        else:  # ranging
            candidates = ['confidence_ensemble']
        
        # Filter out unhealthy models
        healthy_models = [m for m in candidates if self.should_use_model(m)]
        
        # Fallback if all filtered out
        if not healthy_models:
            healthy_models = ['xgboost'] if self.models_fitted.get('xgboost') else candidates
        
        return healthy_models
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_seq: Optional[np.ndarray] = None,
    ) -> 'ModelOrchestrator':
        """Train all sub-models with temporal awareness.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            X_seq: Sequence data for LSTM (n_samples, seq_len, n_features)
            
        Returns:
            Self for chaining
        """
        # Train XGBoost
        print("Training XGBoost model...")
        self.xgboost.fit(X, y)
        self.models_fitted['xgboost'] = True
        self.model_performance['xgboost'].last_trained_step = self._current_step
        
        # Train ConfidenceEnsemble (for ranging markets per strategy.md)
        print("Training ConfidenceEnsemble models...")
        self.confidence_ensemble.fit(X, y)
        self.models_fitted['confidence_ensemble'] = True
        self.model_performance['ensemble'].last_trained_step = self._current_step
        
        # Also train VotingEnsemble as fallback
        print("Training VotingEnsemble models...")
        self.voting_ensemble.fit(X, y)
        self.models_fitted['voting_ensemble'] = True
        
        # Train LSTM if sequence data provided
        if X_seq is not None and self.lstm is not None:
            print("Training LSTM model...")
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.lstm = self.lstm.to(device)
            
            # Create data loader
            X_tensor = torch.FloatTensor(X_seq)
            y_tensor = torch.LongTensor(y)
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)  # No shuffle for temporal
            
            # Train
            optimizer = torch.optim.Adam(self.lstm.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss()
            
            for epoch in range(10):
                loss = self.lstm.train_epoch(loader, optimizer, criterion, device)
                if epoch % 5 == 0:
                    print(f"  Epoch {epoch}: Loss = {loss:.4f}")
            
            self.models_fitted['lstm'] = True
            self.model_performance['lstm'].last_trained_step = self._current_step
        
        return self
    
    def _get_xgboost_prediction(self, X: np.ndarray) -> Tuple[int, float]:
        """Get prediction and confidence from XGBoost."""
        proba = self.xgboost.predict_proba(X)
        pred = int(np.argmax(proba, axis=1)[0])
        conf = float(np.max(proba))
        return pred, conf
    
    def _get_ensemble_prediction(self, X: np.ndarray, use_confidence: bool = True) -> Tuple[int, float, Dict[str, int]]:
        """Get prediction and confidence from ensemble.
        
        Args:
            use_confidence: If True, use ConfidenceEnsemble (strategy.md default for ranging)
        """
        ensemble = self.confidence_ensemble if use_confidence else self.voting_ensemble
        
        proba = ensemble.predict_proba(X)
        pred = int(np.argmax(proba, axis=1)[0])
        conf = float(np.max(proba))
        
        # Get individual model predictions for agreement check
        model_preds = {}
        for member in ensemble.members:
            member_pred = int(member.predict(X)[0])
            model_preds[member.name] = member_pred
        
        # Check agreement (strategy.md: require 2/3 agreement)
        pred_counts = {}
        for p in model_preds.values():
            pred_counts[p] = pred_counts.get(p, 0) + 1
        
        max_agreement = max(pred_counts.values()) / len(model_preds)
        if max_agreement < self.config.ensemble_min_agreement:
            # Not enough agreement - reduce confidence
            conf *= 0.5
        
        return pred, conf, model_preds
    
    def _get_lstm_prediction(self, X_seq: np.ndarray) -> Tuple[int, float]:
        """Get prediction and confidence from LSTM."""
        if self.lstm is None or not self.models_fitted.get('lstm', False):
            return 1, 0.33  # Default hold with low confidence
        
        import torch
        device = next(self.lstm.parameters()).device
        X_tensor = torch.FloatTensor(X_seq).to(device)
        
        with torch.no_grad():
            logits = self.lstm(X_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        
        pred = int(np.argmax(proba, axis=1)[0])
        conf = float(np.max(proba))
        return pred, conf
    
    def _get_sentiment_signal(self, news_text: Optional[str] = None) -> Tuple[int, float]:
        """Get trading signal from sentiment analysis."""
        if self.sentiment_analyzer is None or news_text is None:
            return 1, 0.0  # Neutral with no confidence
        
        sentiment = self.sentiment_analyzer.analyze(news_text)
        
        # Convert sentiment to signal
        if sentiment.get('positive', 0) > 0.5:
            signal = 2  # Up
            conf = sentiment['positive']
        elif sentiment.get('negative', 0) > 0.5:
            signal = 0  # Down
            conf = sentiment['negative']
        else:
            signal = 1  # Hold
            conf = sentiment.get('neutral', 0.5)
        
        return signal, conf
    
    def _get_ppo_position(
        self, 
        observation: np.ndarray,
        base_position: float,
    ) -> float:
        """Get position sizing from PPO agent (Layer 2).
        
        If PPO is not available, falls back to rule-based sizing.
        """
        if not self.config.ppo_enabled or self.ppo_trader is None:
            return base_position  # Fallback to rule-based
        
        try:
            action = self.ppo_trader.predict(observation, deterministic=True)
            # PPO outputs action in [-1, 1], scale by base position direction
            ppo_position = float(action[0]) if hasattr(action, '__len__') else float(action)
            return np.sign(base_position) * abs(ppo_position) if base_position != 0 else 0.0
        except Exception:
            return base_position  # Fallback on error
    
    def generate_signal(
        self,
        X: np.ndarray,
        prices: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        X_seq: Optional[np.ndarray] = None,
        news_text: Optional[str] = None,
        current_drawdown: float = 0.0,
        current_volatility: float = 0.0,
        entry_price: Optional[float] = None,
    ) -> PredictionResult:
        """Generate trading signal using appropriate model(s) based on regime.
        
        Implements full strategy.md logic:
        1. Determine market regime (ADX, volatility)
        2. Check for sentiment override
        3. Select models based on regime + health
        4. Compute best-fit scores
        5. Get predictions and combine
        6. Apply risk management with per-model limits
        7. Calculate ATR-based stop-loss
        8. Handle low confidence (HOLD + 0 position)
        
        Args:
            X: Feature matrix for current prediction (1, n_features)
            prices: Recent price history for regime detection
            high: High prices for ADX calculation
            low: Low prices for ADX calculation
            X_seq: Sequence data for LSTM (1, seq_len, n_features)
            news_text: Optional news text for sentiment
            current_drawdown: Current portfolio drawdown
            current_volatility: Current portfolio volatility
            entry_price: Current entry price for stop-loss calculation
            
        Returns:
            PredictionResult with signal, confidence, and metadata
        """
        self._current_step += 1
        
        # Determine market regime and get ATR
        regime, atr = self.compute_regime(prices, high, low)
        
        # Get sentiment if available
        sentiment_signal, sentiment_conf = self._get_sentiment_signal(news_text)
        
        # Select models based on regime and health
        selected_models = self.select_models(regime, sentiment_conf)
        
        # Collect predictions and compute scores
        model_predictions: Dict[str, int] = {}
        model_confidences: Dict[str, float] = {}
        model_scores: Dict[str, float] = {}
        
        # Get predictions from selected models
        if 'xgboost' in selected_models and self.models_fitted.get('xgboost', False):
            pred, conf = self._get_xgboost_prediction(X)
            model_predictions['xgboost'] = pred
            model_confidences['xgboost'] = conf
            model_scores['xgboost'] = self.compute_model_score('xgboost', regime, conf)
        
        if 'confidence_ensemble' in selected_models and self.models_fitted.get('confidence_ensemble', False):
            pred, conf, member_preds = self._get_ensemble_prediction(X, use_confidence=True)
            model_predictions['ensemble'] = pred
            model_confidences['ensemble'] = conf
            model_predictions.update(member_preds)
            model_scores['ensemble'] = self.compute_model_score('ensemble', regime, conf)
        
        if 'lstm' in selected_models and X_seq is not None and self.models_fitted.get('lstm', False):
            pred, conf = self._get_lstm_prediction(X_seq)
            model_predictions['lstm'] = pred
            model_confidences['lstm'] = conf
            model_scores['lstm'] = self.compute_model_score('lstm', regime, conf)
        
        if 'sentiment' in selected_models:
            model_predictions['sentiment'] = sentiment_signal
            model_confidences['sentiment'] = sentiment_conf
            model_scores['sentiment'] = self.compute_model_score('sentiment', regime, sentiment_conf)
        
        # Combine predictions based on regime weights
        final_signal, final_confidence = self._combine_predictions(
            regime, model_predictions, model_confidences,
            sentiment_override=(sentiment_conf >= self.config.sentiment_override_confidence)
        )
        
        # LOW CONFIDENCE HANDLING (strategy.md):
        # If all models < 60% confidence, return HOLD with 0 position
        if final_confidence < self.config.min_confidence_threshold:
            final_signal = 1  # HOLD
            risk_adjusted_position = 0.0
            stop_loss_price = None
        else:
            # Determine primary model for risk limits
            primary_model = self._get_primary_model(selected_models, model_scores)
            
            # Apply risk management with per-model limits
            risk_adjusted_position = self._apply_risk_adjustment(
                final_signal, final_confidence, regime, primary_model,
                current_drawdown, current_volatility
            )
            
            # Optional PPO position sizing (Layer 2)
            if self.config.ppo_enabled and self.ppo_trader is not None:
                risk_adjusted_position = self._get_ppo_position(X[0], risk_adjusted_position)
            
            # Calculate ATR-based stop-loss
            stop_loss_price = None
            if entry_price is not None and atr > 0:
                limits = self.config.model_risk_limits.get(
                    primary_model,
                    ModelRiskLimits(max_position=1.0, stop_loss_atr_multiplier=2.0, max_drawdown=0.15)
                )
                stop_loss_price = self.risk_manager.calculate_stop_loss(
                    entry_price, atr, limits.stop_loss_atr_multiplier
                )
        
        return PredictionResult(
            signal=final_signal,
            confidence=final_confidence,
            regime=regime,
            selected_models=selected_models,
            model_predictions=model_predictions,
            model_confidences=model_confidences,
            risk_adjusted_position=risk_adjusted_position,
            stop_loss_price=stop_loss_price,
            model_scores=model_scores,
        )
    
    def _get_primary_model(
        self, 
        selected_models: List[str], 
        model_scores: Dict[str, float]
    ) -> str:
        """Get the primary model (highest score) for risk limit lookup."""
        if not model_scores:
            return selected_models[0] if selected_models else 'xgboost'
        
        # Map ensemble variants to 'ensemble' for risk lookup
        score_map = {}
        for model, score in model_scores.items():
            key = 'ensemble' if 'ensemble' in model else model
            if key not in score_map or score > score_map[key]:
                score_map[key] = score
        
        return max(score_map, key=score_map.get) if score_map else 'xgboost'
    
    def _combine_predictions(
        self,
        regime: str,
        predictions: Dict[str, int],
        confidences: Dict[str, float],
        sentiment_override: bool = False,
    ) -> Tuple[int, float]:
        """Combine predictions from multiple models based on regime weights.
        
        From strategy.md:
        - If sentiment override: 60% sentiment, 40% prediction weights
        - Otherwise use regime-specific weights
        """
        if not predictions:
            return 1, 0.0  # Default hold
        
        # Get weights for current regime
        if sentiment_override:
            weights = self.config.news_event_weights
        elif regime == 'trending':
            weights = self.config.trending_weights
        elif regime == 'volatile':
            weights = self.config.volatile_weights
        else:
            weights = self.config.ranging_weights
        
        # Weighted voting
        vote_weights = np.zeros(3)  # [down, hold, up]
        total_weight = 0.0
        
        for model, pred in predictions.items():
            # Handle ensemble member names
            model_key = model
            if model in ['lightgbm', 'random_forest']:
                model_key = model  # These are in ranging_weights
            
            weight = weights.get(model_key, 0.0)
            if weight == 0 and model in predictions:
                weight = 1.0 / len(predictions)  # Equal weight fallback
            
            conf = confidences.get(model, 0.5)
            vote_weights[pred] += weight * conf
            total_weight += weight * conf
        
        if total_weight > 0:
            vote_weights /= total_weight
        
        final_signal = int(np.argmax(vote_weights))
        final_confidence = float(vote_weights[final_signal])
        
        return final_signal, final_confidence
    
    def _apply_risk_adjustment(
        self,
        signal: int,
        confidence: float,
        regime: str,
        primary_model: str,
        drawdown: float,
        volatility: float,
    ) -> float:
        """Apply risk-based position sizing with per-model limits.
        
        From strategy.md:
        - Use model-specific risk limits
        - Scale position by confidence if < threshold
        - Tighter limits for volatile regime
        """
        # Get model-specific risk limits
        limits = self.config.model_risk_limits.get(
            primary_model,
            ModelRiskLimits(max_position=1.0, stop_loss_atr_multiplier=2.0, max_drawdown=0.15)
        )
        
        # Base position from signal
        if signal == 2:
            position = 1.0
        elif signal == 0:
            position = -1.0
        else:
            position = 0.0
        
        # Scale by confidence (strategy.md: if confidence < 0.6, halve position)
        if confidence < self.config.position_scaling_threshold:
            position *= 0.5
        
        # Tighter limits for volatile regime
        if regime == 'volatile':
            position *= 0.5
        
        # Check risk limits using model-specific thresholds
        risk_check = {
            'max_drawdown': limits.max_drawdown,
            'max_volatility': self.config.max_volatility,
        }
        
        if self.risk_manager.should_halt_trading(drawdown, volatility, risk_check):
            return 0.0
        
        # Enforce model-specific position limits
        max_pos = limits.max_position
        if regime == 'volatile':
            max_pos *= 0.5
        
        if not self.risk_manager.check_position_limit(position, max_pos):
            position = np.sign(position) * max_pos
        
        return position
    
    def save(self, path: str) -> None:
        """Save orchestrator state to file."""
        state = {
            'config': self.config,
            'xgboost': self.xgboost,
            'confidence_ensemble': self.confidence_ensemble,
            'voting_ensemble': self.voting_ensemble,
            'models_fitted': self.models_fitted,
            'model_performance': self.model_performance,
            'vol_history': self._vol_history,
            'vol_median': self._vol_median,
            'current_step': self._current_step,
            'last_atr': self._last_atr,
        }
        joblib.dump(state, path)
        print(f"Orchestrator saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ModelOrchestrator':
        """Load orchestrator state from file."""
        state = joblib.load(path)
        
        orchestrator = cls(config=state['config'])
        orchestrator.xgboost = state['xgboost']
        
        # Handle both old and new ensemble names
        if 'confidence_ensemble' in state:
            orchestrator.confidence_ensemble = state['confidence_ensemble']
        elif 'ensemble' in state:
            # Migrate from old format
            orchestrator.confidence_ensemble = state['ensemble']
            
        if 'voting_ensemble' in state:
            orchestrator.voting_ensemble = state['voting_ensemble']
            
        orchestrator.models_fitted = state['models_fitted']
        
        # Handle old format without new fields
        if 'model_performance' in state:
            orchestrator.model_performance = state['model_performance']
        
        orchestrator._vol_history = state.get('vol_history', [])
        orchestrator._vol_median = state.get('vol_median', 0.0)
        orchestrator._current_step = state.get('current_step', 0)
        orchestrator._last_atr = state.get('last_atr', 0.0)
        
        return orchestrator
