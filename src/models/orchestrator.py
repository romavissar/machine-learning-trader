"""Dynamic Model Orchestrator for multi-model trading strategy.

Implements the strategy.md logic for automatic model selection based on market regime.
Uses ADX for trend detection and volatility for regime classification.

Model Selection Logic:
- ADX > 25 (trending): XGBoost single model
- ADX < 20 (ranging): Ensemble voting
- Volatility > 2σ (volatile): LSTM + tighter risk
- Sentiment confidence > 0.8: FinBERT override
"""
import warnings
warnings.filterwarnings('ignore', message='.*feature names.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib

from src.models.prediction.xgboost_model import PriceDirectionModel
from src.models.prediction.lstm_model import LSTMPredictor
from src.models.ensemble import VotingEnsemble, ConfidenceEnsemble, XGBoostMember, LightGBMMember, RandomForestMember
from src.risk.manager import RiskManager
from src.features.technical import TechnicalFeatures


@dataclass
class OrchestratorConfig:
    """Configuration for the model orchestrator."""
    # Regime detection thresholds
    adx_trending_threshold: float = 25.0
    adx_ranging_threshold: float = 20.0
    volatility_high_multiplier: float = 2.0
    
    # Confidence thresholds
    single_model_confidence: float = 0.75
    ensemble_min_agreement: float = 0.67
    sentiment_override_confidence: float = 0.8
    
    # Risk limits
    max_position: float = 1.0
    max_drawdown: float = 0.15
    max_volatility: float = 0.05
    
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


class ModelOrchestrator:
    """Dynamic model orchestrator implementing strategy.md logic.
    
    Automatically selects and combines models based on market regime:
    - Trending markets: XGBoost single model
    - Ranging markets: Ensemble voting
    - High volatility: LSTM with tighter risk
    - News events: FinBERT sentiment override
    """
    
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        xgboost_model: Optional[PriceDirectionModel] = None,
        lstm_model: Optional[LSTMPredictor] = None,
        sentiment_analyzer: Optional[Any] = None,
        risk_manager: Optional[RiskManager] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.risk_manager = risk_manager or RiskManager()
        
        # Initialize models
        self.xgboost = xgboost_model or PriceDirectionModel()
        self.lstm = lstm_model
        self.sentiment_analyzer = sentiment_analyzer
        
        # Ensemble for ranging markets
        self.ensemble = VotingEnsemble(
            members=[XGBoostMember(), LightGBMMember(), RandomForestMember()],
            voting='soft'
        )
        
        # Track model states
        self.models_fitted = {
            'xgboost': False,
            'lstm': False,
            'ensemble': False,
        }
        
        # Store historical volatility for regime detection
        self._vol_history: List[float] = []
        self._vol_median: float = 0.0
    
    def compute_regime(
        self,
        prices: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None,
    ) -> str:
        """Determine market regime based on ADX and volatility.
        
        Returns:
            'trending': ADX > 25
            'ranging': ADX < 20 and normal volatility
            'volatile': Volatility > 2x median
        """
        if len(prices) < 20:
            return 'ranging'  # Default when insufficient data
        
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
            return 'volatile'
        elif adx > self.config.adx_trending_threshold:
            return 'trending'
        else:
            return 'ranging'
    
    def select_models(self, regime: str, sentiment_confidence: float = 0.0) -> List[str]:
        """Select which models to use based on regime and sentiment.
        
        Args:
            regime: Current market regime
            sentiment_confidence: Confidence of sentiment signal (0-1)
            
        Returns:
            List of model names to use
        """
        # Sentiment override if very confident
        if sentiment_confidence >= self.config.sentiment_override_confidence:
            return ['sentiment', 'xgboost']
        
        if regime == 'trending':
            return ['xgboost']
        elif regime == 'volatile':
            return ['lstm', 'xgboost'] if self.lstm is not None else ['xgboost']
        else:  # ranging
            return ['ensemble']
    
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
        
        # Train Ensemble
        print("Training Ensemble models...")
        self.ensemble.fit(X, y)
        self.models_fitted['ensemble'] = True
        
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
        
        return self
    
    def _get_xgboost_prediction(self, X: np.ndarray) -> Tuple[int, float]:
        """Get prediction and confidence from XGBoost."""
        proba = self.xgboost.predict_proba(X)
        pred = int(np.argmax(proba, axis=1)[0])
        conf = float(np.max(proba))
        return pred, conf
    
    def _get_ensemble_prediction(self, X: np.ndarray) -> Tuple[int, float, Dict[str, int]]:
        """Get prediction and confidence from ensemble."""
        proba = self.ensemble.predict_proba(X)
        pred = int(np.argmax(proba, axis=1)[0])
        conf = float(np.max(proba))
        
        # Get individual model predictions
        model_preds = {}
        for member in self.ensemble.members:
            member_pred = int(member.predict(X)[0])
            model_preds[member.name] = member_pred
        
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
    ) -> PredictionResult:
        """Generate trading signal using appropriate model(s) based on regime.
        
        Args:
            X: Feature matrix for current prediction (1, n_features)
            prices: Recent price history for regime detection
            high: High prices for ADX calculation
            low: Low prices for ADX calculation
            X_seq: Sequence data for LSTM (1, seq_len, n_features)
            news_text: Optional news text for sentiment
            current_drawdown: Current portfolio drawdown
            current_volatility: Current portfolio volatility
            
        Returns:
            PredictionResult with signal, confidence, and metadata
        """
        # Determine market regime
        regime = self.compute_regime(prices, high, low)
        
        # Get sentiment if available
        sentiment_signal, sentiment_conf = self._get_sentiment_signal(news_text)
        
        # Select models
        selected_models = self.select_models(regime, sentiment_conf)
        
        # Collect predictions
        model_predictions: Dict[str, int] = {}
        model_confidences: Dict[str, float] = {}
        
        # Get predictions from selected models
        if 'xgboost' in selected_models and self.models_fitted.get('xgboost', False):
            pred, conf = self._get_xgboost_prediction(X)
            model_predictions['xgboost'] = pred
            model_confidences['xgboost'] = conf
        
        if 'ensemble' in selected_models and self.models_fitted.get('ensemble', False):
            pred, conf, member_preds = self._get_ensemble_prediction(X)
            model_predictions['ensemble'] = pred
            model_confidences['ensemble'] = conf
            model_predictions.update(member_preds)
        
        if 'lstm' in selected_models and X_seq is not None and self.models_fitted.get('lstm', False):
            pred, conf = self._get_lstm_prediction(X_seq)
            model_predictions['lstm'] = pred
            model_confidences['lstm'] = conf
        
        if 'sentiment' in selected_models:
            model_predictions['sentiment'] = sentiment_signal
            model_confidences['sentiment'] = sentiment_conf
        
        # Combine predictions based on regime weights
        final_signal, final_confidence = self._combine_predictions(
            regime, model_predictions, model_confidences
        )
        
        # Apply risk management
        risk_adjusted_position = self._apply_risk_adjustment(
            final_signal, final_confidence, regime,
            current_drawdown, current_volatility
        )
        
        return PredictionResult(
            signal=final_signal,
            confidence=final_confidence,
            regime=regime,
            selected_models=selected_models,
            model_predictions=model_predictions,
            model_confidences=model_confidences,
            risk_adjusted_position=risk_adjusted_position,
        )
    
    def _combine_predictions(
        self,
        regime: str,
        predictions: Dict[str, int],
        confidences: Dict[str, float],
    ) -> Tuple[int, float]:
        """Combine predictions from multiple models based on regime weights."""
        if not predictions:
            return 1, 0.0  # Default hold
        
        # Get weights for current regime
        if regime == 'trending':
            weights = self.config.trending_weights
        elif regime == 'volatile':
            weights = self.config.volatile_weights
        else:
            weights = self.config.ranging_weights
        
        # Weighted voting
        vote_weights = np.zeros(3)  # [down, hold, up]
        total_weight = 0.0
        
        for model, pred in predictions.items():
            weight = weights.get(model, 0.0)
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
        drawdown: float,
        volatility: float,
    ) -> float:
        """Apply risk-based position sizing."""
        # Base position from signal
        if signal == 2:
            position = 1.0
        elif signal == 0:
            position = -1.0
        else:
            position = 0.0
        
        # Scale by confidence
        if confidence < 0.6:
            position *= 0.5
        
        # Tighter limits for volatile regime
        if regime == 'volatile':
            position *= 0.5
        
        # Check risk limits
        risk_limits = {
            'max_drawdown': self.config.max_drawdown,
            'max_volatility': self.config.max_volatility,
        }
        
        if self.risk_manager.should_halt_trading(drawdown, volatility, risk_limits):
            return 0.0
        
        # Enforce position limits
        max_pos = self.config.max_position
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
            'ensemble': self.ensemble,
            'models_fitted': self.models_fitted,
            'vol_history': self._vol_history,
            'vol_median': self._vol_median,
        }
        joblib.dump(state, path)
        print(f"Orchestrator saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ModelOrchestrator':
        """Load orchestrator state from file."""
        state = joblib.load(path)
        
        orchestrator = cls(config=state['config'])
        orchestrator.xgboost = state['xgboost']
        orchestrator.ensemble = state['ensemble']
        orchestrator.models_fitted = state['models_fitted']
        orchestrator._vol_history = state['vol_history']
        orchestrator._vol_median = state['vol_median']
        
        return orchestrator

