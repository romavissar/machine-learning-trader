"""XStrat V1: Research-backed trading strategy.

Implements all 6 layers from new_strategy.md:
- Layer 0: SMA200 trend filter (long-only above, short-only below)
- Layer 1: ADX regime detection (trending/ranging/volatile)
- Layer 2: Trade frequency control (max 4/day, 6hr intervals)
- Layer 3: Volatility-targeted position sizing (10-50%)
- Layer 4: Executed stop-losses + drawdown halt
- Layer 5: XGBoost signal generation with all filters applied
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
import joblib
from xgboost import XGBClassifier


@dataclass
class TradeState:
    """Track trading state for frequency control."""
    trades_today: int = 0
    last_trade_time: Optional[datetime] = None
    last_trade_pnl: float = 0.0
    current_day: Optional[datetime] = None
    
    def reset_if_new_day(self, now: datetime):
        if self.current_day is None or now.date() != self.current_day.date():
            self.trades_today = 0
            self.current_day = now


@dataclass
class Position:
    """Track current position."""
    size: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    stop_loss: float = 0.0


@dataclass
class XStratResult:
    """Output from strategy signal generation."""
    signal: int  # 0=short, 1=hold, 2=long
    position_size: float
    stop_loss_price: Optional[float]
    trend: str
    regime: str
    confidence: float
    trade_allowed: bool
    reason: str


class XStratV1:
    """Research-backed trading strategy with 6 protection layers."""
    
    def __init__(self, config_path: str = "config/xstrat_v1.yaml"):
        self.config = self._load_config(config_path)
        self.model: Optional[XGBClassifier] = None
        self.trade_state = TradeState()
        self.position = Position()
        self.equity_peak = 10000.0
        self.current_equity = 10000.0
        self.halted = False
        self._vol_history: list = []
    
    def _load_config(self, path: str) -> dict:
        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> dict:
        return {
            'trend_filter': {'sma_period': 200, 'buffer_zone': 0.02},
            'regime': {'adx_trending': 25, 'adx_ranging': 20, 'vol_multiplier': 2.0},
            'trade_limits': {'max_per_day': 4, 'min_interval_hours': 6, 'loss_cooldown_hours': 12},
            'position': {'max_size': 0.50, 'vol_target': 0.02, 
                        'confidence_threshold': {'trending': 0.65, 'ranging': 0.80, 'volatile': 1.0}},
            'risk': {'stop_loss_atr': 2.0, 'max_drawdown_halt': 0.10, 'drawdown_recovery': 0.05},
            'costs': {'transaction': 0.001, 'min_expected_profit': 0.003}
        }
    
    # =========================================================================
    # Layer 0: SMA200 Trend Filter
    # =========================================================================
    def get_trend(self, prices: np.ndarray) -> str:
        """Determine trend direction using SMA200."""
        period = self.config['trend_filter']['sma_period']
        if len(prices) < period:
            return 'neutral'
        sma = np.mean(prices[-period:])
        current = prices[-1]
        buffer = sma * self.config['trend_filter']['buffer_zone']
        if current > sma + buffer:
            return 'long_only'
        if current < sma - buffer:
            return 'short_only'
        return 'neutral'
    
    # =========================================================================
    # Layer 1: Regime Detection
    # =========================================================================
    def get_regime(self, adx: float, volatility: float) -> str:
        """Determine market regime from ADX and volatility."""
        cfg = self.config['regime']
        # Update volatility history
        if volatility > 0:
            self._vol_history.append(volatility)
            if len(self._vol_history) > 100:
                self._vol_history = self._vol_history[-100:]
        vol_median = np.median(self._vol_history) if self._vol_history else volatility
        
        if vol_median > 0 and volatility > cfg['vol_multiplier'] * vol_median:
            return 'volatile'
        if adx > cfg['adx_trending']:
            return 'trending'
        return 'ranging'
    
    # =========================================================================
    # Layer 2: Trade Frequency Control
    # =========================================================================
    def can_trade(self, now: datetime) -> Tuple[bool, str]:
        """Check if trading is allowed based on frequency limits."""
        cfg = self.config['trade_limits']
        self.trade_state.reset_if_new_day(now)
        
        if self.trade_state.trades_today >= cfg['max_per_day']:
            return False, f"Daily limit reached ({cfg['max_per_day']})"
        
        if self.trade_state.last_trade_time:
            hours_since = (now - self.trade_state.last_trade_time).total_seconds() / 3600
            if hours_since < cfg['min_interval_hours']:
                return False, f"Min interval not met ({hours_since:.1f}h < {cfg['min_interval_hours']}h)"
            if self.trade_state.last_trade_pnl < 0 and hours_since < cfg['loss_cooldown_hours']:
                return False, f"Loss cooldown ({hours_since:.1f}h < {cfg['loss_cooldown_hours']}h)"
        
        return True, "OK"
    
    def record_trade(self, now: datetime, pnl: float):
        """Record a trade for frequency tracking."""
        self.trade_state.trades_today += 1
        self.trade_state.last_trade_time = now
        self.trade_state.last_trade_pnl = pnl
    
    # =========================================================================
    # Layer 3: Position Sizing
    # =========================================================================
    def calculate_position_size(self, confidence: float, regime: str, 
                                 trend_aligned: bool, volatility: float) -> float:
        """Calculate position size using volatility targeting."""
        cfg = self.config['position']
        threshold = cfg['confidence_threshold'].get(regime, 0.80)
        
        if confidence < threshold:
            return 0.0
        
        # Base: scale from 10% to 50% based on confidence
        base = 0.1 + (confidence - threshold) * (0.4 / (1.0 - threshold + 0.01))
        
        # Regime multiplier
        regime_mult = {'trending': 1.5, 'ranging': 0.5, 'volatile': 0.0}.get(regime, 0.5)
        
        # Trend alignment bonus
        trend_mult = 1.2 if trend_aligned else 0.5
        
        # Volatility targeting
        vol_mult = min(1.0, cfg['vol_target'] / (volatility + 0.001))
        
        return min(cfg['max_size'], base * regime_mult * trend_mult * vol_mult)
    
    # =========================================================================
    # Layer 4: Stop-Loss & Drawdown
    # =========================================================================
    def calculate_stop_loss(self, entry_price: float, atr: float, is_long: bool) -> float:
        """Calculate ATR-based stop-loss price."""
        distance = self.config['risk']['stop_loss_atr'] * atr
        return entry_price - distance if is_long else entry_price + distance
    
    def check_stop_loss(self, current_price: float) -> Tuple[bool, str]:
        """Check if stop-loss should be triggered."""
        if self.position.size == 0:
            return False, ""
        if self.position.size > 0 and current_price <= self.position.stop_loss:
            return True, "Long stop hit"
        if self.position.size < 0 and current_price >= self.position.stop_loss:
            return True, "Short stop hit"
        return False, ""
    
    def check_drawdown(self) -> Tuple[bool, str]:
        """Check drawdown and manage halt state."""
        cfg = self.config['risk']
        if self.current_equity > self.equity_peak:
            self.equity_peak = self.current_equity
        drawdown = (self.equity_peak - self.current_equity) / self.equity_peak
        
        if self.halted:
            if drawdown <= cfg['drawdown_recovery']:
                self.halted = False
                return False, "Resumed from halt"
            return True, f"Halted (DD={drawdown:.1%})"
        
        if drawdown >= cfg['max_drawdown_halt']:
            self.halted = True
            return True, f"Halt triggered (DD={drawdown:.1%})"
        return False, ""
    
    def update_equity(self, equity: float):
        """Update current equity for drawdown tracking."""
        self.current_equity = equity
    
    # =========================================================================
    # Layer 5: Signal Generation (XGBoost)
    # =========================================================================
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XStratV1':
        """Train XGBoost model for direction prediction."""
        self.model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective='multi:softprob',
            num_class=3, eval_metric='mlogloss', tree_method='hist', verbosity=0
        )
        self.model.fit(X, y)
        return self
    
    def predict_signal(self, X: np.ndarray) -> Tuple[int, float]:
        """Get signal and confidence from model."""
        if self.model is None:
            return 1, 0.33  # Hold with low confidence
        proba = self.model.predict_proba(X)
        signal = int(np.argmax(proba, axis=1)[0])
        confidence = float(np.max(proba))
        return signal, confidence
    
    # =========================================================================
    # Main Entry Point
    # =========================================================================
    def generate_signal(self, X: np.ndarray, prices: np.ndarray, 
                        adx: float, atr: float, volatility: float,
                        now: datetime = None) -> XStratResult:
        """Generate trading signal with all layers applied."""
        now = now or datetime.now()
        
        # Layer 4: Check drawdown halt first
        halted, halt_reason = self.check_drawdown()
        if halted:
            return XStratResult(1, 0.0, None, 'N/A', 'N/A', 0.0, False, halt_reason)
        
        # Layer 4: Check stop-loss
        stop_hit, stop_reason = self.check_stop_loss(prices[-1])
        if stop_hit:
            return XStratResult(1, 0.0, None, 'N/A', 'N/A', 0.0, True, stop_reason)
        
        # Layer 0: Trend filter
        trend = self.get_trend(prices)
        
        # Layer 1: Regime detection
        regime = self.get_regime(adx, volatility)
        
        # Layer 5: Get raw signal
        signal, confidence = self.predict_signal(X)
        
        # Apply trend filter
        if trend == 'long_only' and signal == 0:  # Want short but long-only
            return XStratResult(1, 0.0, None, trend, regime, confidence, False, "Trend filter: long-only")
        if trend == 'short_only' and signal == 2:  # Want long but short-only
            return XStratResult(1, 0.0, None, trend, regime, confidence, False, "Trend filter: short-only")
        
        # Volatile regime: halt
        if regime == 'volatile':
            return XStratResult(1, 0.0, None, trend, regime, confidence, False, "Volatile: halt")
        
        # Layer 2: Trade frequency
        can, freq_reason = self.can_trade(now)
        if not can:
            return XStratResult(1, 0.0, None, trend, regime, confidence, False, freq_reason)
        
        # Layer 3: Position sizing
        trend_aligned = (trend == 'long_only' and signal == 2) or (trend == 'short_only' and signal == 0)
        position_size = self.calculate_position_size(confidence, regime, trend_aligned, volatility)
        
        if position_size == 0:
            return XStratResult(1, 0.0, None, trend, regime, confidence, False, "Below confidence threshold")
        
        # Calculate stop-loss
        stop_loss = self.calculate_stop_loss(prices[-1], atr, signal == 2)
        
        return XStratResult(signal, position_size, stop_loss, trend, regime, confidence, True, "Signal valid")
    
    # =========================================================================
    # Persistence
    # =========================================================================
    def save(self, path: str):
        """Save model to file."""
        joblib.dump({'model': self.model, 'config': self.config}, path)
        print(f"XStrat V1 saved to {path}")
    
    @classmethod
    def load(cls, path: str, config_path: str = None) -> 'XStratV1':
        """Load model from file."""
        data = joblib.load(path)
        strat = cls(config_path or "config/xstrat_v1.yaml")
        strat.model = data['model']
        if 'config' in data:
            strat.config = data['config']
        return strat

