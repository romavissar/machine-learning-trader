"""Technical feature engineering module.

For basic features, use TechnicalFeatures class.
For comprehensive 30+ features, use AdvancedFeatures from advanced.py.
"""
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from .advanced import AdvancedFeatures, create_features


class TechnicalFeatures:
    """Basic technical indicators for quick feature generation."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.c, self.v = df['close'], df['volume']
        self.h = df.get('high', df['close'])
        self.l = df.get('low', df['close'])
        self._macd = MACD(self.c)
        self._bb = BollingerBands(self.c)
        self._adx = ADXIndicator(self.h, self.l, self.c, window=14)
        self._atr = AverageTrueRange(self.h, self.l, self.c, window=14)

    def rsi(self) -> pd.Series: return RSIIndicator(self.c).rsi()
    def macd(self) -> pd.Series: return self._macd.macd()
    def macd_signal(self) -> pd.Series: return self._macd.macd_signal()
    def macd_hist(self) -> pd.Series: return self._macd.macd_diff()
    
    def ma_20(self) -> pd.Series: return SMAIndicator(self.c, 20).sma_indicator()
    def sma_200(self) -> pd.Series: return SMAIndicator(self.c, 200).sma_indicator()
    
    def volatility_20h(self) -> pd.Series: return self.c.pct_change().rolling(20).std()
    
    def adx(self, period: int = 14) -> pd.Series:
        """Average Directional Index - measures trend strength.
        
        ADX > 25: Strong trend (trending market)
        ADX < 20: Weak trend (ranging market)
        """
        if period == 14:
            return self._adx.adx()
        return ADXIndicator(self.h, self.l, self.c, window=period).adx()
    
    def adx_pos(self) -> pd.Series:
        """Positive Directional Indicator (+DI)."""
        return self._adx.adx_pos()
    
    def adx_neg(self) -> pd.Series:
        """Negative Directional Indicator (-DI)."""
        return self._adx.adx_neg()
    
    def atr(self, period: int = 14) -> pd.Series:
        """Average True Range - measures volatility."""
        if period == 14:
            return self._atr.average_true_range()
        return AverageTrueRange(self.h, self.l, self.c, window=period).average_true_range()
    
    def historical_volatility(self, period: int = 20) -> pd.Series:
        """Annualized historical volatility based on log returns.
        
        Used for regime detection: Vol > 2*median indicates high volatility regime.
        """
        log_returns = np.log(self.c / self.c.shift(1))
        rolling_std = log_returns.rolling(window=period).std()
        # Annualize (assuming hourly data, ~8760 hours/year)
        return rolling_std * np.sqrt(8760)
    
    def volatility_regime(self, period: int = 20, lookback: int = 100) -> pd.Series:
        """Detect volatility regime: 'normal', 'high', or 'low'.
        
        Compares current volatility to rolling median.
        """
        vol = self.historical_volatility(period)
        vol_median = vol.rolling(lookback, min_periods=period).median()
        
        regime = pd.Series(index=self.df.index, dtype='object')
        regime[vol > 2 * vol_median] = 'high'
        regime[vol < 0.5 * vol_median] = 'low'
        regime[(vol >= 0.5 * vol_median) & (vol <= 2 * vol_median)] = 'normal'
        return regime.fillna('normal')
    
    def market_regime(self, adx_period: int = 14, vol_period: int = 20) -> pd.Series:
        """Combined market regime detection for model selection.
        
        Returns: 'trending', 'ranging', or 'volatile'
        - ADX > 25: trending
        - ADX < 20 and normal vol: ranging  
        - High volatility (>2σ): volatile
        """
        adx_values = self.adx(adx_period)
        vol_regime = self.volatility_regime(vol_period)
        
        regime = pd.Series(index=self.df.index, dtype='object')
        # High volatility takes priority
        regime[vol_regime == 'high'] = 'volatile'
        # Then check ADX for trend
        regime[(vol_regime != 'high') & (adx_values > 25)] = 'trending'
        # Default to ranging
        regime[(vol_regime != 'high') & (adx_values <= 25)] = 'ranging'
        return regime.fillna('ranging')
    
    def bb_upper(self) -> pd.Series: return self._bb.bollinger_hband()
    def bb_mid(self) -> pd.Series: return self._bb.bollinger_mavg()
    def bb_lower(self) -> pd.Series: return self._bb.bollinger_lband()
    
    def volume_trend(self) -> pd.Series: return SMAIndicator(self.v, 20).sma_indicator()
    
    def all_features(self) -> pd.DataFrame:
        """Get all basic technical features as DataFrame."""
        features = pd.DataFrame(index=self.df.index)
        features['rsi'] = self.rsi()
        features['macd'] = self.macd()
        features['macd_signal'] = self.macd_signal()
        features['macd_hist'] = self.macd_hist()
        features['ma_20'] = self.ma_20()
        features['sma_200'] = self.sma_200()
        features['volatility_20h'] = self.volatility_20h()
        features['bb_upper'] = self.bb_upper()
        features['bb_mid'] = self.bb_mid()
        features['bb_lower'] = self.bb_lower()
        features['volume_trend'] = self.volume_trend()
        features['adx'] = self.adx()
        features['adx_pos'] = self.adx_pos()
        features['adx_neg'] = self.adx_neg()
        features['atr'] = self.atr()
        features['historical_volatility'] = self.historical_volatility()
        return features
    
    def advanced_features(self, include_lagged: bool = True) -> pd.DataFrame:
        """Get comprehensive 30+ advanced features."""
        return create_features(self.df, include_lagged=include_lagged)

