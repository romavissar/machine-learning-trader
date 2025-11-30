"""Advanced feature engineering for ML trading models.

This module provides 30+ technical indicators organized by category:
- Momentum: RSI, Stochastic RSI, Williams %R, CCI, ROC
- Trend: ADX, MACD, Parabolic SAR, Aroon, Ichimoku
- Volume: OBV, VWAP, MFI, Accumulation/Distribution, CMF
- Volatility: ATR, Bollinger Bands, Keltner Channels, Donchian
- Price Action: Support/Resistance, Pivot Points, Price patterns

All features are calculated using ONLY past data to prevent lookahead bias.
"""
import numpy as np
import pandas as pd
from ta.momentum import (
    RSIIndicator, StochasticOscillator, WilliamsRIndicator,
    ROCIndicator, StochRSIIndicator
)
from ta.trend import (
    MACD, ADXIndicator, CCIIndicator, AroonIndicator,
    PSARIndicator, IchimokuIndicator, SMAIndicator, EMAIndicator
)
from ta.volatility import (
    AverageTrueRange, BollingerBands, KeltnerChannel, DonchianChannel
)
from ta.volume import (
    OnBalanceVolumeIndicator, AccDistIndexIndicator,
    MFIIndicator, ChaikinMoneyFlowIndicator, VolumeWeightedAveragePrice
)
from typing import Optional


class AdvancedFeatures:
    """Comprehensive feature engineering for trading ML models.
    
    Creates 30+ features from OHLCV data, organized by category.
    All features use only past data (no lookahead bias).
    """
    
    def __init__(self, df: pd.DataFrame, include_lagged: bool = True, lag_periods: int = 5):
        """Initialize with OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
            include_lagged: Whether to include lagged versions of features
            lag_periods: Number of lag periods to include
        """
        self.df = df.copy()
        self.o = df['open']
        self.h = df['high']
        self.l = df['low']
        self.c = df['close']
        self.v = df['volume']
        self.include_lagged = include_lagged
        self.lag_periods = lag_periods
        
    def compute_all(self) -> pd.DataFrame:
        """Compute all features and return as DataFrame."""
        features = pd.DataFrame(index=self.df.index)
        
        # Momentum features
        momentum = self.momentum_features()
        features = pd.concat([features, momentum], axis=1)
        
        # Trend features
        trend = self.trend_features()
        features = pd.concat([features, trend], axis=1)
        
        # Volume features
        volume = self.volume_features()
        features = pd.concat([features, volume], axis=1)
        
        # Volatility features
        volatility = self.volatility_features()
        features = pd.concat([features, volatility], axis=1)
        
        # Price action features
        price_action = self.price_action_features()
        features = pd.concat([features, price_action], axis=1)
        
        # Returns features
        returns = self.returns_features()
        features = pd.concat([features, returns], axis=1)
        
        # Add lagged features if requested
        if self.include_lagged:
            lagged = self.create_lagged_features(features)
            features = pd.concat([features, lagged], axis=1)
        
        return features
    
    def momentum_features(self) -> pd.DataFrame:
        """Momentum indicators - measure speed and strength of price movement."""
        features = pd.DataFrame(index=self.df.index)
        
        # RSI (Relative Strength Index) - multiple periods
        features['rsi_14'] = RSIIndicator(self.c, window=14).rsi()
        features['rsi_7'] = RSIIndicator(self.c, window=7).rsi()
        features['rsi_21'] = RSIIndicator(self.c, window=21).rsi()
        
        # Stochastic RSI
        stoch_rsi = StochRSIIndicator(self.c, window=14, smooth1=3, smooth2=3)
        features['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
        features['stoch_rsi_d'] = stoch_rsi.stochrsi_d()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(self.h, self.l, self.c, window=14, smooth_window=3)
        features['stoch_k'] = stoch.stoch()
        features['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        features['williams_r'] = WilliamsRIndicator(self.h, self.l, self.c, lbp=14).williams_r()
        
        # Rate of Change
        features['roc_10'] = ROCIndicator(self.c, window=10).roc()
        features['roc_20'] = ROCIndicator(self.c, window=20).roc()
        
        # CCI (Commodity Channel Index)
        features['cci'] = CCIIndicator(self.h, self.l, self.c, window=20).cci()
        
        return features
    
    def trend_features(self) -> pd.DataFrame:
        """Trend indicators - identify direction and strength of trends."""
        features = pd.DataFrame(index=self.df.index)
        
        # MACD
        macd = MACD(self.c, window_slow=26, window_fast=12, window_sign=9)
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_hist'] = macd.macd_diff()
        
        # ADX (Average Directional Index)
        adx = ADXIndicator(self.h, self.l, self.c, window=14)
        features['adx'] = adx.adx()
        features['adx_pos'] = adx.adx_pos()
        features['adx_neg'] = adx.adx_neg()
        
        # Aroon
        aroon = AroonIndicator(self.h, self.l, window=25)
        features['aroon_up'] = aroon.aroon_up()
        features['aroon_down'] = aroon.aroon_down()
        features['aroon_indicator'] = aroon.aroon_indicator()
        
        # Parabolic SAR
        psar = PSARIndicator(self.h, self.l, self.c, step=0.02, max_step=0.2)
        features['psar'] = psar.psar()
        features['psar_up'] = psar.psar_up()
        features['psar_down'] = psar.psar_down()
        
        # Moving averages and crossovers
        features['sma_10'] = SMAIndicator(self.c, window=10).sma_indicator()
        features['sma_20'] = SMAIndicator(self.c, window=20).sma_indicator()
        features['sma_50'] = SMAIndicator(self.c, window=50).sma_indicator()
        features['sma_200'] = SMAIndicator(self.c, window=200).sma_indicator()
        features['ema_12'] = EMAIndicator(self.c, window=12).ema_indicator()
        features['ema_26'] = EMAIndicator(self.c, window=26).ema_indicator()
        
        # MA ratios (price relative to MA)
        features['price_sma_20_ratio'] = self.c / features['sma_20']
        features['price_sma_50_ratio'] = self.c / features['sma_50']
        features['sma_20_50_ratio'] = features['sma_20'] / features['sma_50']
        
        # Ichimoku Cloud (simplified - key lines only)
        try:
            ichimoku = IchimokuIndicator(self.h, self.l, window1=9, window2=26, window3=52)
            features['ichimoku_a'] = ichimoku.ichimoku_a()
            features['ichimoku_b'] = ichimoku.ichimoku_b()
            features['ichimoku_base'] = ichimoku.ichimoku_base_line()
            features['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        except Exception:
            pass  # Skip if not enough data
        
        return features
    
    def volume_features(self) -> pd.DataFrame:
        """Volume indicators - analyze trading volume patterns."""
        features = pd.DataFrame(index=self.df.index)
        
        # On-Balance Volume
        features['obv'] = OnBalanceVolumeIndicator(self.c, self.v).on_balance_volume()
        
        # OBV change and momentum
        features['obv_change'] = features['obv'].pct_change()
        features['obv_sma_10'] = SMAIndicator(features['obv'], window=10).sma_indicator()
        
        # Accumulation/Distribution
        features['ad'] = AccDistIndexIndicator(self.h, self.l, self.c, self.v).acc_dist_index()
        
        # Money Flow Index
        features['mfi'] = MFIIndicator(self.h, self.l, self.c, self.v, window=14).money_flow_index()
        
        # Chaikin Money Flow
        features['cmf'] = ChaikinMoneyFlowIndicator(self.h, self.l, self.c, self.v, window=20).chaikin_money_flow()
        
        # VWAP (Volume Weighted Average Price) - simplified daily reset not needed for crypto
        try:
            features['vwap'] = VolumeWeightedAveragePrice(self.h, self.l, self.c, self.v, window=14).volume_weighted_average_price()
            features['price_vwap_ratio'] = self.c / features['vwap']
        except Exception:
            pass
        
        # Volume ratios
        vol_sma_20 = SMAIndicator(self.v, window=20).sma_indicator()
        features['volume_sma_20'] = vol_sma_20
        features['volume_ratio'] = self.v / vol_sma_20
        
        # Volume momentum
        features['volume_change'] = self.v.pct_change()
        features['volume_change_5'] = self.v.pct_change(5)
        
        return features
    
    def volatility_features(self) -> pd.DataFrame:
        """Volatility indicators - measure price variability."""
        features = pd.DataFrame(index=self.df.index)
        
        # ATR (Average True Range)
        features['atr'] = AverageTrueRange(self.h, self.l, self.c, window=14).average_true_range()
        features['atr_7'] = AverageTrueRange(self.h, self.l, self.c, window=7).average_true_range()
        
        # ATR percent (normalized by price)
        features['atr_pct'] = features['atr'] / self.c * 100
        
        # Bollinger Bands
        bb = BollingerBands(self.c, window=20, window_dev=2)
        features['bb_upper'] = bb.bollinger_hband()
        features['bb_mid'] = bb.bollinger_mavg()
        features['bb_lower'] = bb.bollinger_lband()
        features['bb_width'] = bb.bollinger_wband()
        features['bb_pct'] = bb.bollinger_pband()
        
        # Keltner Channel
        kc = KeltnerChannel(self.h, self.l, self.c, window=20, window_atr=10)
        features['kc_upper'] = kc.keltner_channel_hband()
        features['kc_mid'] = kc.keltner_channel_mband()
        features['kc_lower'] = kc.keltner_channel_lband()
        features['kc_width'] = kc.keltner_channel_wband()
        features['kc_pct'] = kc.keltner_channel_pband()
        
        # Donchian Channel
        dc = DonchianChannel(self.h, self.l, self.c, window=20)
        features['dc_upper'] = dc.donchian_channel_hband()
        features['dc_mid'] = dc.donchian_channel_mband()
        features['dc_lower'] = dc.donchian_channel_lband()
        features['dc_width'] = dc.donchian_channel_wband()
        features['dc_pct'] = dc.donchian_channel_pband()
        
        # Historical volatility
        features['volatility_10'] = self.c.pct_change().rolling(10).std()
        features['volatility_20'] = self.c.pct_change().rolling(20).std()
        features['volatility_50'] = self.c.pct_change().rolling(50).std()
        
        # Volatility ratio (short-term vs long-term)
        features['vol_ratio_10_50'] = features['volatility_10'] / features['volatility_50']
        
        return features
    
    def price_action_features(self) -> pd.DataFrame:
        """Price action features - patterns and levels."""
        features = pd.DataFrame(index=self.df.index)
        
        # Candle features
        features['body'] = self.c - self.o
        features['body_pct'] = features['body'] / self.o * 100
        features['upper_shadow'] = self.h - np.maximum(self.c, self.o)
        features['lower_shadow'] = np.minimum(self.c, self.o) - self.l
        features['range'] = self.h - self.l
        features['range_pct'] = features['range'] / self.c * 100
        
        # Body to range ratio
        features['body_range_ratio'] = np.abs(features['body']) / (features['range'] + 1e-8)
        
        # High/Low position
        features['close_position'] = (self.c - self.l) / (self.h - self.l + 1e-8)
        
        # Gap features
        features['gap'] = self.o - self.c.shift(1)
        features['gap_pct'] = features['gap'] / self.c.shift(1) * 100
        
        # Pivot points (classic)
        pivot = (self.h.shift(1) + self.l.shift(1) + self.c.shift(1)) / 3
        features['pivot'] = pivot
        features['r1'] = 2 * pivot - self.l.shift(1)
        features['s1'] = 2 * pivot - self.h.shift(1)
        features['r2'] = pivot + (self.h.shift(1) - self.l.shift(1))
        features['s2'] = pivot - (self.h.shift(1) - self.l.shift(1))
        
        # Distance from pivot levels
        features['dist_pivot'] = (self.c - features['pivot']) / features['pivot'] * 100
        features['dist_r1'] = (self.c - features['r1']) / features['r1'] * 100
        features['dist_s1'] = (self.c - features['s1']) / features['s1'] * 100
        
        # Rolling high/low (support/resistance approximation)
        features['rolling_high_20'] = self.h.rolling(20).max()
        features['rolling_low_20'] = self.l.rolling(20).min()
        features['dist_high_20'] = (self.c - features['rolling_high_20']) / features['rolling_high_20'] * 100
        features['dist_low_20'] = (self.c - features['rolling_low_20']) / features['rolling_low_20'] * 100
        
        return features
    
    def returns_features(self) -> pd.DataFrame:
        """Return-based features at multiple horizons."""
        features = pd.DataFrame(index=self.df.index)
        
        # Simple returns
        features['return_1'] = self.c.pct_change(1)
        features['return_3'] = self.c.pct_change(3)
        features['return_5'] = self.c.pct_change(5)
        features['return_10'] = self.c.pct_change(10)
        features['return_20'] = self.c.pct_change(20)
        features['return_50'] = self.c.pct_change(50)
        
        # Log returns
        features['log_return_1'] = np.log(self.c / self.c.shift(1))
        features['log_return_5'] = np.log(self.c / self.c.shift(5))
        
        # Cumulative returns
        features['cum_return_10'] = self.c.pct_change().rolling(10).sum()
        features['cum_return_20'] = self.c.pct_change().rolling(20).sum()
        
        # Return momentum (acceleration)
        features['return_momentum'] = features['return_5'] - features['return_5'].shift(5)
        
        # Skewness and kurtosis of returns
        features['return_skew_20'] = features['return_1'].rolling(20).skew()
        features['return_kurt_20'] = features['return_1'].rolling(20).kurt()
        
        return features
    
    def create_lagged_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create lagged versions of key features."""
        key_features = ['rsi_14', 'macd', 'adx', 'atr_pct', 'bb_pct', 'volume_ratio', 'return_1']
        lagged = pd.DataFrame(index=features.index)
        
        for col in key_features:
            if col in features.columns:
                for lag in range(1, self.lag_periods + 1):
                    lagged[f'{col}_lag_{lag}'] = features[col].shift(lag)
        
        return lagged


def create_features(df: pd.DataFrame, include_lagged: bool = True) -> pd.DataFrame:
    """Convenience function to create all features from OHLCV data.
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume (and timestamp index)
        include_lagged: Whether to include lagged features
        
    Returns:
        DataFrame with 30+ technical features
    """
    af = AdvancedFeatures(df, include_lagged=include_lagged)
    features = af.compute_all()
    
    # Replace infinities with NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    
    return features


def get_feature_names() -> list:
    """Get list of all feature names (approximate)."""
    return [
        # Momentum
        'rsi_14', 'rsi_7', 'rsi_21', 'stoch_rsi_k', 'stoch_rsi_d',
        'stoch_k', 'stoch_d', 'williams_r', 'roc_10', 'roc_20', 'cci',
        # Trend
        'macd', 'macd_signal', 'macd_hist', 'adx', 'adx_pos', 'adx_neg',
        'aroon_up', 'aroon_down', 'aroon_indicator', 'psar', 'psar_up', 'psar_down',
        'sma_10', 'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
        'price_sma_20_ratio', 'price_sma_50_ratio', 'sma_20_50_ratio',
        # Volume
        'obv', 'obv_change', 'ad', 'mfi', 'cmf', 'vwap', 'price_vwap_ratio',
        'volume_sma_20', 'volume_ratio', 'volume_change', 'volume_change_5',
        # Volatility
        'atr', 'atr_7', 'atr_pct', 'bb_upper', 'bb_mid', 'bb_lower', 'bb_width', 'bb_pct',
        'kc_upper', 'kc_mid', 'kc_lower', 'kc_width', 'kc_pct',
        'volatility_10', 'volatility_20', 'volatility_50', 'vol_ratio_10_50',
        # Price action
        'body', 'body_pct', 'upper_shadow', 'lower_shadow', 'range', 'range_pct',
        'body_range_ratio', 'close_position', 'gap', 'gap_pct',
        'pivot', 'r1', 's1', 'r2', 's2', 'dist_pivot', 'dist_r1', 'dist_s1',
        'rolling_high_20', 'rolling_low_20', 'dist_high_20', 'dist_low_20',
        # Returns
        'return_1', 'return_3', 'return_5', 'return_10', 'return_20', 'return_50',
        'log_return_1', 'log_return_5', 'cum_return_10', 'cum_return_20',
        'return_momentum', 'return_skew_20', 'return_kurt_20',
    ]

