import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

class TechnicalFeatures:
    def __init__(self, df: pd.DataFrame):
        self.c, self.v = df['close'], df['volume']
        self._macd = MACD(self.c)
        self._bb = BollingerBands(self.c)

    def rsi(self) -> pd.Series: return RSIIndicator(self.c).rsi()
    def macd(self) -> pd.Series: return self._macd.macd()
    def macd_signal(self) -> pd.Series: return self._macd.macd_signal()
    def macd_hist(self) -> pd.Series: return self._macd.macd_diff()
    
    def ma_20(self) -> pd.Series: return SMAIndicator(self.c, 20).sma_indicator()
    def sma_200(self) -> pd.Series: return SMAIndicator(self.c, 200).sma_indicator()
    
    def volatility_20h(self) -> pd.Series: return self.c.pct_change().rolling(20).std()
    
    def bb_upper(self) -> pd.Series: return self._bb.bollinger_hband()
    def bb_mid(self) -> pd.Series: return self._bb.bollinger_mavg()
    def bb_lower(self) -> pd.Series: return self._bb.bollinger_lband()
    
    def volume_trend(self) -> pd.Series: return SMAIndicator(self.v, 20).sma_indicator()

