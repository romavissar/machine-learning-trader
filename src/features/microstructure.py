"""Microstructure feature engineering for HFT/optimal execution."""
import numpy as np
import pandas as pd


class MicrostructureFeatures:
    """Order book microstructure features with vectorized implementations."""

    @staticmethod
    def bid_ask_spread(df: pd.DataFrame) -> pd.Series:
        """Spread = ask_price - bid_price."""
        return df["ask_price"] - df["bid_price"]

    @staticmethod
    def bid_ask_imbalance(df: pd.DataFrame) -> pd.Series:
        """Imbalance = (V_bid - V_ask) / (V_bid + V_ask) ∈ [-1, 1]."""
        total = df["bid_volume"] + df["ask_volume"]
        return (df["bid_volume"] - df["ask_volume"]) / total

    @staticmethod
    def signed_volume(df: pd.DataFrame) -> pd.Series:
        """Tick rule: sign(ΔP) * volume. Propagates last sign on zero change."""
        sign = np.sign(df["price"].diff()).replace(0, np.nan).ffill().fillna(1)
        return sign * df["volume"]

    @staticmethod
    def smart_price(df: pd.DataFrame) -> pd.Series:
        """Inverse vol-weighted mid: (P_bid * V_ask + P_ask * V_bid) / (V_bid + V_ask)."""
        total = df["bid_volume"] + df["ask_volume"]
        return (df["bid_price"] * df["ask_volume"] + df["ask_price"] * df["bid_volume"]) / total

    @classmethod
    def compute_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all microstructure features from order book DataFrame."""
        return pd.DataFrame({
            "bid_ask_spread": cls.bid_ask_spread(df),
            "bid_ask_imbalance": cls.bid_ask_imbalance(df),
            "signed_volume": cls.signed_volume(df),
            "smart_price": cls.smart_price(df),
        })

