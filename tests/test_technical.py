import numpy as np
import pandas as pd
import pytest

from src.features.technical import TechnicalFeatures


@pytest.fixture
def price_frame():
    closes = pd.Series(np.linspace(100, 150, 60))
    volume = pd.Series(np.linspace(1_000, 1_600, 60))
    return pd.DataFrame({"close": closes, "volume": volume})


def test_rsi_reaches_hundred_on_monotonic_gains(price_frame):
    rsi = TechnicalFeatures(price_frame).rsi().dropna()
    assert rsi.iloc[-1] == pytest.approx(100.0, rel=1e-3)


def test_macd_matches_manual_ema(price_frame):
    tf = TechnicalFeatures(price_frame)
    ema_fast = price_frame["close"].ewm(span=12, adjust=False).mean()
    ema_slow = price_frame["close"].ewm(span=26, adjust=False).mean()
    expected = ema_fast - ema_slow
    pd.testing.assert_series_equal(
        tf.macd().dropna(),
        expected.dropna(),
        check_exact=False,
        rtol=1e-6,
    )

