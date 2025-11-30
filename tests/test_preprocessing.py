import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import Preprocessor


@pytest.fixture
def raw_frames():
    ts = pd.date_range("2023-01-01", periods=65, freq="H")
    base = pd.DataFrame(
        {
            "timestamp": ts,
            "open": np.linspace(100, 120, len(ts)),
            "high": np.linspace(101, 121, len(ts)),
            "low": np.linspace(99, 119, len(ts)),
            "close": np.linspace(100, 130, len(ts)),
            "volume": np.linspace(500, 600, len(ts)),
        }
    )
    base.loc[3, "open"] = np.nan
    other = base.copy()
    other["close"] += 1
    return {"binance": base, "coinbase": other}


@pytest.fixture
def aligned(raw_frames):
    return Preprocessor()._align(raw_frames)


def test_minmax_normalization_within_bounds(aligned):
    norm = Preprocessor("minmax")._normalize(aligned)
    cols = [c for c in norm.columns if c.startswith("open")]
    assert norm[cols].ge(0).all().all()
    assert norm[cols].le(1).all().all()


def test_lag_features_created(raw_frames):
    result = Preprocessor().transform(raw_frames)
    assert "close_binance_lag_60" in result.columns
    assert len(result) == 5


def test_transform_cleans_nans(raw_frames):
    result = Preprocessor().transform(raw_frames)
    assert not result.isna().any().any()

