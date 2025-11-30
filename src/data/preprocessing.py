from typing import Dict, Literal

import pandas as pd


class Preprocessor:
    def __init__(self, normalization: Literal["zscore", "minmax"] = "zscore") -> None:
        self.normalization = normalization

    def transform(self, frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return self._align(frames).pipe(self._clean).pipe(self._normalize).pipe(self._lag)

    def _align(self, frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # Concat with exchange as a column (pandas 2.x compatible)
        dfs = []
        for name, df in frames.items():
            frame = df.copy()
            frame["exchange"] = name
            dfs.append(frame)
        base = pd.concat(dfs, ignore_index=True)
        base["timestamp"] = pd.to_datetime(base["timestamp"])
        wide = (
            base.pivot_table(index="timestamp", columns="exchange")
            .sort_index()
            .interpolate(method="time")
            .ffill()
            .bfill()
        )
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        return wide

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        filled = df.sort_index().ffill().bfill()
        low, high = filled.quantile(0.01), filled.quantile(0.99)
        return filled.clip(lower=low, upper=high, axis=1)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in df.columns if c.startswith(("open", "high", "low", "close"))]
        if not cols:
            return df
        normed = df.copy()
        if self.normalization == "minmax":
            mins, maxs = normed[cols].min(), normed[cols].max()
            denom = (maxs - mins).replace(0, 1)
            normed[cols] = (normed[cols] - mins) / denom
        else:
            means, stds = normed[cols].mean(), normed[cols].std().replace(0, 1)
            normed[cols] = (normed[cols] - means) / stds
        return normed

    def _lag(self, df: pd.DataFrame) -> pd.DataFrame:
        returns = df.filter(like="close_").pct_change().add_prefix("ret_")
        base = df.join(returns)
        lag_cols = [c for c in base.columns if c.startswith(("open", "high", "low", "close", "volume", "ret_"))]
        lags = pd.concat(
            {f"{c}_lag_{i}": base[c].shift(i) for c in lag_cols for i in range(1, 61)},
            axis=1,
        )
        return base.join(lags).dropna()

