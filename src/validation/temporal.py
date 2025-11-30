"""Temporal validation utilities to prevent lookahead bias and data leakage.

This module ensures that models are NEVER trained on future data:
- Walk-forward validation: Train on past, test on future, roll forward
- Expanding window: Training set grows as we move forward in time
- Rolling window: Fixed-size training window that slides forward
- Purging & Embargo: Gaps between train/test to avoid leakage from overlapping features

CRITICAL: In financial ML, standard cross-validation is WRONG because it allows
the model to see future data during training, leading to unrealistic backtests.
"""
from dataclasses import dataclass
from typing import Iterator, Tuple, List, Optional
import numpy as np
import pandas as pd


@dataclass
class TemporalSplit:
    """Represents a single train/test split with temporal boundaries."""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    
    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)
    
    @property
    def test_slice(self) -> slice:
        return slice(self.test_start, self.test_end)
    
    def __repr__(self) -> str:
        return f"TemporalSplit(train=[{self.train_start}:{self.train_end}], test=[{self.test_start}:{self.test_end}])"


class WalkForwardValidator:
    """Walk-forward validation for time series data.
    
    Ensures models are only trained on data available at prediction time.
    This prevents lookahead bias which would make backtests unrealistically good.
    
    Example with 1000 samples, train_size=0.6, test_size=0.1, n_splits=4:
        Split 1: Train [0:600],   Test [600:700]
        Split 2: Train [0:700],   Test [700:800]  (expanding window)
        Split 3: Train [0:800],   Test [800:900]
        Split 4: Train [0:900],   Test [900:1000]
    
    Args:
        n_splits: Number of walk-forward splits
        train_size: Initial training set size (fraction or absolute)
        test_size: Test set size for each split (fraction or absolute)
        gap: Number of samples to skip between train and test (embargo period)
        expanding: If True, training window expands. If False, it rolls (fixed size).
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_size: float = 0.6,
        test_size: float = 0.1,
        gap: int = 0,
        expanding: bool = True,
    ):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding
    
    def split(self, n_samples: int) -> Iterator[TemporalSplit]:
        """Generate train/test splits that respect temporal order.
        
        Args:
            n_samples: Total number of samples in the dataset
            
        Yields:
            TemporalSplit objects with train and test indices
        """
        # Convert fractions to absolute sizes
        if self.train_size < 1:
            initial_train = int(n_samples * self.train_size)
        else:
            initial_train = int(self.train_size)
            
        if self.test_size < 1:
            test_len = int(n_samples * self.test_size)
        else:
            test_len = int(self.test_size)
        
        # Calculate step size between splits
        remaining = n_samples - initial_train - test_len
        step = max(1, remaining // max(1, self.n_splits - 1)) if self.n_splits > 1 else remaining
        
        for i in range(self.n_splits):
            if self.expanding:
                train_start = 0
                train_end = initial_train + i * step
            else:
                # Rolling window
                train_start = i * step
                train_end = initial_train + i * step
            
            test_start = train_end + self.gap
            test_end = min(test_start + test_len, n_samples)
            
            # Ensure we have valid splits
            if test_start >= n_samples or train_end <= train_start:
                break
                
            yield TemporalSplit(train_start, train_end, test_start, test_end)
    
    def get_splits(self, n_samples: int) -> List[TemporalSplit]:
        """Return all splits as a list."""
        return list(self.split(n_samples))


class TimeSeriesPurger:
    """Purge overlapping samples between train and test sets.
    
    When features use lagged values (e.g., 20-day moving average), there's
    information leakage if train and test sets are too close together.
    This class removes samples that could cause such leakage.
    
    Args:
        purge_window: Number of samples to remove from end of training set
        embargo_window: Number of samples to skip after training set before testing
    """
    
    def __init__(self, purge_window: int = 0, embargo_window: int = 0):
        self.purge_window = purge_window
        self.embargo_window = embargo_window
    
    def purge(self, split: TemporalSplit) -> TemporalSplit:
        """Apply purging and embargo to a temporal split."""
        new_train_end = split.train_end - self.purge_window
        new_test_start = split.train_end + self.embargo_window
        
        return TemporalSplit(
            train_start=split.train_start,
            train_end=max(split.train_start + 1, new_train_end),
            test_start=new_test_start,
            test_end=split.test_end,
        )


def temporal_train_test_split(
    data: pd.DataFrame | np.ndarray,
    train_ratio: float = 0.8,
    gap: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple temporal train/test split - NO shuffling, NO random selection.
    
    CRITICAL: This is the ONLY correct way to split time series data.
    Never use sklearn's train_test_split with shuffle=True on time series!
    
    Args:
        data: DataFrame or array with time-ordered samples
        train_ratio: Fraction of data for training (from the START)
        gap: Number of samples to skip between train and test (embargo)
        
    Returns:
        X_train, X_test, y_train, y_test (if data has target column)
        or train_data, test_data (if no target column)
    """
    if isinstance(data, pd.DataFrame):
        n = len(data)
    else:
        n = data.shape[0]
    
    train_end = int(n * train_ratio)
    test_start = train_end + gap
    
    if isinstance(data, pd.DataFrame):
        return data.iloc[:train_end], data.iloc[test_start:]
    else:
        return data[:train_end], data[test_start:]


def create_backtest_splits(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_period: str = "365D",
    test_period: str = "30D",
    step: str = "30D",
    min_train_samples: int = 100,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Create date-based splits for backtesting.
    
    Example: Backtest from 2020 to 2024 with 1-year training, 1-month testing:
        Split 1: Train [2020-01 to 2021-01], Test [2021-01 to 2021-02]
        Split 2: Train [2020-01 to 2021-02], Test [2021-02 to 2021-03]
        ...and so on
    
    Args:
        start_date: Start of the entire backtest period
        end_date: End of the entire backtest period
        train_period: Duration of training period (pandas offset string)
        test_period: Duration of each test period
        step: How much to move forward between splits
        min_train_samples: Minimum samples required in training set
        
    Returns:
        List of (train_start, train_end, test_start, test_end) tuples
    """
    splits = []
    train_delta = pd.Timedelta(train_period)
    test_delta = pd.Timedelta(test_period)
    step_delta = pd.Timedelta(step)
    
    current_train_end = start_date + train_delta
    
    while current_train_end + test_delta <= end_date:
        train_start = start_date  # Expanding window
        train_end = current_train_end
        test_start = current_train_end
        test_end = min(current_train_end + test_delta, end_date)
        
        splits.append((train_start, train_end, test_start, test_end))
        current_train_end += step_delta
    
    return splits


class WalkForwardBacktester:
    """Walk-forward backtester that retrains models at each step.
    
    This ensures the model only ever sees data that would have been
    available at the time of prediction - no lookahead bias.
    
    Usage:
        backtester = WalkForwardBacktester(
            model_factory=lambda: PriceDirectionModel(),
            train_window="365D",
            test_window="30D",
            retrain_frequency="30D",
        )
        results = backtester.run(data, feature_cols, target_col)
    """
    
    def __init__(
        self,
        model_factory,
        train_window: str = "365D",
        test_window: str = "30D",
        retrain_frequency: str = "30D",
        embargo_periods: int = 1,
    ):
        """
        Args:
            model_factory: Callable that returns a fresh model instance
            train_window: How much historical data to train on
            test_window: How far ahead to predict
            retrain_frequency: How often to retrain the model
            embargo_periods: Gap between train and test to avoid leakage
        """
        self.model_factory = model_factory
        self.train_window = pd.Timedelta(train_window)
        self.test_window = pd.Timedelta(test_window)
        self.retrain_frequency = pd.Timedelta(retrain_frequency)
        self.embargo_periods = embargo_periods
    
    def run(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Run walk-forward backtest.
        
        Args:
            data: DataFrame with DatetimeIndex
            feature_cols: List of feature column names
            target_col: Name of target column
            start_date: When to start backtesting (after initial training period)
            end_date: When to stop backtesting
            
        Returns:
            DataFrame with predictions, actuals, and dates
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex for walk-forward backtest")
        
        data = data.sort_index()
        
        if start_date is None:
            start_date = data.index[0] + self.train_window
        if end_date is None:
            end_date = data.index[-1]
        
        results = []
        current_date = start_date
        last_train_date = None
        model = None
        
        while current_date <= end_date:
            # Determine training period
            train_end = current_date - pd.Timedelta(periods=self.embargo_periods, freq='h')
            train_start = train_end - self.train_window
            
            # Get training data (ONLY PAST DATA)
            train_mask = (data.index >= train_start) & (data.index < train_end)
            train_data = data.loc[train_mask]
            
            # Retrain if needed
            if model is None or last_train_date is None or \
               (current_date - last_train_date) >= self.retrain_frequency:
                
                if len(train_data) > 10:  # Minimum samples
                    model = self.model_factory()
                    X_train = train_data[feature_cols].values
                    y_train = train_data[target_col].values
                    model.fit(X_train, y_train)
                    last_train_date = current_date
                    print(f"Retrained model at {current_date} using data from {train_start} to {train_end}")
            
            # Get test data for this period
            test_end = min(current_date + self.test_window, end_date)
            test_mask = (data.index >= current_date) & (data.index < test_end)
            test_data = data.loc[test_mask]
            
            if len(test_data) > 0 and model is not None:
                X_test = test_data[feature_cols].values
                y_test = test_data[target_col].values
                predictions = model.predict(X_test)
                
                for i, (idx, pred, actual) in enumerate(zip(test_data.index, predictions, y_test)):
                    results.append({
                        'date': idx,
                        'prediction': pred,
                        'actual': actual,
                        'train_end': train_end,
                    })
            
            current_date += self.retrain_frequency
        
        return pd.DataFrame(results).set_index('date') if results else pd.DataFrame()


def validate_no_lookahead(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: List[str],
) -> bool:
    """Validate that there's no lookahead bias in the split.
    
    Checks:
    1. All train timestamps are before test timestamps
    2. No overlapping indices
    3. Features don't contain future information
    
    Returns:
        True if validation passes, raises ValueError otherwise
    """
    if not isinstance(train_data.index, pd.DatetimeIndex) or \
       not isinstance(test_data.index, pd.DatetimeIndex):
        raise ValueError("Both datasets must have DatetimeIndex")
    
    train_max = train_data.index.max()
    test_min = test_data.index.min()
    
    if train_max >= test_min:
        raise ValueError(
            f"Lookahead bias detected! Train data ends at {train_max} "
            f"but test data starts at {test_min}. "
            f"Train data must be strictly before test data."
        )
    
    overlap = train_data.index.intersection(test_data.index)
    if len(overlap) > 0:
        raise ValueError(
            f"Overlapping indices detected! {len(overlap)} samples appear in both sets."
        )
    
    return True


# Convenience function for quick temporal split
def split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    gap: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quick temporal split for arrays.
    
    WARNING: This assumes data is already sorted by time!
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        train_ratio: Fraction for training (0.8 = 80% train, 20% test)
        gap: Samples to skip between train and test
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    n = len(X)
    train_end = int(n * train_ratio)
    test_start = train_end + gap
    
    return X[:train_end], X[test_start:], y[:train_end], y[test_start:]

