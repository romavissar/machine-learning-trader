"""Monte Carlo simulation for strategy variance analysis.

Runs multiple backtest simulations with random noise in model signals
to generate confidence bands and assess strategy robustness.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation results."""
    equity_curves: np.ndarray  # (n_simulations, n_timesteps)
    final_values: np.ndarray  # (n_simulations,)
    returns: np.ndarray  # (n_simulations, n_timesteps)
    sharpe_ratios: np.ndarray  # (n_simulations,)
    max_drawdowns: np.ndarray  # (n_simulations,)
    percentiles: Dict[int, np.ndarray]  # {percentile: equity_curve}
    actual_equity: Optional[np.ndarray] = None  # Actual backtest result
    timestamps: Optional[np.ndarray] = None


class MonteCarloBacktest:
    """Monte Carlo simulation for strategy variance analysis.
    
    Adds noise to model confidence scores and occasionally flips
    low-confidence signals to simulate real-world uncertainty.
    
    Usage:
        mc = MonteCarloBacktest(orchestrator, n_simulations=1000)
        results = mc.run(features, prices)
        bands = results.percentiles
    """
    
    def __init__(
        self,
        signal_generator: Callable,
        n_simulations: int = 1000,
        noise_std: float = 0.1,
        confidence_flip_threshold: float = 0.55,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        random_seed: Optional[int] = None,
    ):
        """Initialize Monte Carlo backtester.
        
        Args:
            signal_generator: Callable that takes (features, idx) and returns (signal, confidence)
            n_simulations: Number of Monte Carlo simulations
            noise_std: Standard deviation of Gaussian noise added to confidence
            confidence_flip_threshold: Signals with confidence below this may flip
            initial_capital: Starting capital for each simulation
            transaction_cost: Transaction cost as fraction (e.g., 0.001 = 0.1%)
            random_seed: Random seed for reproducibility
        """
        self.signal_generator = signal_generator
        self.n_simulations = n_simulations
        self.noise_std = noise_std
        self.confidence_flip_threshold = confidence_flip_threshold
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def _add_noise_to_signal(
        self,
        signal: int,
        confidence: float,
        rng: np.random.Generator,
    ) -> Tuple[int, float]:
        """Add noise to signal and potentially flip low-confidence signals.
        
        Args:
            signal: Original signal (0=down, 1=hold, 2=up)
            confidence: Original confidence (0-1)
            rng: Random number generator
            
        Returns:
            Tuple of (noisy_signal, noisy_confidence)
        """
        # Add Gaussian noise to confidence
        noisy_confidence = confidence + rng.normal(0, self.noise_std)
        noisy_confidence = np.clip(noisy_confidence, 0.0, 1.0)
        
        # Potentially flip signal if confidence is low
        if confidence < self.confidence_flip_threshold:
            flip_prob = (self.confidence_flip_threshold - confidence) * 0.5
            if rng.random() < flip_prob:
                # Flip signal
                if signal == 0:
                    signal = rng.choice([1, 2])
                elif signal == 2:
                    signal = rng.choice([0, 1])
                else:  # signal == 1
                    signal = rng.choice([0, 2])
        
        return signal, noisy_confidence
    
    def _run_single_simulation(
        self,
        returns: np.ndarray,
        base_signals: np.ndarray,
        base_confidences: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float, float]:
        """Run a single Monte Carlo simulation.
        
        Args:
            returns: Asset returns array
            base_signals: Original signals from model
            base_confidences: Original confidences from model
            rng: Random number generator
            
        Returns:
            Tuple of (equity_curve, sharpe_ratio, max_drawdown)
        """
        n_steps = len(returns)
        equity = np.zeros(n_steps + 1)
        equity[0] = self.initial_capital
        
        position = 0.0
        strategy_returns = []
        
        for i in range(n_steps):
            # Add noise to signal
            noisy_signal, noisy_conf = self._add_noise_to_signal(
                base_signals[i], base_confidences[i], rng
            )
            
            # Convert signal to position
            if noisy_signal == 2:
                new_position = 1.0 * noisy_conf
            elif noisy_signal == 0:
                new_position = -1.0 * noisy_conf
            else:
                new_position = 0.0
            
            # Apply transaction costs
            position_change = abs(new_position - position)
            tc = position_change * self.transaction_cost
            
            # Calculate return
            step_return = position * returns[i] - tc
            strategy_returns.append(step_return)
            
            # Update equity
            equity[i + 1] = equity[i] * (1 + step_return)
            position = new_position
        
        # Calculate metrics
        strategy_returns = np.array(strategy_returns)
        sharpe = self._calculate_sharpe(strategy_returns)
        max_dd = self._calculate_max_drawdown(equity)
        
        return equity[1:], sharpe, max_dd
    
    def _calculate_sharpe(self, returns: np.ndarray, annualization: float = np.sqrt(8760)) -> float:
        """Calculate Sharpe ratio (annualized for hourly data)."""
        if len(returns) < 2 or np.std(returns) < 1e-8:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * annualization)
    
    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """Calculate maximum drawdown as a positive fraction."""
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.where(peak > 0, peak, 1.0)
        return float(np.max(drawdown))
    
    def run(
        self,
        returns: np.ndarray,
        base_signals: np.ndarray,
        base_confidences: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        actual_equity: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulations.
        
        Args:
            returns: Asset returns array (n_timesteps,)
            base_signals: Original signals from model (n_timesteps,)
            base_confidences: Original confidences from model (n_timesteps,)
            timestamps: Optional timestamps for results
            actual_equity: Optional actual backtest equity curve for comparison
            show_progress: Whether to show progress bar
            
        Returns:
            MonteCarloResult with all simulation data
        """
        n_steps = len(returns)
        
        # Storage for results
        all_equity = np.zeros((self.n_simulations, n_steps))
        all_sharpe = np.zeros(self.n_simulations)
        all_max_dd = np.zeros(self.n_simulations)
        
        # Run simulations
        iterator = range(self.n_simulations)
        if show_progress:
            iterator = tqdm(iterator, desc="Monte Carlo simulations")
        
        for sim_idx in iterator:
            rng = np.random.default_rng(seed=sim_idx)  # Reproducible per-simulation
            
            equity, sharpe, max_dd = self._run_single_simulation(
                returns, base_signals, base_confidences, rng
            )
            
            all_equity[sim_idx] = equity
            all_sharpe[sim_idx] = sharpe
            all_max_dd[sim_idx] = max_dd
        
        # Calculate returns from equity curves
        all_returns = np.diff(all_equity, axis=1) / all_equity[:, :-1]
        all_returns = np.hstack([np.zeros((self.n_simulations, 1)), all_returns])
        
        # Final values
        final_values = all_equity[:, -1]
        
        # Calculate percentiles
        percentiles = self.compute_confidence_bands(all_equity)
        
        return MonteCarloResult(
            equity_curves=all_equity,
            final_values=final_values,
            returns=all_returns,
            sharpe_ratios=all_sharpe,
            max_drawdowns=all_max_dd,
            percentiles=percentiles,
            actual_equity=actual_equity,
            timestamps=timestamps,
        )
    
    def compute_confidence_bands(
        self,
        equity_curves: np.ndarray,
        percentiles: List[int] = [5, 25, 50, 75, 95],
    ) -> Dict[int, np.ndarray]:
        """Compute percentile bands from equity curves.
        
        Args:
            equity_curves: Array of shape (n_simulations, n_timesteps)
            percentiles: List of percentiles to compute
            
        Returns:
            Dictionary mapping percentile to equity curve
        """
        result = {}
        for p in percentiles:
            result[p] = np.percentile(equity_curves, p, axis=0)
        return result
    
    def summary_statistics(self, results: MonteCarloResult) -> Dict[str, float]:
        """Calculate summary statistics from Monte Carlo results.
        
        Args:
            results: MonteCarloResult from run()
            
        Returns:
            Dictionary of summary statistics
        """
        return {
            'mean_final_value': float(np.mean(results.final_values)),
            'std_final_value': float(np.std(results.final_values)),
            'median_final_value': float(np.median(results.final_values)),
            'p5_final_value': float(np.percentile(results.final_values, 5)),
            'p95_final_value': float(np.percentile(results.final_values, 95)),
            'mean_sharpe': float(np.mean(results.sharpe_ratios)),
            'std_sharpe': float(np.std(results.sharpe_ratios)),
            'mean_max_drawdown': float(np.mean(results.max_drawdowns)),
            'max_max_drawdown': float(np.max(results.max_drawdowns)),
            'prob_profit': float(np.mean(results.final_values > self.initial_capital)),
            'prob_loss_10pct': float(np.mean(results.final_values < self.initial_capital * 0.9)),
        }


def run_monte_carlo_analysis(
    orchestrator,
    features: np.ndarray,
    prices: np.ndarray,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    n_simulations: int = 1000,
    initial_capital: float = 10000.0,
    show_progress: bool = True,
) -> MonteCarloResult:
    """Convenience function to run Monte Carlo analysis on orchestrator.
    
    Args:
        orchestrator: ModelOrchestrator instance
        features: Feature matrix (n_samples, n_features)
        prices: Price array for returns calculation
        high: High prices for regime detection
        low: Low prices for regime detection
        timestamps: Optional timestamps
        n_simulations: Number of simulations
        initial_capital: Starting capital
        show_progress: Show progress bar
        
    Returns:
        MonteCarloResult with simulation data
    """
    # Calculate returns
    returns = np.diff(prices) / prices[:-1]
    
    # Generate base signals from orchestrator
    n_samples = len(features)
    base_signals = np.zeros(n_samples - 1, dtype=int)
    base_confidences = np.zeros(n_samples - 1)
    
    print("Generating base signals from orchestrator...")
    for i in tqdm(range(n_samples - 1), disable=not show_progress, desc="Base signals"):
        # Need enough history for regime detection
        lookback = min(i + 1, 50)
        price_history = prices[i - lookback + 1:i + 1]
        high_history = high[i - lookback + 1:i + 1] if high is not None else None
        low_history = low[i - lookback + 1:i + 1] if low is not None else None
        
        result = orchestrator.generate_signal(
            X=features[i:i + 1],
            prices=price_history,
            high=high_history,
            low=low_history,
        )
        
        base_signals[i] = result.signal
        base_confidences[i] = result.confidence
    
    # Run actual backtest
    actual_equity = _run_actual_backtest(returns, base_signals, base_confidences, initial_capital)
    
    # Run Monte Carlo
    mc = MonteCarloBacktest(
        signal_generator=None,  # Not used when running directly
        n_simulations=n_simulations,
        initial_capital=initial_capital,
    )
    
    results = mc.run(
        returns=returns,
        base_signals=base_signals,
        base_confidences=base_confidences,
        timestamps=timestamps[1:] if timestamps is not None else None,
        actual_equity=actual_equity,
        show_progress=show_progress,
    )
    
    return results


def _run_actual_backtest(
    returns: np.ndarray,
    signals: np.ndarray,
    confidences: np.ndarray,
    initial_capital: float,
    transaction_cost: float = 0.001,
) -> np.ndarray:
    """Run actual backtest without noise for comparison."""
    n_steps = len(returns)
    equity = np.zeros(n_steps + 1)
    equity[0] = initial_capital
    
    position = 0.0
    
    for i in range(n_steps):
        # Convert signal to position
        if signals[i] == 2:
            new_position = 1.0 * confidences[i]
        elif signals[i] == 0:
            new_position = -1.0 * confidences[i]
        else:
            new_position = 0.0
        
        # Apply transaction costs
        position_change = abs(new_position - position)
        tc = position_change * transaction_cost
        
        # Calculate return
        step_return = position * returns[i] - tc
        equity[i + 1] = equity[i] * (1 + step_return)
        position = new_position
    
    return equity[1:]

