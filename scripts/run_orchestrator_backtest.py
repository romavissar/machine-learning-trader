"""Run backtest with the Model Orchestrator and Monte Carlo analysis.

This script:
1. Loads a trained orchestrator (or trains one if not found)
2. Runs a walk-forward backtest tracking model selection and regime
3. Performs Monte Carlo simulations to estimate strategy variance
4. Saves results for visualization

Output columns:
- timestamp: Time of prediction
- signal: Predicted signal (0=down, 1=hold, 2=up)
- confidence: Model confidence in prediction
- regime: Market regime (trending/ranging/volatile)
- selected_models: Which models were used
- actual: Actual price direction
- position: Risk-adjusted position
- return: Actual return for the period
- pnl: Strategy PnL for the period
- equity: Running equity curve
"""
import sys
from pathlib import Path
import warnings

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', message='.*feature names.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from tqdm import tqdm

from src.models.orchestrator import ModelOrchestrator, OrchestratorConfig
from src.features.technical import TechnicalFeatures
from src.validation.monte_carlo import MonteCarloBacktest, MonteCarloResult


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and prepare OHLCV data."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()
    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df


def create_features(df: pd.DataFrame) -> tuple:
    """Create features from OHLCV data."""
    # Extract arrays
    prices = df['close'].values.astype(np.float32)
    high = df['high'].values.astype(np.float32)
    low = df['low'].values.astype(np.float32)
    
    # Create technical features
    tech = TechnicalFeatures(df)
    features_df = tech.all_features()
    
    # Add price-based features
    features_df['close'] = prices
    features_df['return_1'] = df['close'].pct_change().fillna(0)
    features_df['return_5'] = df['close'].pct_change(5).fillna(0)
    features_df['volume_change'] = df['volume'].pct_change().fillna(0).clip(-10, 10)
    features_df['range'] = ((df['high'] - df['low']) / df['close']).fillna(0)
    features_df['momentum'] = (df['close'] / df['close'].shift(10) - 1).fillna(0)
    
    # Create target
    returns = df['close'].pct_change().shift(-1)
    targets = np.where(returns > 0.001, 2,
              np.where(returns < -0.001, 0, 1))
    
    # Drop warmup period
    warmup = 50
    features_df = features_df.iloc[warmup:-1]
    prices = prices[warmup:-1]
    high = high[warmup:-1]
    low = low[warmup:-1]
    targets = targets[warmup:-1]
    timestamps = df.index[warmup:-1]
    
    # Handle NaN/Inf
    features = features_df.values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features, prices, high, low, targets, timestamps


def run_backtest(
    orchestrator: ModelOrchestrator,
    features: np.ndarray,
    prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    targets: np.ndarray,
    timestamps: np.ndarray,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
    test_start_ratio: float = 0.7,
) -> pd.DataFrame:
    """Run backtest with regime and model tracking.
    
    Args:
        orchestrator: Trained ModelOrchestrator
        features: Feature matrix
        prices: Price array
        high: High prices
        low: Low prices  
        targets: Target labels
        timestamps: Timestamps
        initial_capital: Starting capital
        transaction_cost: Transaction cost as fraction
        test_start_ratio: Where to start the backtest (after training period)
        
    Returns:
        DataFrame with backtest results
    """
    n_samples = len(features)
    test_start = int(n_samples * test_start_ratio)
    
    print(f"\n[BACKTEST] Running backtest from index {test_start} to {n_samples}")
    print(f"   Test period: {timestamps[test_start]} to {timestamps[-1]}")
    
    results = []
    equity = initial_capital
    position = 0.0
    
    for i in tqdm(range(test_start, n_samples - 1), desc="Backtest"):
        # Get price history for regime detection
        lookback = min(i, 50)
        price_history = prices[i - lookback:i + 1]
        high_history = high[i - lookback:i + 1]
        low_history = low[i - lookback:i + 1]
        
        # Generate signal
        result = orchestrator.generate_signal(
            X=features[i:i + 1],
            prices=price_history,
            high=high_history,
            low=low_history,
        )
        
        # Calculate return
        actual_return = (prices[i + 1] - prices[i]) / prices[i]
        
        # Calculate PnL with transaction costs
        new_position = result.risk_adjusted_position
        position_change = abs(new_position - position)
        tc = position_change * transaction_cost
        pnl = position * actual_return - tc
        
        # Update equity
        equity = equity * (1 + pnl)
        position = new_position
        
        results.append({
            'timestamp': timestamps[i],
            'signal': result.signal,
            'confidence': result.confidence,
            'regime': result.regime,
            'selected_models': ','.join(result.selected_models),
            'actual': targets[i],
            'position': new_position,
            'return': actual_return,
            'pnl': pnl,
            'equity': equity,
            'price': prices[i],
        })
    
    return pd.DataFrame(results)


def run_monte_carlo(
    backtest_results: pd.DataFrame,
    n_simulations: int = 1000,
    noise_std: float = 0.1,
    initial_capital: float = 10000.0,
) -> MonteCarloResult:
    """Run Monte Carlo simulations on backtest results.
    
    Args:
        backtest_results: DataFrame from run_backtest
        n_simulations: Number of Monte Carlo simulations
        noise_std: Standard deviation of noise to add
        initial_capital: Starting capital
        
    Returns:
        MonteCarloResult with simulation data
    """
    print(f"\n[MC] Running {n_simulations} Monte Carlo simulations...")
    
    # Extract data from backtest
    returns = backtest_results['return'].values
    signals = backtest_results['signal'].values
    confidences = backtest_results['confidence'].values
    timestamps = backtest_results['timestamp'].values
    actual_equity = backtest_results['equity'].values
    
    # Create Monte Carlo backtester
    mc = MonteCarloBacktest(
        signal_generator=None,
        n_simulations=n_simulations,
        noise_std=noise_std,
        initial_capital=initial_capital,
    )
    
    # Run simulations
    results = mc.run(
        returns=returns,
        base_signals=signals,
        base_confidences=confidences,
        timestamps=timestamps,
        actual_equity=actual_equity,
        show_progress=True,
    )
    
    # Print summary statistics
    stats = mc.summary_statistics(results)
    print("\n[STATS] Monte Carlo Statistics:")
    print(f"   Mean final value:    ${stats['mean_final_value']:,.2f}")
    print(f"   Median final value:  ${stats['median_final_value']:,.2f}")
    print(f"   5th percentile:      ${stats['p5_final_value']:,.2f}")
    print(f"   95th percentile:     ${stats['p95_final_value']:,.2f}")
    print(f"   Mean Sharpe:         {stats['mean_sharpe']:.2f}")
    print(f"   Mean Max Drawdown:   {stats['mean_max_drawdown']:.2%}")
    print(f"   Probability of profit: {stats['prob_profit']:.1%}")
    print(f"   Probability of >10% loss: {stats['prob_loss_10pct']:.1%}")
    
    return results


def analyze_results(backtest_df: pd.DataFrame) -> dict:
    """Analyze backtest results."""
    # Basic metrics
    total_return = (backtest_df['equity'].iloc[-1] / backtest_df['equity'].iloc[0]) - 1
    
    # Sharpe ratio (annualized for hourly data)
    returns = backtest_df['pnl']
    sharpe = returns.mean() / returns.std() * np.sqrt(8760) if returns.std() > 0 else 0
    
    # Max drawdown
    equity = backtest_df['equity']
    peak = equity.expanding().max()
    drawdown = (peak - equity) / peak
    max_drawdown = drawdown.max()
    
    # Win rate
    winning_trades = (backtest_df['pnl'] > 0).sum()
    total_trades = (backtest_df['position'] != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Regime analysis
    regime_performance = backtest_df.groupby('regime')['pnl'].agg(['mean', 'std', 'count'])
    
    # Model selection analysis
    model_usage = backtest_df['selected_models'].value_counts()
    
    # Accuracy
    correct = (backtest_df['signal'] == backtest_df['actual']).sum()
    accuracy = correct / len(backtest_df)
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'accuracy': accuracy,
        'total_trades': total_trades,
        'regime_performance': regime_performance,
        'model_usage': model_usage,
    }


def main():
    """Main entry point for backtest."""
    data_path = PROJECT_ROOT / "data" / "BTC_USDT_1h.csv"
    model_path = PROJECT_ROOT / "models" / "orchestrator.joblib"
    output_dir = PROJECT_ROOT / "outputs"
    
    # Check for data
    if not data_path.exists():
        print(f"[ERROR] Data file not found: {data_path}")
        print("   Please run data/scripts/download_data.py first")
        return
    
    # Load or train orchestrator
    if model_path.exists():
        print(f"Loading trained orchestrator from {model_path}...")
        orchestrator = ModelOrchestrator.load(str(model_path))
    else:
        print("No trained model found. Training new orchestrator...")
        from train_orchestrator import train_orchestrator as train_func
        orchestrator, _ = train_func(
            data_path=data_path,
            output_dir=PROJECT_ROOT / "models",
        )
    
    # Load and prepare data
    df = load_data(data_path)
    features, prices, high, low, targets, timestamps = create_features(df)
    
    print("=" * 60)
    print("Orchestrator Backtest with Monte Carlo Analysis")
    print("=" * 60)
    
    # Run backtest
    backtest_df = run_backtest(
        orchestrator=orchestrator,
        features=features,
        prices=prices,
        high=high,
        low=low,
        targets=targets,
        timestamps=timestamps,
        initial_capital=10000.0,
        test_start_ratio=0.7,
    )
    
    # Analyze results
    print("\n[ANALYSIS] Backtest Analysis:")
    analysis = analyze_results(backtest_df)
    print(f"   Total Return:    {analysis['total_return']:.2%}")
    print(f"   Sharpe Ratio:    {analysis['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown:    {analysis['max_drawdown']:.2%}")
    print(f"   Win Rate:        {analysis['win_rate']:.2%}")
    print(f"   Accuracy:        {analysis['accuracy']:.2%}")
    print(f"   Total Trades:    {analysis['total_trades']}")
    
    print("\n[REGIME] Regime Performance:")
    print(analysis['regime_performance'])
    
    print("\n[MODELS] Model Usage:")
    print(analysis['model_usage'])
    
    # Run Monte Carlo simulations
    mc_results = run_monte_carlo(
        backtest_results=backtest_df,
        n_simulations=1000,
        noise_std=0.1,
        initial_capital=10000.0,
    )
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    backtest_path = output_dir / "backtest_results.csv"
    backtest_df.to_csv(backtest_path, index=False)
    print(f"\n[SAVE] Backtest results saved to {backtest_path}")
    
    mc_path = output_dir / "monte_carlo_results.joblib"
    joblib.dump({
        'equity_curves': mc_results.equity_curves,
        'percentiles': mc_results.percentiles,
        'actual_equity': mc_results.actual_equity,
        'timestamps': mc_results.timestamps,
        'sharpe_ratios': mc_results.sharpe_ratios,
        'max_drawdowns': mc_results.max_drawdowns,
        'final_values': mc_results.final_values,
    }, mc_path)
    print(f"[SAVE] Monte Carlo results saved to {mc_path}")
    
    # Save price data for visualization
    price_data = backtest_df[['timestamp', 'price']].copy()
    price_data.to_csv(output_dir / "price_data.csv", index=False)
    
    print("\n" + "=" * 60)
    print("[OK] Backtest and Monte Carlo analysis complete!")
    print(f"   Run scripts/visualize_monte_carlo.py to generate plots")
    print("=" * 60)
    
    return backtest_df, mc_results, analysis


if __name__ == "__main__":
    main()

