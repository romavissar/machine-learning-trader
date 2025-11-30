"""Backtest XStrat V1 with Monte Carlo analysis and visualization."""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from datetime import datetime, timedelta
from tqdm import tqdm

from src.models.xstrat_v1 import XStratV1
from src.features.technical import TechnicalFeatures


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df.set_index('timestamp').sort_index()


def create_features(df: pd.DataFrame) -> tuple:
    """Create features and metadata."""
    tech = TechnicalFeatures(df)
    features = tech.all_features()
    features['return_1'] = df['close'].pct_change().fillna(0)
    features['return_5'] = df['close'].pct_change(5).fillna(0)
    features['sma200'] = df['close'].rolling(200).mean()
    features['price_vs_sma200'] = (df['close'] - features['sma200']) / features['sma200']
    
    adx, atr, vol = tech.adx(), tech.atr(), tech.historical_volatility()
    returns = df['close'].pct_change().shift(-1)
    targets = np.where(returns > 0.001, 2, np.where(returns < -0.001, 0, 1))
    
    warmup = 200
    X = np.nan_to_num(features.iloc[warmup:-1].values.astype(np.float32), nan=0.0)
    prices = df['close'].values[warmup:-1]
    high = df['high'].values[warmup:-1]
    low = df['low'].values[warmup:-1]
    adx_arr = np.nan_to_num(adx.iloc[warmup:-1].values.astype(np.float32), nan=15.0)
    atr_arr = np.nan_to_num(atr.iloc[warmup:-1].values.astype(np.float32), nan=0.0)
    vol_arr = np.nan_to_num(vol.iloc[warmup:-1].values.astype(np.float32), nan=0.02)
    targets = targets[warmup:-1]
    timestamps = df.index[warmup:-1]
    
    return X, prices, high, low, adx_arr, atr_arr, vol_arr, targets, timestamps


def run_backtest(strat: XStratV1, X, prices, adx, atr, vol, targets, timestamps,
                 start_ratio=0.7, initial_capital=10000.0, tx_cost=0.001) -> pd.DataFrame:
    """Run backtest with all XStrat layers."""
    n = len(X)
    start_idx = int(n * start_ratio)
    
    results = []
    equity = initial_capital
    position = 0.0
    entry_price = 0.0
    strat.current_equity = equity
    strat.equity_peak = equity
    
    for i in tqdm(range(start_idx, n - 1), desc="Backtest"):
        now = pd.Timestamp(timestamps[i])
        price_history = prices[max(0, i-200):i+1]
        
        result = strat.generate_signal(
            X=X[i:i+1], prices=price_history, adx=adx[i], atr=atr[i],
            volatility=vol[i], now=now.to_pydatetime()
        )
        
        # Calculate PnL
        actual_return = (prices[i+1] - prices[i]) / prices[i]
        new_position = result.position_size * (1 if result.signal == 2 else -1 if result.signal == 0 else 0)
        
        if not result.trade_allowed:
            new_position = position  # Keep current position
        
        pos_change = abs(new_position - position)
        tc = pos_change * tx_cost
        pnl = position * actual_return - tc
        equity *= (1 + pnl)
        
        # Update strategy state
        if pos_change > 0.01:
            strat.record_trade(now.to_pydatetime(), pnl)
            if new_position != 0:
                strat.position.size = new_position
                strat.position.entry_price = prices[i]
                strat.position.stop_loss = result.stop_loss_price or 0.0
        
        strat.update_equity(equity)
        position = new_position
        
        results.append({
            'timestamp': timestamps[i], 'signal': result.signal, 'confidence': result.confidence,
            'trend': result.trend, 'regime': result.regime, 'position': position,
            'trade_allowed': result.trade_allowed, 'reason': result.reason,
            'actual': targets[i], 'return': actual_return, 'pnl': pnl, 'equity': equity,
            'price': prices[i]
        })
    
    return pd.DataFrame(results)


def run_monte_carlo(results: pd.DataFrame, n_sims=1000, noise_std=0.1) -> dict:
    """Run Monte Carlo simulations."""
    returns = results['return'].values
    positions = results['position'].values
    initial = results['equity'].iloc[0] if len(results) > 0 else 10000
    
    np.random.seed(42)
    equity_curves = np.zeros((n_sims, len(returns)))
    
    for i in tqdm(range(n_sims), desc="Monte Carlo"):
        noisy_pos = positions * (1 + np.random.normal(0, noise_std, len(positions)))
        noisy_pos = np.clip(noisy_pos, -0.5, 0.5)
        sim_pnl = noisy_pos[:-1] * returns[1:]
        sim_equity = initial * np.cumprod(1 + np.concatenate([[0], sim_pnl]))
        equity_curves[i] = sim_equity
    
    final_values = equity_curves[:, -1]
    sharpe_ratios = []
    max_dds = []
    
    for curve in equity_curves:
        rets = np.diff(curve) / curve[:-1]
        sharpe_ratios.append(np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(8760))
        peak = np.maximum.accumulate(curve)
        dd = (peak - curve) / (peak + 1e-8)
        max_dds.append(np.max(dd))
    
    return {
        'equity_curves': equity_curves,
        'final_values': final_values,
        'sharpe_ratios': np.array(sharpe_ratios),
        'max_drawdowns': np.array(max_dds),
        'percentiles': {p: np.percentile(equity_curves, p, axis=0) for p in [5, 25, 50, 75, 95]},
        'actual_equity': results['equity'].values,
        'timestamps': results['timestamp'].values
    }


def visualize(results: pd.DataFrame, mc: dict, output_path: Path):
    """Create visualization."""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[2, 2, 1])
    
    dates = pd.to_datetime(results['timestamp'])
    colors = {'bg': '#0d1117', 'panel': '#161b22', 'grid': '#30363d', 
              'btc': '#f7931a', 'equity': '#58a6ff', 'median': '#3fb950'}
    
    for ax in axes:
        ax.set_facecolor(colors['panel'])
        ax.grid(True, alpha=0.2, color=colors['grid'])
    
    # Panel 1: Price
    axes[0].plot(dates, results['price'], color=colors['btc'], lw=1.5)
    axes[0].set_ylabel('BTC Price')
    axes[0].set_title('XStrat V1: Backtest Results', fontweight='bold')
    
    # Panel 2: Equity with MC bands
    n = min(len(dates), mc['equity_curves'].shape[1])
    for curve in mc['equity_curves'][::50]:
        axes[1].plot(dates[:n], curve[:n], color=colors['grid'], alpha=0.1, lw=0.5)
    axes[1].fill_between(dates[:n], mc['percentiles'][5][:n], mc['percentiles'][95][:n], 
                         color=colors['median'], alpha=0.15, label='5-95%')
    axes[1].plot(dates[:n], mc['percentiles'][50][:n], color=colors['median'], 
                 lw=1.5, ls='--', label='Median')
    axes[1].plot(dates[:n], mc['actual_equity'][:n], color=colors['equity'], lw=2, label='Actual')
    axes[1].axhline(10000, color='white', ls=':', alpha=0.3)
    axes[1].set_ylabel('Equity')
    axes[1].legend(loc='upper left')
    
    # Panel 3: Regime
    regime_colors = {'trending': '#238636', 'ranging': '#1f6feb', 'volatile': '#da3633'}
    for i in range(len(dates) - 1):
        color = regime_colors.get(results['regime'].iloc[i], '#1f6feb')
        axes[2].axvspan(dates.iloc[i], dates.iloc[i+1], color=color, alpha=0.7)
    axes[2].set_ylabel('Regime')
    axes[2].set_yticks([])
    
    # Stats box
    stats = (f"Final: ${mc['actual_equity'][-1]:,.0f}  |  "
             f"Sharpe: {np.mean(mc['sharpe_ratios']):.2f}  |  "
             f"MaxDD: {np.mean(mc['max_drawdowns']):.1%}  |  "
             f"P(Profit): {np.mean(mc['final_values'] > 10000):.1%}")
    fig.text(0.5, 0.02, stats, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor=colors['panel'], alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, facecolor=colors['bg'], bbox_inches='tight')
    print(f"[SAVE] {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("XStrat V1 Backtest + Monte Carlo")
    print("=" * 60)
    
    # Load model
    model_path = PROJECT_ROOT / "models" / "xstrat_v1.joblib"
    if not model_path.exists():
        print("[ERROR] Model not found. Run train_xstrat.py first.")
        return
    
    strat = XStratV1.load(str(model_path), str(PROJECT_ROOT / "config" / "xstrat_v1.yaml"))
    
    # Load data
    df = load_data(PROJECT_ROOT / "data" / "BTC_USDT_1h.csv")
    X, prices, high, low, adx, atr, vol, targets, ts = create_features(df)
    
    # Backtest
    results = run_backtest(strat, X, prices, adx, atr, vol, targets, ts)
    
    # Analyze
    total_ret = (results['equity'].iloc[-1] / results['equity'].iloc[0]) - 1
    trades = (results['position'].diff().abs() > 0.01).sum()
    accuracy = (results['signal'] == results['actual']).mean()
    
    print(f"\n[RESULTS]")
    print(f"  Return: {total_ret:.2%}")
    print(f"  Trades: {trades}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Regime dist: {results['regime'].value_counts().to_dict()}")
    
    # Monte Carlo
    mc = run_monte_carlo(results, n_sims=1000)
    print(f"\n[MC] Mean Sharpe: {np.mean(mc['sharpe_ratios']):.2f}")
    print(f"[MC] P(Profit): {np.mean(mc['final_values'] > 10000):.1%}")
    
    # Save
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    results.to_csv(output_dir / "xstrat_v1_results.csv", index=False)
    joblib.dump(mc, output_dir / "xstrat_v1_monte_carlo.joblib")
    visualize(results, mc, output_dir / "xstrat_v1_monte_carlo.png")
    
    print("\n" + "=" * 60)
    print("[OK] Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

