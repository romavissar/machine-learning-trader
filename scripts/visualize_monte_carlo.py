"""Visualize Monte Carlo simulation results against actual BTC prices.

Generates a multi-panel figure showing:
1. BTC actual price over the backtest period
2. Strategy equity curves with Monte Carlo confidence bands
3. Market regime timeline

Usage:
    python scripts/visualize_monte_carlo.py

Requires:
    - outputs/backtest_results.csv (from run_orchestrator_backtest.py)
    - outputs/monte_carlo_results.joblib (from run_orchestrator_backtest.py)
"""
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import joblib
from datetime import datetime


# Style configuration - Modern dark theme
plt.style.use('dark_background')
COLORS = {
    'background': '#0d1117',
    'panel_bg': '#161b22',
    'grid': '#30363d',
    'text': '#c9d1d9',
    'text_muted': '#8b949e',
    'btc_price': '#f7931a',  # Bitcoin orange
    'actual_equity': '#58a6ff',  # Bright blue
    'median_equity': '#3fb950',  # Green
    'percentile_90': 'rgba(63, 185, 80, 0.3)',
    'percentile_50': 'rgba(63, 185, 80, 0.15)',
    'simulations': '#30363d',
    'regime_trending': '#238636',
    'regime_ranging': '#1f6feb',
    'regime_volatile': '#da3633',
}


def load_results(output_dir: Path) -> tuple:
    """Load backtest and Monte Carlo results."""
    backtest_path = output_dir / "backtest_results.csv"
    mc_path = output_dir / "monte_carlo_results.joblib"
    
    if not backtest_path.exists():
        raise FileNotFoundError(f"Backtest results not found at {backtest_path}")
    if not mc_path.exists():
        raise FileNotFoundError(f"Monte Carlo results not found at {mc_path}")
    
    print(f"Loading backtest results from {backtest_path}...")
    backtest_df = pd.read_csv(backtest_path, parse_dates=['timestamp'])
    
    print(f"Loading Monte Carlo results from {mc_path}...")
    mc_data = joblib.load(mc_path)
    
    return backtest_df, mc_data


def create_figure(backtest_df: pd.DataFrame, mc_data: dict, output_path: Path):
    """Create the Monte Carlo visualization figure."""
    print("Creating visualization...")
    
    # Extract data
    timestamps = backtest_df['timestamp'].values
    prices = backtest_df['price'].values
    actual_equity = mc_data['actual_equity']
    equity_curves = mc_data['equity_curves']
    percentiles = mc_data['percentiles']
    regimes = backtest_df['regime'].values
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(16, 12), facecolor=COLORS['background'])
    
    # Use GridSpec for custom layout
    gs = fig.add_gridspec(
        3, 1, 
        height_ratios=[2, 2, 0.5],
        hspace=0.15,
        left=0.08, right=0.95,
        top=0.92, bottom=0.08,
    )
    
    ax1 = fig.add_subplot(gs[0])  # BTC Price
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Strategy Equity
    ax3 = fig.add_subplot(gs[2], sharex=ax1)  # Regime Timeline
    
    # Set panel backgrounds
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(COLORS['panel_bg'])
        ax.tick_params(colors=COLORS['text'])
        ax.spines['bottom'].set_color(COLORS['grid'])
        ax.spines['top'].set_color(COLORS['grid'])
        ax.spines['left'].set_color(COLORS['grid'])
        ax.spines['right'].set_color(COLORS['grid'])
    
    # Convert timestamps
    dates = pd.to_datetime(timestamps)
    
    # =========================================================================
    # Panel 1: BTC Price
    # =========================================================================
    ax1.plot(dates, prices, color=COLORS['btc_price'], linewidth=1.5, label='BTC/USDT')
    ax1.fill_between(dates, prices.min() * 0.99, prices, 
                     color=COLORS['btc_price'], alpha=0.1)
    
    ax1.set_ylabel('Price (USDT)', color=COLORS['text'], fontsize=11)
    ax1.set_title('Bitcoin Price', color=COLORS['text'], fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.2, color=COLORS['grid'])
    ax1.legend(loc='upper left', facecolor=COLORS['panel_bg'], edgecolor=COLORS['grid'])
    
    # Format y-axis with commas
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Hide x-axis labels for top panel
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # =========================================================================
    # Panel 2: Strategy Equity with Monte Carlo Bands
    # =========================================================================
    
    # Ensure we have the right number of data points
    n_points = min(len(dates), equity_curves.shape[1], len(actual_equity))
    plot_dates = dates[:n_points]
    
    # Plot a sample of simulation paths (for visual effect)
    n_sample = min(200, equity_curves.shape[0])
    sample_indices = np.random.choice(equity_curves.shape[0], n_sample, replace=False)
    for idx in sample_indices:
        ax2.plot(plot_dates, equity_curves[idx, :n_points], 
                color=COLORS['simulations'], alpha=0.03, linewidth=0.5)
    
    # Plot percentile bands
    if 5 in percentiles and 95 in percentiles:
        ax2.fill_between(plot_dates, 
                        percentiles[5][:n_points], 
                        percentiles[95][:n_points],
                        color=COLORS['median_equity'], alpha=0.15,
                        label='5th-95th percentile')
    
    if 25 in percentiles and 75 in percentiles:
        ax2.fill_between(plot_dates,
                        percentiles[25][:n_points],
                        percentiles[75][:n_points],
                        color=COLORS['median_equity'], alpha=0.25,
                        label='25th-75th percentile')
    
    # Plot median
    if 50 in percentiles:
        ax2.plot(plot_dates, percentiles[50][:n_points],
                color=COLORS['median_equity'], linewidth=2, linestyle='--',
                label='Median simulation')
    
    # Plot actual equity curve
    ax2.plot(plot_dates, actual_equity[:n_points],
            color=COLORS['actual_equity'], linewidth=2.5,
            label='Actual strategy')
    
    # Add initial capital reference line
    initial_capital = actual_equity[0] if len(actual_equity) > 0 else 10000
    ax2.axhline(y=initial_capital, color=COLORS['text_muted'], 
               linestyle=':', alpha=0.5, linewidth=1)
    ax2.text(plot_dates[0], initial_capital * 1.01, 
            f'Initial: ${initial_capital:,.0f}', 
            color=COLORS['text_muted'], fontsize=9)
    
    ax2.set_ylabel('Portfolio Value (USDT)', color=COLORS['text'], fontsize=11)
    ax2.set_title('Strategy Performance with Monte Carlo Confidence Bands (1000 simulations)', 
                  color=COLORS['text'], fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.2, color=COLORS['grid'])
    ax2.legend(loc='upper left', facecolor=COLORS['panel_bg'], edgecolor=COLORS['grid'])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Hide x-axis labels
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # =========================================================================
    # Panel 3: Regime Timeline
    # =========================================================================
    
    # Create regime color map
    regime_colors = {
        'trending': COLORS['regime_trending'],
        'ranging': COLORS['regime_ranging'],
        'volatile': COLORS['regime_volatile'],
    }
    
    # Plot regime as colored bars
    regime_numeric = np.zeros(len(regimes))
    for i, regime in enumerate(regimes):
        if regime == 'trending':
            regime_numeric[i] = 2
        elif regime == 'volatile':
            regime_numeric[i] = 0
        else:  # ranging
            regime_numeric[i] = 1
    
    # Create colored segments
    for i in range(len(plot_dates) - 1):
        regime = regimes[i] if i < len(regimes) else 'ranging'
        color = regime_colors.get(regime, COLORS['regime_ranging'])
        ax3.axvspan(plot_dates[i], plot_dates[i + 1], 
                   facecolor=color, alpha=0.7)
    
    ax3.set_ylabel('Regime', color=COLORS['text'], fontsize=10)
    ax3.set_yticks([])
    ax3.set_xlabel('Date', color=COLORS['text'], fontsize=11)
    
    # Create legend for regimes
    legend_elements = [
        Patch(facecolor=COLORS['regime_trending'], label='Trending (ADX > 25)'),
        Patch(facecolor=COLORS['regime_ranging'], label='Ranging (ADX < 25)'),
        Patch(facecolor=COLORS['regime_volatile'], label='Volatile (High Vol)'),
    ]
    ax3.legend(handles=legend_elements, loc='upper left', ncol=3,
              facecolor=COLORS['panel_bg'], edgecolor=COLORS['grid'],
              fontsize=9)
    
    # Format x-axis dates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # =========================================================================
    # Add summary statistics box
    # =========================================================================
    
    # Calculate statistics
    final_values = mc_data['final_values']
    sharpe_ratios = mc_data['sharpe_ratios']
    max_drawdowns = mc_data['max_drawdowns']
    
    stats_text = (
        f"Monte Carlo Summary (n={len(final_values):,})\n"
        f"─────────────────────────\n"
        f"Final Value:\n"
        f"  Mean: ${np.mean(final_values):,.0f}\n"
        f"  Median: ${np.median(final_values):,.0f}\n"
        f"  5th pctl: ${np.percentile(final_values, 5):,.0f}\n"
        f"  95th pctl: ${np.percentile(final_values, 95):,.0f}\n"
        f"─────────────────────────\n"
        f"Sharpe Ratio:\n"
        f"  Mean: {np.mean(sharpe_ratios):.2f}\n"
        f"─────────────────────────\n"
        f"Max Drawdown:\n"
        f"  Mean: {np.mean(max_drawdowns):.1%}\n"
        f"  Worst: {np.max(max_drawdowns):.1%}\n"
        f"─────────────────────────\n"
        f"P(Profit): {np.mean(final_values > initial_capital):.1%}"
    )
    
    # Add text box
    props = dict(boxstyle='round,pad=0.5', facecolor=COLORS['panel_bg'], 
                edgecolor=COLORS['grid'], alpha=0.9)
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, color=COLORS['text'], family='monospace')
    
    # =========================================================================
    # Main title
    # =========================================================================
    fig.suptitle('Model Orchestrator: Monte Carlo Analysis vs Bitcoin Price',
                fontsize=16, fontweight='bold', color=COLORS['text'], y=0.98)
    
    # Add timestamp
    fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            fontsize=8, color=COLORS['text_muted'], ha='right')
    
    # Save figure
    plt.savefig(output_path, dpi=150, facecolor=COLORS['background'],
               edgecolor='none', bbox_inches='tight')
    print(f"[SAVE] Figure saved to {output_path}")
    
    plt.close()


def create_distribution_plot(mc_data: dict, output_path: Path):
    """Create distribution plots for Monte Carlo results."""
    print("Creating distribution plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=COLORS['background'])
    
    final_values = mc_data['final_values']
    sharpe_ratios = mc_data['sharpe_ratios']
    max_drawdowns = mc_data['max_drawdowns']
    initial_capital = final_values[0] if len(final_values) > 0 else 10000
    
    # Calculate returns
    returns = (final_values - initial_capital) / initial_capital * 100
    
    for ax in axes:
        ax.set_facecolor(COLORS['panel_bg'])
        ax.tick_params(colors=COLORS['text'])
        ax.spines['bottom'].set_color(COLORS['grid'])
        ax.spines['top'].set_color(COLORS['grid'])
        ax.spines['left'].set_color(COLORS['grid'])
        ax.spines['right'].set_color(COLORS['grid'])
    
    # Panel 1: Return distribution
    axes[0].hist(returns, bins=50, color=COLORS['median_equity'], alpha=0.7, edgecolor=COLORS['grid'])
    axes[0].axvline(x=0, color=COLORS['text_muted'], linestyle='--', linewidth=1, label='Break-even')
    axes[0].axvline(x=np.median(returns), color=COLORS['actual_equity'], linestyle='-', linewidth=2, label=f'Median: {np.median(returns):.1f}%')
    axes[0].set_xlabel('Return (%)', color=COLORS['text'])
    axes[0].set_ylabel('Frequency', color=COLORS['text'])
    axes[0].set_title('Return Distribution', color=COLORS['text'], fontweight='bold')
    axes[0].legend(facecolor=COLORS['panel_bg'], edgecolor=COLORS['grid'])
    axes[0].grid(True, alpha=0.2, color=COLORS['grid'])
    
    # Panel 2: Sharpe ratio distribution
    axes[1].hist(sharpe_ratios, bins=50, color=COLORS['btc_price'], alpha=0.7, edgecolor=COLORS['grid'])
    axes[1].axvline(x=0, color=COLORS['text_muted'], linestyle='--', linewidth=1, label='Zero Sharpe')
    axes[1].axvline(x=np.median(sharpe_ratios), color=COLORS['actual_equity'], linestyle='-', linewidth=2, label=f'Median: {np.median(sharpe_ratios):.2f}')
    axes[1].set_xlabel('Sharpe Ratio', color=COLORS['text'])
    axes[1].set_ylabel('Frequency', color=COLORS['text'])
    axes[1].set_title('Sharpe Ratio Distribution', color=COLORS['text'], fontweight='bold')
    axes[1].legend(facecolor=COLORS['panel_bg'], edgecolor=COLORS['grid'])
    axes[1].grid(True, alpha=0.2, color=COLORS['grid'])
    
    # Panel 3: Max drawdown distribution
    axes[2].hist(max_drawdowns * 100, bins=50, color=COLORS['regime_volatile'], alpha=0.7, edgecolor=COLORS['grid'])
    axes[2].axvline(x=np.median(max_drawdowns) * 100, color=COLORS['actual_equity'], linestyle='-', linewidth=2, label=f'Median: {np.median(max_drawdowns):.1%}')
    axes[2].set_xlabel('Max Drawdown (%)', color=COLORS['text'])
    axes[2].set_ylabel('Frequency', color=COLORS['text'])
    axes[2].set_title('Max Drawdown Distribution', color=COLORS['text'], fontweight='bold')
    axes[2].legend(facecolor=COLORS['panel_bg'], edgecolor=COLORS['grid'])
    axes[2].grid(True, alpha=0.2, color=COLORS['grid'])
    
    fig.suptitle('Monte Carlo Simulation Distributions',
                fontsize=14, fontweight='bold', color=COLORS['text'], y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=COLORS['background'],
               edgecolor='none', bbox_inches='tight')
    print(f"[SAVE] Distribution plot saved to {output_path}")
    
    plt.close()


def main():
    """Main entry point for visualization."""
    output_dir = PROJECT_ROOT / "outputs"
    
    if not output_dir.exists():
        print(f"[ERROR] Output directory not found: {output_dir}")
        print("   Please run scripts/run_orchestrator_backtest.py first")
        return
    
    try:
        backtest_df, mc_data = load_results(output_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("   Please run scripts/run_orchestrator_backtest.py first")
        return
    
    print("=" * 60)
    print("Monte Carlo Visualization")
    print("=" * 60)
    
    # Create main figure
    main_plot_path = output_dir / "monte_carlo_analysis.png"
    create_figure(backtest_df, mc_data, main_plot_path)
    
    # Create distribution plot
    dist_plot_path = output_dir / "monte_carlo_distributions.png"
    create_distribution_plot(mc_data, dist_plot_path)
    
    print("\n" + "=" * 60)
    print("[OK] Visualization complete!")
    print(f"   Main plot: {main_plot_path}")
    print(f"   Distributions: {dist_plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

