#!/usr/bin/env python3
"""
Generate performance and drawdown charts for ALL tested combinations.
Saves each combination's charts in a folder named by binary pattern.

Example: results/charts/RESIST/RESIST_000010001/ contains:
  - performance.png
  - drawdown.png
  - metrics.json
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import argparse
import sys

# Signal conditions
SIGNAL_CONDITIONS = {
    'mount': ['cyber_negative', 'value_gt_vindex'],
    'climb': ['cyber_negative', 'value_gt_vindex'],
    'arrow': ['lower_close', 'cyber_negative', 'value_gt_vindex', 'rsl_itrend_w_weak', 'rsl_itrend_d_weak'],
    'collect': ['lower_close', 'cyber_negative', 'value_gt_vindex', 'rsl_itrend_w_weak', 'rsl_itrend_d_weak'],
    'solid': ['cyber_negative', 'value_gt_vindex', 'rsl_itrend_w_weak', 'rsl_itrend_d_weak', 'rsl_close_weak', 
              'lower_low', 'lower_close', 'lower_close_prev', 'below_weekly_low'],
    'resist': ['cyber_negative', 'value_gt_vindex', 'rsl_itrend_w_weak', 'rsl_itrend_d_weak', 'rsl_close_weak',
               'lower_low', 'lower_close', 'lower_close_prev', 'below_weekly_low'],
}

CONDITION_FUNCTIONS = {
    'cyber_negative': lambda df: df['cyber_cycle'] < 0,
    'value_gt_vindex': lambda df: df['swing_value'] > df['vindex'],
    'rsl_itrend_w_weak': lambda df: df['rsl_itrend_w'] < 100,
    'rsl_itrend_d_weak': lambda df: df['rsl_itrend_d'] < 100,
    'rsl_close_weak': lambda df: df['rsl_close'] < 100,
    'lower_low': lambda df: df['Low'] < df['Low'].shift(1),
    'lower_close': lambda df: df['Close'] < df['Close'].shift(1),
    'lower_close_prev': lambda df: df['Close'].shift(1) < df['Close'].shift(2),
    'below_weekly_low': lambda df: df['Close'] < df['weekly_low'],
}


def load_data(data_dir):
    """Load all ticker data."""
    data_path = Path(data_dir)
    all_data = {}
    
    for f in data_path.glob('*_indicators.csv'):
        ticker = f.stem.replace('_indicators', '')
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        all_data[ticker] = df
    
    return all_data


def get_enabled_conditions(combo_id, conditions):
    """Get list of enabled conditions for a combo ID."""
    return [conditions[i] for i in range(len(conditions)) if combo_id & (1 << i)]


def combo_to_binary_string(combo_id, num_conditions):
    """Convert combo ID to binary string."""
    return format(combo_id, f'0{num_conditions}b')


def simulate_combination(all_data, enabled_conditions, initial_capital=100000):
    """Simulate a strategy and return NAV history."""
    # Get common trading days
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index)
    trading_days = sorted(all_dates)
    
    # Generate signals
    signals = {}
    for ticker, df in all_data.items():
        if len(enabled_conditions) == 0:
            signals[ticker] = pd.Series(False, index=df.index)
            continue
            
        mask = pd.Series(True, index=df.index)
        for cond_name in enabled_conditions:
            if cond_name in CONDITION_FUNCTIONS:
                try:
                    cond_mask = CONDITION_FUNCTIONS[cond_name](df)
                    mask = mask & cond_mask
                except:
                    mask = pd.Series(False, index=df.index)
        signals[ticker] = mask
    
    # Simulation
    capital = initial_capital
    positions = {}
    nav_history = []
    max_positions = 10
    position_size = 0.1
    stop_loss_pct = 0.05
    exit_bars = 10
    
    for i, date in enumerate(trading_days):
        # Calculate portfolio value
        portfolio_value = capital
        for ticker, pos in positions.items():
            if ticker in all_data and date in all_data[ticker].index:
                current_price = all_data[ticker].loc[date, 'Close']
                portfolio_value += pos['shares'] * current_price
        
        nav_history.append({'date': date, 'nav': portfolio_value})
        
        # Process exits
        positions_to_close = []
        for ticker, pos in positions.items():
            if ticker not in all_data or date not in all_data[ticker].index:
                continue
            
            current_price = all_data[ticker].loc[date, 'Close']
            pnl_pct = (current_price / pos['entry_price'] - 1)
            bars_held = pos['bars_held'] + 1
            
            if pnl_pct <= -stop_loss_pct or bars_held >= exit_bars:
                exit_value = pos['shares'] * current_price
                capital += exit_value
                positions_to_close.append(ticker)
            else:
                pos['bars_held'] = bars_held
        
        for ticker in positions_to_close:
            del positions[ticker]
        
        # Process entries
        if i > 0 and len(positions) < max_positions:
            prev_date = trading_days[i - 1]
            
            for ticker, df in all_data.items():
                if ticker in positions or len(positions) >= max_positions:
                    continue
                if prev_date not in df.index or date not in df.index:
                    continue
                
                if ticker in signals and signals[ticker].get(prev_date, False):
                    entry_price = df.loc[date, 'Open']
                    if entry_price > 0:
                        shares = (capital * position_size) / entry_price
                        cost = shares * entry_price
                        
                        if cost <= capital:
                            capital -= cost
                            positions[ticker] = {
                                'entry_price': entry_price,
                                'shares': shares,
                                'bars_held': 0
                            }
    
    return pd.DataFrame(nav_history).set_index('date')


def calculate_drawdown(nav_series):
    """Calculate drawdown series from NAV."""
    peak = nav_series.expanding().max()
    drawdown = (nav_series - peak) / peak * 100
    return drawdown


def calculate_metrics(nav_df):
    """Calculate performance metrics."""
    if len(nav_df) < 2:
        return {}
    
    nav = nav_df['nav']
    returns = nav.pct_change().dropna()
    
    total_return = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
    
    # Annualized metrics
    trading_days = len(nav)
    years = trading_days / 252
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Sharpe
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Max drawdown
    drawdown = calculate_drawdown(nav)
    max_dd = drawdown.min()
    
    # Volatility
    volatility = returns.std() * np.sqrt(252) * 100
    
    return {
        'total_return_%': round(total_return, 2),
        'cagr_%': round(cagr, 2),
        'sharpe': round(sharpe, 2),
        'max_drawdown_%': round(max_dd, 2),
        'volatility_%': round(volatility, 2),
        'trading_days': trading_days,
    }


def generate_charts(nav_df, output_dir, signal_name, binary_str, enabled_conditions, metrics):
    """Generate and save performance and drawdown charts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    nav = nav_df['nav']
    drawdown = calculate_drawdown(nav)
    
    # Style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Performance Chart
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(nav.index, nav, color='#2ecc71', linewidth=2)
    ax1.fill_between(nav.index, nav.iloc[0], nav, alpha=0.3, color='#2ecc71')
    ax1.axhline(y=nav.iloc[0], color='gray', linestyle='--', alpha=0.5)
    
    cond_str = ', '.join(enabled_conditions) if enabled_conditions else 'NONE'
    ax1.set_title(f'{signal_name} [{binary_str}]\nConditions: {cond_str}\n'
                  f'Return: {metrics.get("total_return_%", 0):.1f}% | Sharpe: {metrics.get("sharpe", 0):.2f} | Max DD: {metrics.get("max_drawdown_%", 0):.1f}%',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    fig1.tight_layout()
    fig1.savefig(output_path / 'performance.png', dpi=120, facecolor='white', bbox_inches='tight')
    plt.close(fig1)
    
    # Drawdown Chart
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.fill_between(drawdown.index, 0, drawdown, color='#e74c3c', alpha=0.7)
    ax2.plot(drawdown.index, drawdown, color='#c0392b', linewidth=1)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    ax2.set_title(f'{signal_name} [{binary_str}] - Drawdown', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_ylim(min(drawdown.min() * 1.1, -1), 1)
    
    fig2.tight_layout()
    fig2.savefig(output_path / 'drawdown.png', dpi=120, facecolor='white', bbox_inches='tight')
    plt.close(fig2)
    
    # Save metrics
    metrics['conditions'] = enabled_conditions
    metrics['binary'] = binary_str
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)


def process_signal(signal_type, data_dir, output_base_dir):
    """Process all combinations for a signal type."""
    signal_type = signal_type.lower()
    conditions = SIGNAL_CONDITIONS[signal_type]
    num_conditions = len(conditions)
    total_combos = 2 ** num_conditions - 1
    
    print(f"\n{'='*60}")
    print(f"Processing {signal_type.upper()} ({total_combos} combinations)")
    print(f"{'='*60}")
    
    # Load data
    print("Loading data...")
    all_data = load_data(data_dir)
    print(f"Loaded {len(all_data)} tickers")
    
    # Create output directory
    signal_output_dir = Path(output_base_dir) / signal_type.upper()
    signal_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each combination
    all_metrics = []
    
    for combo_id in range(1, total_combos + 1):
        enabled = get_enabled_conditions(combo_id, conditions)
        binary_str = combo_to_binary_string(combo_id, num_conditions)
        folder_name = f"{signal_type.upper()}_{binary_str}"
        
        print(f"  [{combo_id}/{total_combos}] {folder_name}: {', '.join(enabled)[:50]}...", end=" ")
        
        try:
            # Simulate
            nav_df = simulate_combination(all_data, enabled)
            
            if len(nav_df) < 10:
                print("SKIP (insufficient data)")
                continue
            
            # Calculate metrics
            metrics = calculate_metrics(nav_df)
            metrics['combo_id'] = combo_id
            metrics['folder'] = folder_name
            
            # Generate charts
            combo_output_dir = signal_output_dir / folder_name
            generate_charts(nav_df, combo_output_dir, signal_type.upper(), binary_str, enabled, metrics)
            
            all_metrics.append(metrics)
            print(f"OK ({metrics.get('total_return_%', 0):.1f}%)")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Save summary
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_df = summary_df.sort_values('total_return_%', ascending=False)
        summary_df.to_excel(signal_output_dir / f'{signal_type.upper()}_summary.xlsx', index=False)
        print(f"\nSummary saved to {signal_output_dir / f'{signal_type.upper()}_summary.xlsx'}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Generate charts for all signal combinations')
    parser.add_argument('--signal', '-s', type=str, default='all',
                        help='Signal type (resist, solid, arrow, collect, climb, mount, or all)')
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Path to preprocessed data directory')
    parser.add_argument('--output', '-o', type=str, default='results/charts',
                        help='Output directory for charts')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CHART GENERATOR - All Signal Combinations")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Signal: {args.signal}")
    
    if args.signal.lower() == 'all':
        signals_to_process = list(SIGNAL_CONDITIONS.keys())
    else:
        signals_to_process = [args.signal.lower()]
    
    for signal in signals_to_process:
        if signal not in SIGNAL_CONDITIONS:
            print(f"Unknown signal: {signal}")
            continue
        process_signal(signal, args.data, args.output)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

