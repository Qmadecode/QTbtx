#!/usr/bin/env python3
"""
QTbtx Signal Optimizer - Main Entry Point

Usage:
    python run_optimizer.py --signal resist --data /path/to/data
    python run_optimizer.py --signal all --data /path/to/data
"""

import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

from optimizers import SignalOptimizer, SIGNAL_CONDITIONS


def export_to_excel(df: pd.DataFrame, output_path: str, signal_type: str, conditions: list):
    """Export results to Excel with multiple sheets"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Combinations', index=False)
        df.nlargest(50, 'total_return_%').to_excel(writer, sheet_name='Top_50_Return', index=False)
        
        if len(df[df['total_trades'] >= 10]) > 0:
            df[df['total_trades'] >= 10].nlargest(50, 'sharpe').to_excel(writer, sheet_name='Top_50_Sharpe', index=False)
        
        by_num = df.groupby('num_conditions').agg({
            'total_return_%': 'mean',
            'total_trades': 'mean',
            'fill_rate_%': 'mean',
            'win_rate_%': 'mean',
            'sharpe': 'mean',
        }).round(2)
        by_num.to_excel(writer, sheet_name='By_Num_Conditions')
        
        # Condition importance
        importance = []
        for i, cond in enumerate(conditions):
            mask = df['combo_id'].apply(lambda x: bool(x & (1 << i)))
            on_return = df[mask]['total_return_%'].mean()
            off_return = df[~mask]['total_return_%'].mean()
            importance.append({
                'condition': cond,
                'avg_return_ON': round(on_return, 2),
                'avg_return_OFF': round(off_return, 2),
                'impact': round(on_return - off_return, 2),
            })
        importance_df = pd.DataFrame(importance).sort_values('impact', ascending=False)
        importance_df.to_excel(writer, sheet_name='Condition_Importance', index=False)
    
    print(f"Excel saved: {output_path}")


def run_single_signal(signal_type: str, data_path: str, output_dir: str):
    """Run optimization for a single signal type"""
    print(f"\n{'='*70}")
    print(f"OPTIMIZING: {signal_type.upper()}")
    print(f"{'='*70}")
    
    optimizer = SignalOptimizer(signal_type=signal_type)
    optimizer.load_data(data_path)
    results_df = optimizer.run_all_combinations()
    
    # Export results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'{output_dir}/{signal_type}_optimization_{timestamp}.xlsx'
    export_to_excel(results_df, output_file, signal_type, optimizer.conditions)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY - {signal_type.upper()}")
    print(f"{'='*60}")
    print(f"Total combinations: {len(results_df)}")
    print(f"With trades: {len(results_df[results_df['total_trades'] > 0])}")
    print(f"Positive return: {len(results_df[results_df['total_return_%'] > 0])}")
    
    if len(results_df) > 0:
        best = results_df.nlargest(1, 'total_return_%').iloc[0]
        print(f"\nBest by Return:")
        print(f"  Combo #{best['combo_id']}: {best['total_return_%']:.2f}%")
        print(f"  Conditions: {best['conditions']}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='QTbtx Signal Optimizer')
    parser.add_argument('--signal', type=str, required=True, 
                        help='Signal type (mount, climb, arrow, collect, solid, resist) or "all"')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to preprocessed data directory')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("QTbtx SIGNAL OPTIMIZER")
    print("="*70)
    print(f"Data path: {args.data}")
    print(f"Output: {args.output}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.signal.lower() == 'all':
        # Run all signals
        all_results = {}
        for signal_type in SIGNAL_CONDITIONS.keys():
            results = run_single_signal(signal_type, args.data, args.output)
            all_results[signal_type] = results
        
        # Summary
        print(f"\n{'='*70}")
        print("ALL SIGNALS COMPLETED")
        print(f"{'='*70}")
        for signal_type, df in all_results.items():
            best = df.nlargest(1, 'total_return_%').iloc[0]
            print(f"{signal_type.upper():10s}: Best return {best['total_return_%']:.2f}%")
    else:
        run_single_signal(args.signal.lower(), args.data, args.output)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()

