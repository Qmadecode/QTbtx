#!/usr/bin/env python3
"""
Run optimization for all signals on Google Cloud
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimizers import SignalOptimizer, SIGNAL_CONDITIONS


def main():
    # Configuration from environment
    DATA_PATH = os.environ.get('DATA_PATH', '/data/preprocessed')
    OUTPUT_PATH = os.environ.get('OUTPUT_PATH', '/output')
    SIGNALS = os.environ.get('SIGNALS', 'all').lower().split(',')
    
    print("="*70)
    print("QTbtx CLOUD OPTIMIZER")
    print("="*70)
    print(f"Data: {DATA_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Signals: {SIGNALS}")
    print(f"Started: {datetime.now().isoformat()}")
    
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    
    # Determine which signals to run
    if 'all' in SIGNALS:
        signals_to_run = list(SIGNAL_CONDITIONS.keys())
    else:
        signals_to_run = [s.strip() for s in SIGNALS if s.strip() in SIGNAL_CONDITIONS]
    
    print(f"\nRunning: {signals_to_run}")
    
    results_summary = {}
    
    for signal_type in signals_to_run:
        print(f"\n{'='*60}")
        print(f"OPTIMIZING: {signal_type.upper()}")
        print(f"{'='*60}")
        
        try:
            optimizer = SignalOptimizer(signal_type=signal_type)
            optimizer.load_data(DATA_PATH)
            results_df = optimizer.run_all_combinations()
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'{OUTPUT_PATH}/{signal_type}_optimization_{timestamp}.csv'
            results_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")
            
            # Summary
            best = results_df.nlargest(1, 'total_return_%').iloc[0]
            results_summary[signal_type] = {
                'combinations': len(results_df),
                'best_return': best['total_return_%'],
                'best_combo': best['combo_id'],
            }
            
        except Exception as e:
            print(f"ERROR: {signal_type} failed: {e}")
            results_summary[signal_type] = {'error': str(e)}
    
    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for signal, data in results_summary.items():
        if 'error' in data:
            print(f"{signal.upper():10s}: ERROR - {data['error']}")
        else:
            print(f"{signal.upper():10s}: {data['combinations']} combos, best: {data['best_return']:.2f}%")
    
    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()

