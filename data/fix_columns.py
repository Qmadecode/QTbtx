#!/usr/bin/env python3
"""Quick fix to rename lowercase columns to uppercase in existing CSV files."""

import pandas as pd
from pathlib import Path
import sys

def fix_columns(data_dir):
    """Rename lowercase OHLCV columns to uppercase."""
    data_path = Path(data_dir)
    files = list(data_path.glob('*_indicators.csv'))
    
    print(f"Found {len(files)} files to fix")
    
    column_map = {
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    
    for i, f in enumerate(files, 1):
        df = pd.read_csv(f, index_col=0)
        
        # Check if already uppercase
        if 'Open' in df.columns:
            print(f"[{i}/{len(files)}] {f.stem} - already fixed")
            continue
            
        # Rename columns
        df = df.rename(columns=column_map)
        df.to_csv(f)
        print(f"[{i}/{len(files)}] {f.stem} - fixed")
    
    print("\nDone!")

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/preprocessed'
    fix_columns(data_dir)

