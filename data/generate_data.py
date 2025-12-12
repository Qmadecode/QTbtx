#!/usr/bin/env python3
"""
Generate preprocessed indicator data for US100 stocks using FMP API.
Downloads price data via Financial Modeling Prep and calculates all required indicators.
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# FMP API Key - set as environment variable or replace here
FMP_API_KEY = os.environ.get('FMP_API_KEY', 'YOUR_FMP_API_KEY')

# US100 Constituents (NASDAQ-100)
US100_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'TSLA', 'AVGO', 'COST',
    'NFLX', 'AMD', 'PEP', 'ADBE', 'CSCO', 'TMUS', 'INTC', 'CMCSA', 'TXN', 'QCOM',
    'AMGN', 'HON', 'INTU', 'AMAT', 'ISRG', 'BKNG', 'SBUX', 'VRTX', 'LRCX', 'ADP',
    'MDLZ', 'GILD', 'ADI', 'REGN', 'PANW', 'MU', 'KLAC', 'SNPS', 'CDNS', 'ASML',
    'MELI', 'PYPL', 'ORLY', 'CTAS', 'MAR', 'MNST', 'NXPI', 'ABNB', 'MRVL', 'FTNT',
    'CSX', 'PCAR', 'WDAY', 'CHTR', 'CPRT', 'DXCM', 'AEP', 'KDP', 'ROST', 'MCHP',
    'ADSK', 'KHC', 'PAYX', 'ODFL', 'AZN', 'LULU', 'EXC', 'IDXX', 'FAST', 'VRSK',
    'CTSH', 'EA', 'BIIB', 'GEHC', 'XEL', 'CSGP', 'DDOG', 'TEAM', 'ANSS', 'ON',
    'FANG', 'DLTR', 'WBD', 'ZS', 'ILMN', 'BKR', 'TTD', 'MDB', 'GFS', 'ARM',
    'CRWD', 'CDW', 'SPLK', 'SIRI', 'LCID', 'RIVN', 'WBA', 'JD', 'PDD', 'EBAY'
]


def calculate_ema(series, period):
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsl(price, benchmark, period=10):
    """Calculate Relative Strength Line."""
    ratio = price / benchmark
    rsl = (ratio / ratio.rolling(period).mean()) * 100
    return rsl


def calculate_cyber_cycle(close, period=10):
    """
    Calculate CyberCycle indicator.
    Simplified version using smoothed momentum.
    """
    smooth = close.ewm(span=period, adjust=False).mean()
    cycle = smooth - smooth.shift(period)
    return cycle


def calculate_swing_value(high, low, close, period=5):
    """Calculate Swing Value indicator."""
    tr = np.maximum(high - low, 
                    np.maximum(abs(high - close.shift(1)), 
                              abs(low - close.shift(1))))
    atr = tr.rolling(period).mean()
    swing = (close - close.shift(period)) / atr
    return swing.fillna(0)


def calculate_vindex(high, low, close, period=10):
    """Calculate VIndex (Volatility Index)."""
    tr = np.maximum(high - low,
                    np.maximum(abs(high - close.shift(1)),
                              abs(low - close.shift(1))))
    vindex = tr.rolling(period).mean() / close * 100
    return vindex


def calculate_itrend(close, period=20):
    """Calculate iTrend indicator."""
    ema_fast = calculate_ema(close, period // 2)
    ema_slow = calculate_ema(close, period)
    itrend = ema_fast - ema_slow
    return itrend


def download_fmp_data(ticker, start_date, end_date, api_key):
    """Download OHLCV data from FMP for a single ticker."""
    try:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
        params = {
            'apikey': api_key,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'historical' not in data or len(data['historical']) < 100:
            print(f"  Skipping {ticker}: insufficient data")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data['historical'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        
        # Rename columns to standard format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        print(f"  Error downloading {ticker}: {e}")
        return None


def download_benchmark(start_date, end_date, api_key):
    """Download NASDAQ-100 benchmark data (QQQ)."""
    print("Downloading benchmark (QQQ)...")
    
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/QQQ"
    params = {
        'apikey': api_key,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d')
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data['historical'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    
    return df['close']


def process_ticker(ticker, data, benchmark_close):
    """Calculate all indicators for a single ticker."""
    df = pd.DataFrame()
    
    # Basic OHLCV
    df['date'] = data.index
    df['open'] = data['Open'].values
    df['high'] = data['High'].values
    df['low'] = data['Low'].values
    df['close'] = data['Close'].values
    df['volume'] = data['Volume'].values
    df['ticker'] = ticker
    
    # Set index
    df = df.set_index('date')
    
    # Calculate True Range
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Calculate ATR
    df['atr'] = df['tr'].rolling(14).mean()
    
    # CyberCycle
    df['cyber_cycle'] = calculate_cyber_cycle(df['close'])
    
    # Swing Value and VIndex
    df['swing_value'] = calculate_swing_value(df['high'], df['low'], df['close'])
    df['vindex'] = calculate_vindex(df['high'], df['low'], df['close'])
    
    # iTrend (Daily and Weekly approximation)
    df['itrend_d'] = calculate_itrend(df['close'], period=20)
    df['itrend_w'] = calculate_itrend(df['close'], period=100)  # ~5 weeks
    
    # RSL calculations
    benchmark_aligned = benchmark_close.reindex(df.index).ffill()
    
    df['rsl_close'] = calculate_rsl(df['close'], benchmark_aligned).values
    df['rsl_itrend_d'] = calculate_rsl(df['close'], benchmark_aligned, period=20).values
    df['rsl_itrend_w'] = calculate_rsl(df['close'], benchmark_aligned, period=100).values
    
    # Weekly low (rolling 5 days)
    df['weekly_low'] = df['low'].rolling(5).min()
    
    # Previous values
    df['close_prev'] = df['close'].shift(1)
    df['close_prev2'] = df['close'].shift(2)
    df['low_prev'] = df['low'].shift(1)
    df['high_prev'] = df['high'].shift(1)
    
    # Swing trend
    df['swing_trend'] = np.where(df['swing_value'] > 0, 1, -1)
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(0)
    
    return df


def generate_all_data(output_dir, years=3, api_key=None):
    """Generate preprocessed data for all US100 tickers."""
    if api_key is None:
        api_key = FMP_API_KEY
    
    if api_key == 'YOUR_FMP_API_KEY':
        print("ERROR: Please set your FMP API key!")
        print("  Option 1: export FMP_API_KEY='your_key_here'")
        print("  Option 2: python generate_data.py --api-key your_key_here")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365 + 100)  # Extra for warmup
    
    print(f"Generating data from {start_date.date()} to {end_date.date()}")
    print(f"Output directory: {output_path}")
    print(f"Tickers to process: {len(US100_TICKERS)}")
    print()
    
    # Download benchmark
    benchmark_close = download_benchmark(start_date, end_date, api_key)
    
    # Process each ticker
    successful = 0
    failed = 0
    
    for i, ticker in enumerate(US100_TICKERS, 1):
        print(f"[{i}/{len(US100_TICKERS)}] Processing {ticker}...", end=" ")
        
        # Download data
        data = download_fmp_data(ticker, start_date, end_date, api_key)
        if data is None:
            failed += 1
            continue
        
        # Calculate indicators
        try:
            df = process_ticker(ticker, data, benchmark_close)
            
            # Save to CSV
            output_file = output_path / f"{ticker}_indicators.csv"
            df.to_csv(output_file)
            print(f"OK ({len(df)} rows)")
            successful += 1
            
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
    
    print()
    print("=" * 50)
    print(f"Data generation complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate preprocessed indicator data using FMP')
    parser.add_argument('--output', '-o', type=str, default='preprocessed',
                        help='Output directory for CSV files')
    parser.add_argument('--years', '-y', type=int, default=3,
                        help='Years of historical data to download')
    parser.add_argument('--api-key', '-k', type=str, default=None,
                        help='FMP API key (or set FMP_API_KEY env variable)')
    
    args = parser.parse_args()
    
    generate_all_data(args.output, args.years, args.api_key)
