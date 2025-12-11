"""
Base Optimizer - Core optimization logic for all signals
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .signal_definitions import (
    SIGNAL_CONDITIONS, TRADE_PARAMS, get_signal_config,
    compute_condition_mask, get_enabled_conditions
)
from .trade_logic import calculate_entry_limit, check_exit_conditions


@dataclass
class CombinationResult:
    """Results for a single feature combination"""
    combo_id: int
    combo_binary: str
    conditions_enabled: List[str]
    num_conditions: int
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    total_signals: int = 0
    signals_filled: int = 0
    fill_rate: float = 0.0
    stop_loss_exits: int = 0
    trailing_exits: int = 0
    time_exits: int = 0
    target_exits: int = 0
    trades_per_year: float = 0.0
    avg_trade_pnl: float = 0.0


class SignalOptimizer:
    """
    Universal signal optimizer - works for any signal type
    
    Usage:
        optimizer = SignalOptimizer(signal_type='resist')
        optimizer.load_data('/path/to/data')
        results = optimizer.run_all_combinations()
    """
    
    def __init__(
        self,
        signal_type: str,
        start_date: str = '2022-12-01',
        end_date: str = 'open',
        initial_capital: float = 2_000_000,
        max_positions: int = 20,
        position_size_pct: float = 5.0,
        rsl_threshold: float = 100,
        cyber_threshold: float = 0,
        trade_params: Dict = None,
    ):
        # Get signal configuration
        self.signal_config = get_signal_config(signal_type)
        self.signal_type = signal_type.lower()
        self.conditions = self.signal_config['conditions']
        self.num_conditions = len(self.conditions)
        self.total_combinations = 2 ** self.num_conditions - 1
        
        # Backtest config
        self.start_date = pd.to_datetime(start_date)
        self.end_date = datetime.now() if end_date == 'open' else pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.rsl_threshold = rsl_threshold
        self.cyber_threshold = cyber_threshold
        
        # Trade params (use signal defaults if not provided)
        self.trade_params = trade_params or TRADE_PARAMS.get(self.signal_type, {
            'exitbarslong': 10,
            'stop_loss_pct': 0.05,
            'trailing_trigger': 20.0,
        })
        
        # Config dict for condition computation
        self.config = {
            'rsl_threshold': rsl_threshold,
            'cyber_threshold': cyber_threshold,
            'initial_capital': initial_capital,
            'max_positions': max_positions,
            'position_size_pct': position_size_pct,
        }
        
        # Data storage
        self.all_data: Dict[str, pd.DataFrame] = {}
        self.trading_days: pd.DatetimeIndex = None
        self.condition_masks: Dict[str, Dict[str, pd.Series]] = {}
        self.results: List[CombinationResult] = []
    
    def load_data(self, data_dir: str):
        """Load preprocessed indicator data"""
        print(f"Loading data for {self.signal_type.upper()} signal...")
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        files = list(data_path.glob('*_indicators.csv'))
        print(f"Found {len(files)} indicator files")
        
        for file in files:
            ticker = file.stem.replace('_indicators', '')
            try:
                df = pd.read_csv(file, index_col=0, parse_dates=True)
                if len(df) > 50:
                    self.all_data[ticker] = df
            except Exception as e:
                print(f"Error loading {ticker}: {e}")
        
        print(f"Loaded {len(self.all_data)} tickers")
        
        # Get trading days
        all_dates = set()
        for df in self.all_data.values():
            dates = df[(df.index >= self.start_date) & (df.index <= self.end_date)].index
            all_dates.update(dates)
        
        self.trading_days = pd.DatetimeIndex(sorted(all_dates))
        print(f"Trading days: {len(self.trading_days)}")
    
    def compute_condition_masks(self):
        """Pre-compute condition masks for all tickers"""
        print(f"Computing {len(self.conditions)} condition masks...")
        
        for cond in self.conditions:
            self.condition_masks[cond] = {}
        
        for ticker, df in self.all_data.items():
            mask_df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            if len(mask_df) < 20:
                continue
            
            try:
                for cond in self.conditions:
                    self.condition_masks[cond][ticker] = compute_condition_mask(
                        mask_df, cond, self.config
                    )
            except Exception as e:
                print(f"Error computing masks for {ticker}: {e}")
                continue
        
        print(f"Computed masks for {len(self.condition_masks[self.conditions[0]])} tickers")
    
    def get_signals_for_combo(self, combo_id: int) -> Dict[str, pd.Series]:
        """Get combined signal mask for a combination"""
        signals = {}
        enabled = get_enabled_conditions(combo_id, self.conditions)
        
        if not enabled:
            return signals
        
        for ticker in self.condition_masks[self.conditions[0]].keys():
            combined = None
            for cond in enabled:
                if ticker not in self.condition_masks[cond]:
                    combined = None
                    break
                mask = self.condition_masks[cond][ticker]
                combined = mask.copy() if combined is None else (combined & mask)
            
            if combined is not None and combined.any():
                signals[ticker] = combined
        
        return signals
    
    def simulate_combination(self, combo_id: int) -> CombinationResult:
        """Run realistic portfolio simulation for a combination"""
        enabled = get_enabled_conditions(combo_id, self.conditions)
        binary_str = format(combo_id, f'0{self.num_conditions}b')
        
        result = CombinationResult(
            combo_id=combo_id,
            combo_binary=binary_str,
            conditions_enabled=enabled,
            num_conditions=len(enabled),
        )
        
        if not enabled:
            return result
        
        signals = self.get_signals_for_combo(combo_id)
        if not signals:
            return result
        
        result.total_signals = int(sum(m.sum() for m in signals.values()))
        
        # Run simulation
        nav_history, trades, signal_stats = self._run_simulation(signals)
        
        # Calculate metrics
        if len(nav_history) > 1:
            nav = pd.Series(nav_history)
            initial, final = nav.iloc[0], nav.iloc[-1]
            result.total_return = (final / initial - 1) * 100
            
            years = len(self.trading_days) / 252
            if years > 0:
                result.cagr = ((final / initial) ** (1/years) - 1) * 100
                result.trades_per_year = len(trades) / years
            
            returns = nav.pct_change().dropna()
            if len(returns) > 0:
                result.volatility = returns.std() * np.sqrt(252) * 100
                if result.volatility > 0:
                    result.sharpe_ratio = result.cagr / result.volatility
                
                cumul = (1 + returns).cumprod()
                dd = (cumul - cumul.expanding().max()) / cumul.expanding().max()
                result.max_drawdown = dd.min() * 100
        
        result.total_trades = len(trades)
        result.signals_filled = signal_stats['filled']
        result.fill_rate = (signal_stats['filled'] / signal_stats['generated'] * 100) if signal_stats['generated'] > 0 else 0
        result.stop_loss_exits = signal_stats.get('stop_loss', 0)
        result.trailing_exits = signal_stats.get('trailing', 0)
        result.time_exits = signal_stats.get('time_exit', 0)
        result.target_exits = signal_stats.get('target_exit', 0)
        
        if trades:
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]
            result.win_rate = len(wins) / len(trades) * 100
            result.avg_trade_pnl = sum(t['pnl'] for t in trades) / len(trades)
            
            total_wins = sum(t['pnl'] for t in wins) if wins else 0
            total_losses = abs(sum(t['pnl'] for t in losses)) if losses else 0
            if total_losses > 0:
                result.profit_factor = total_wins / total_losses
        
        return result
    
    def _run_simulation(self, signals: Dict[str, pd.Series]) -> Tuple[List[float], List[Dict], Dict]:
        """Run portfolio simulation with realistic logic"""
        cash = self.initial_capital
        positions = {}
        nav_history = [self.initial_capital]
        trades = []
        signal_stats = {'generated': 0, 'filled': 0, 'stop_loss': 0, 'trailing': 0, 'time_exit': 0, 'target_exit': 0}
        position_value = self.initial_capital * (self.position_size_pct / 100)
        
        for day_idx, date in enumerate(self.trading_days):
            if day_idx < 3:
                continue
            
            # Get current OHLC
            current_ohlc = {}
            for ticker, df in self.all_data.items():
                if date in df.index:
                    row = df.loc[date]
                    current_ohlc[ticker] = {'Open': row['Open'], 'High': row['High'], 'Low': row['Low'], 'Close': row['Close']}
            
            # Update position runup
            for ticker, pos in positions.items():
                if ticker in current_ohlc:
                    price = current_ohlc[ticker]['Close']
                    pnl_pct = (price / pos['entry_price'] - 1) * 100
                    if pnl_pct > pos['runup']:
                        pos['runup'] = pnl_pct
            
            # Process exits
            positions_to_close = []
            for ticker, pos in positions.items():
                if ticker not in self.all_data or ticker not in current_ohlc:
                    continue
                ticker_data = self.all_data[ticker]
                data_through_today = ticker_data[ticker_data.index <= date]
                if len(data_through_today) < 3:
                    continue
                
                should_exit, exit_type, exit_price = check_exit_conditions(pos, data_through_today, self.trade_params)
                
                if should_exit:
                    if exit_price == "NEXT_OPEN":
                        actual_exit_price = current_ohlc[ticker]['Open']
                    else:
                        high_today = current_ohlc[ticker]['High']
                        open_today = current_ohlc[ticker]['Open']
                        if open_today >= exit_price:
                            actual_exit_price = open_today
                        elif high_today >= exit_price:
                            actual_exit_price = exit_price
                        else:
                            if exit_type == "Time Exit":
                                actual_exit_price = current_ohlc[ticker]['Close']
                            else:
                                continue
                    positions_to_close.append((ticker, actual_exit_price, exit_type))
            
            for ticker, exit_price, exit_type in positions_to_close:
                pos = positions[ticker]
                pnl = (exit_price - pos['entry_price']) * pos['quantity']
                trades.append({'ticker': ticker, 'entry_price': pos['entry_price'], 'exit_price': exit_price, 'pnl': pnl, 'exit_type': exit_type})
                
                if 'Stop' in exit_type:
                    signal_stats['stop_loss'] += 1
                elif 'Trailing' in exit_type:
                    signal_stats['trailing'] += 1
                elif 'Time' in exit_type:
                    signal_stats['time_exit'] += 1
                elif 'Target' in exit_type:
                    signal_stats['target_exit'] += 1
                
                cash += pos['quantity'] * exit_price
                del positions[ticker]
            
            # Process entries (T-1 signals)
            available_slots = self.max_positions - len(positions)
            if available_slots > 0 and day_idx >= 1:
                yesterday = self.trading_days[day_idx - 1]
                
                for ticker, signal_mask in signals.items():
                    if available_slots <= 0:
                        break
                    if yesterday not in signal_mask.index or not signal_mask.loc[yesterday]:
                        continue
                    if ticker in positions or ticker not in current_ohlc or ticker not in self.all_data:
                        continue
                    
                    signal_stats['generated'] += 1
                    ticker_data = self.all_data[ticker]
                    data_through_today = ticker_data[ticker_data.index <= date]
                    if len(data_through_today) < 3:
                        continue
                    
                    entry_limit = calculate_entry_limit(data_through_today, signal_bar_idx=-2)
                    if entry_limit is None:
                        continue
                    
                    low_today = current_ohlc[ticker]['Low']
                    open_today = current_ohlc[ticker]['Open']
                    
                    if low_today <= entry_limit:
                        fill_price = min(open_today, entry_limit)
                        quantity = int(position_value / fill_price)
                        if quantity > 0 and cash >= quantity * fill_price:
                            positions[ticker] = {'ticker': ticker, 'entry_price': fill_price, 'entry_date': date, 'quantity': quantity, 'runup': 0.0}
                            cash -= quantity * fill_price
                            available_slots -= 1
                            signal_stats['filled'] += 1
            
            # Calculate NAV
            positions_value = sum(pos['quantity'] * current_ohlc.get(ticker, {}).get('Close', pos['entry_price']) for ticker, pos in positions.items())
            nav_history.append(cash + positions_value)
        
        return nav_history, trades, signal_stats
    
    def run_all_combinations(self) -> pd.DataFrame:
        """Test all combinations"""
        if not self.condition_masks:
            self.compute_condition_masks()
        
        print(f"\n{'='*60}")
        print(f"{self.signal_type.upper()} OPTIMIZATION - {self.total_combinations} COMBINATIONS")
        print(f"{'='*60}")
        
        start = datetime.now()
        
        for combo_id in range(1, self.total_combinations + 1):
            if combo_id % max(1, self.total_combinations // 10) == 0:
                elapsed = (datetime.now() - start).total_seconds()
                rate = combo_id / elapsed if elapsed > 0 else 0
                remaining = (self.total_combinations - combo_id) / rate if rate > 0 else 0
                print(f"Progress: {combo_id}/{self.total_combinations} ({rate:.1f}/s, ~{remaining:.0f}s remaining)")
            
            result = self.simulate_combination(combo_id)
            self.results.append(result)
        
        elapsed = (datetime.now() - start).total_seconds()
        print(f"\nCompleted in {elapsed:.1f}s")
        
        # Create DataFrame
        df = pd.DataFrame([{
            'combo_id': r.combo_id,
            'binary': r.combo_binary,
            'conditions': ', '.join(r.conditions_enabled),
            'num_conditions': r.num_conditions,
            'total_return_%': round(r.total_return, 2),
            'cagr_%': round(r.cagr, 2),
            'sharpe': round(r.sharpe_ratio, 2),
            'max_dd_%': round(r.max_drawdown, 2),
            'volatility_%': round(r.volatility, 2),
            'win_rate_%': round(r.win_rate, 1),
            'profit_factor': round(r.profit_factor, 2),
            'total_trades': r.total_trades,
            'signals_generated': r.total_signals,
            'signals_filled': r.signals_filled,
            'fill_rate_%': round(r.fill_rate, 1),
            'stop_loss_exits': r.stop_loss_exits,
            'trailing_exits': r.trailing_exits,
            'time_exits': r.time_exits,
            'target_exits': r.target_exits,
            'trades_per_year': round(r.trades_per_year, 1),
            'avg_trade_pnl': round(r.avg_trade_pnl, 2),
        } for r in self.results])
        
        return df

