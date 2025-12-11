"""
QTbtx Signal Optimizer - Google Colab Version
==============================================

INSTRUCTIONS:
1. Mount Google Drive
2. Set DATA_PATH and SIGNAL_TYPE below
3. Run all cells

Supported signals: mount, climb, arrow, collect, solid, resist
"""

# ============================================================
# CELL 1: Mount Drive (run this first!)
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')

# ============================================================
# CONFIGURATION - EDIT THIS!
# ============================================================

# Your data path on Google Drive
DATA_PATH = '/content/drive/MyDrive/Preprocessed data/preprocessed'

# Output path
OUTPUT_PATH = '/content/drive/MyDrive/Preprocessed data/optimization_results'

# Which signal to optimize: 'mount', 'climb', 'arrow', 'collect', 'solid', 'resist', or 'all'
SIGNAL_TYPE = 'resist'

# ============================================================
# IMPORTS AND SETUP
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SIGNAL DEFINITIONS
# ============================================================

ALL_CONDITIONS = [
    'cyber_negative', 'value_gt_vindex', 'rsl_itrend_w_weak',
    'rsl_itrend_d_weak', 'rsl_close_weak', 'lower_low',
    'lower_close', 'lower_close_prev', 'below_weekly_low',
]

SIGNAL_CONDITIONS = {
    'mount': ['cyber_negative', 'value_gt_vindex'],
    'climb': ['cyber_negative', 'value_gt_vindex'],
    'arrow': ['lower_close', 'cyber_negative', 'value_gt_vindex', 'rsl_itrend_w_weak', 'rsl_itrend_d_weak'],
    'collect': ['lower_close', 'cyber_negative', 'value_gt_vindex', 'rsl_itrend_w_weak', 'rsl_itrend_d_weak'],
    'solid': ALL_CONDITIONS,
    'resist': ALL_CONDITIONS,
}

TRADE_PARAMS = {
    'mount': {'exitbarslong': 10, 'stop_loss_pct': 0.05, 'trailing_trigger': 20.0},
    'climb': {'exitbarslong': 10, 'stop_loss_pct': 0.05, 'trailing_trigger': 20.0},
    'arrow': {'exitbarslong': 10, 'stop_loss_pct': 0.05, 'trailing_trigger': 20.0},
    'collect': {'exitbarslong': 10, 'stop_loss_pct': 0.05, 'trailing_trigger': 20.0},
    'solid': {'exitbarslong': 10, 'stop_loss_pct': 0.07, 'trailing_trigger': 12.0},
    'resist': {'exitbarslong': 10, 'stop_loss_pct': 0.05, 'trailing_trigger': 20.0},
}

CONFIG = {
    'start_date': '2022-12-01',
    'end_date': 'open',
    'initial_capital': 2_000_000,
    'max_positions': 20,
    'position_size_pct': 5.0,
    'rsl_threshold': 100,
    'cyber_threshold': 0,
}

# ============================================================
# TRADE LOGIC
# ============================================================

def calculate_true_range(data):
    high, low = data['High'], data['Low']
    close_prev = data['Close'].shift(1)
    return pd.Series(np.maximum.reduce([high - low, (high - close_prev).abs(), (low - close_prev).abs()]), index=data.index)

def calculate_atr(data, period=10):
    return calculate_true_range(data).rolling(window=period).mean()

def calculate_entry_limit(data, signal_bar_idx=-2):
    if len(data) < 3:
        return None
    if 'entry_limit' in data.columns:
        el = data['entry_limit'].iloc[signal_bar_idx]
        if not pd.isna(el):
            return round(el, 2)
    low = data['Low'].iloc[signal_bar_idx]
    tr = data['tr'].iloc[signal_bar_idx - 1] if 'tr' in data.columns else calculate_true_range(data).iloc[signal_bar_idx - 1]
    if pd.isna(low) or pd.isna(tr):
        return None
    return round(low - tr / 4, 2)

def check_exit_conditions(position, current_data, trade_params):
    if len(current_data) < 3:
        return False, "", 0.0
    close_t1 = current_data['Close'].iloc[-2]
    entry_price, runup, entry_date = position['entry_price'], position['runup'], position['entry_date']
    if pd.isna(close_t1):
        return False, "", 0.0
    bars_held = (current_data.index[-2] - entry_date).days
    stop_level = entry_price * (1 - trade_params['stop_loss_pct'])
    
    if close_t1 < stop_level:
        return True, "Stop Loss", "NEXT_OPEN"
    
    if runup > trade_params['trailing_trigger']:
        hist = current_data.iloc[:-1]
        high_since = hist.loc[hist.index >= entry_date, 'High'].max()
        pct = 0.20 if runup >= 30 else (0.15 if runup >= 25 else (0.10 if runup >= 20 else (high_since - entry_price) / high_since))
        if close_t1 < high_since * (1 - pct):
            return True, "Trailing Stop", "NEXT_OPEN"
    
    if bars_held > trade_params['exitbarslong']:
        hist = current_data.iloc[:-1]
        te = hist['time_exit'].iloc[-1] if 'time_exit' in hist.columns else None
        if te is None or pd.isna(te):
            sv = hist.get('swing_value', pd.Series()).iloc[-1] if 'swing_value' in hist.columns else None
            if sv and not pd.isna(sv):
                a5 = hist['atr_5'].iloc[-1] if 'atr_5' in hist.columns else calculate_atr(hist, 5).iloc[-1]
                if not pd.isna(a5):
                    te = round(sv + a5, 2)
        return True, "Time Exit", te if te and not pd.isna(te) else round(close_t1, 2)
    
    hist = current_data.iloc[:-1]
    tp = hist['target_exit'].iloc[-1] if 'target_exit' in hist.columns else None
    if tp is None or pd.isna(tp):
        sv = hist.get('swing_value', pd.Series()).iloc[-1] if 'swing_value' in hist.columns else None
        if sv and not pd.isna(sv):
            a10 = hist['atr_10'].iloc[-1] if 'atr_10' in hist.columns else calculate_atr(hist, 10).iloc[-1]
            if not pd.isna(a10):
                tp = sv + a10
    if tp and not pd.isna(tp) and close_t1 >= tp:
        return True, "Target Exit", round(hist['High'].iloc[-1], 2)
    
    return False, "", 0.0

# ============================================================
# CONDITION MASKS
# ============================================================

def compute_condition_mask(ticker_data, condition, config):
    rsl_th, cyber_th = config.get('rsl_threshold', 100), config.get('cyber_threshold', 0)
    idx = ticker_data.index
    
    if condition == 'cyber_negative':
        return (ticker_data.get('cyber_cycle', pd.Series(np.nan, index=idx)) < cyber_th).fillna(False)
    elif condition == 'value_gt_vindex':
        return (ticker_data.get('swing_value', pd.Series(np.nan, index=idx)) > ticker_data.get('vindex', pd.Series(np.nan, index=idx))).fillna(False)
    elif condition == 'rsl_itrend_w_weak':
        return (ticker_data.get('rsl_itrend_w', pd.Series(np.nan, index=idx)) < rsl_th).fillna(False)
    elif condition == 'rsl_itrend_d_weak':
        return (ticker_data.get('rsl_itrend_d', pd.Series(np.nan, index=idx)) < rsl_th).fillna(False)
    elif condition == 'rsl_close_weak':
        return (ticker_data.get('rsl_close', pd.Series(np.nan, index=idx)) < rsl_th).fillna(False)
    elif condition == 'lower_low':
        return (ticker_data['Low'] < ticker_data['Low'].shift(1)).fillna(False)
    elif condition == 'lower_close':
        return (ticker_data['Close'] < ticker_data['Close'].shift(1)).fillna(False)
    elif condition == 'lower_close_prev':
        return (ticker_data['Close'].shift(1) < ticker_data['Close'].shift(2)).fillna(False)
    elif condition == 'below_weekly_low':
        return (ticker_data['Close'] < ticker_data.get('weekly_low', pd.Series(np.nan, index=idx))).fillna(False)
    raise ValueError(f"Unknown condition: {condition}")

# ============================================================
# OPTIMIZER CLASS
# ============================================================

@dataclass
class CombinationResult:
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
    def __init__(self, signal_type, config=CONFIG):
        self.signal_type = signal_type.lower()
        self.conditions = SIGNAL_CONDITIONS[self.signal_type]
        self.trade_params = TRADE_PARAMS[self.signal_type]
        self.config = config
        self.start_date = pd.to_datetime(config['start_date'])
        self.end_date = datetime.now() if config['end_date'] == 'open' else pd.to_datetime(config['end_date'])
        self.all_data = {}
        self.trading_days = None
        self.condition_masks = {c: {} for c in self.conditions}
        self.results = []
    
    def load_data(self, data_dir):
        print(f"Loading data for {self.signal_type.upper()}...")
        files = list(Path(data_dir).glob('*_indicators.csv'))
        print(f"Found {len(files)} files")
        for f in files:
            ticker = f.stem.replace('_indicators', '')
            try:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                if len(df) > 50:
                    self.all_data[ticker] = df
            except:
                pass
        print(f"Loaded {len(self.all_data)} tickers")
        all_dates = set()
        for df in self.all_data.values():
            dates = df[(df.index >= self.start_date) & (df.index <= self.end_date)].index
            all_dates.update(dates)
        self.trading_days = pd.DatetimeIndex(sorted(all_dates))
        print(f"Trading days: {len(self.trading_days)}")
    
    def compute_masks(self):
        print("Computing condition masks...")
        for ticker, df in self.all_data.items():
            mdf = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            if len(mdf) < 20:
                continue
            try:
                for cond in self.conditions:
                    self.condition_masks[cond][ticker] = compute_condition_mask(mdf, cond, self.config)
            except:
                pass
        print(f"Computed masks for {len(self.condition_masks[self.conditions[0]])} tickers")
    
    def get_signals(self, combo_id):
        signals = {}
        enabled = [self.conditions[i] for i in range(len(self.conditions)) if combo_id & (1 << i)]
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
    
    def simulate(self, signals):
        cash = self.config['initial_capital']
        positions = {}
        nav_history = [cash]
        trades = []
        stats = {'generated': 0, 'filled': 0, 'stop_loss': 0, 'trailing': 0, 'time_exit': 0, 'target_exit': 0}
        pos_value = cash * (self.config['position_size_pct'] / 100)
        
        for di, date in enumerate(self.trading_days):
            if di < 3:
                continue
            ohlc = {}
            for t, df in self.all_data.items():
                if date in df.index:
                    r = df.loc[date]
                    ohlc[t] = {'Open': r['Open'], 'High': r['High'], 'Low': r['Low'], 'Close': r['Close']}
            
            for t, p in positions.items():
                if t in ohlc:
                    pnl = (ohlc[t]['Close'] / p['entry_price'] - 1) * 100
                    if pnl > p['runup']:
                        p['runup'] = pnl
            
            to_close = []
            for t, p in positions.items():
                if t not in self.all_data or t not in ohlc:
                    continue
                data = self.all_data[t][self.all_data[t].index <= date]
                if len(data) < 3:
                    continue
                should, etype, eprice = check_exit_conditions(p, data, self.trade_params)
                if should:
                    if eprice == "NEXT_OPEN":
                        actual = ohlc[t]['Open']
                    else:
                        h, o = ohlc[t]['High'], ohlc[t]['Open']
                        if o >= eprice:
                            actual = o
                        elif h >= eprice:
                            actual = eprice
                        elif etype == "Time Exit":
                            actual = ohlc[t]['Close']
                        else:
                            continue
                    to_close.append((t, actual, etype))
            
            for t, ep, et in to_close:
                p = positions[t]
                pnl = (ep - p['entry_price']) * p['quantity']
                trades.append({'ticker': t, 'pnl': pnl, 'exit_type': et})
                if 'Stop' in et:
                    stats['stop_loss'] += 1
                elif 'Trail' in et:
                    stats['trailing'] += 1
                elif 'Time' in et:
                    stats['time_exit'] += 1
                elif 'Target' in et:
                    stats['target_exit'] += 1
                cash += p['quantity'] * ep
                del positions[t]
            
            slots = self.config['max_positions'] - len(positions)
            if slots > 0 and di >= 1:
                yesterday = self.trading_days[di - 1]
                for t, mask in signals.items():
                    if slots <= 0:
                        break
                    if yesterday not in mask.index or not mask.loc[yesterday]:
                        continue
                    if t in positions or t not in ohlc or t not in self.all_data:
                        continue
                    stats['generated'] += 1
                    data = self.all_data[t][self.all_data[t].index <= date]
                    if len(data) < 3:
                        continue
                    el = calculate_entry_limit(data, -2)
                    if el is None:
                        continue
                    if ohlc[t]['Low'] <= el:
                        fp = min(ohlc[t]['Open'], el)
                        qty = int(pos_value / fp)
                        if qty > 0 and cash >= qty * fp:
                            positions[t] = {'entry_price': fp, 'entry_date': date, 'quantity': qty, 'runup': 0.0}
                            cash -= qty * fp
                            slots -= 1
                            stats['filled'] += 1
            
            pv = sum(p['quantity'] * ohlc.get(t, {}).get('Close', p['entry_price']) for t, p in positions.items())
            nav_history.append(cash + pv)
        
        return nav_history, trades, stats
    
    def run_combo(self, combo_id):
        enabled = [self.conditions[i] for i in range(len(self.conditions)) if combo_id & (1 << i)]
        binary = format(combo_id, f'0{len(self.conditions)}b')
        result = CombinationResult(combo_id, binary, enabled, len(enabled))
        if not enabled:
            return result
        signals = self.get_signals(combo_id)
        if not signals:
            return result
        result.total_signals = int(sum(m.sum() for m in signals.values()))
        nav, trades, stats = self.simulate(signals)
        if len(nav) > 1:
            s = pd.Series(nav)
            i, f = s.iloc[0], s.iloc[-1]
            result.total_return = (f / i - 1) * 100
            years = len(self.trading_days) / 252
            if years > 0:
                result.cagr = ((f / i) ** (1/years) - 1) * 100
                result.trades_per_year = len(trades) / years
            rets = s.pct_change().dropna()
            if len(rets) > 0:
                result.volatility = rets.std() * np.sqrt(252) * 100
                if result.volatility > 0:
                    result.sharpe_ratio = result.cagr / result.volatility
                c = (1 + rets).cumprod()
                dd = (c - c.expanding().max()) / c.expanding().max()
                result.max_drawdown = dd.min() * 100
        result.total_trades = len(trades)
        result.signals_filled = stats['filled']
        result.fill_rate = (stats['filled'] / stats['generated'] * 100) if stats['generated'] > 0 else 0
        result.stop_loss_exits = stats['stop_loss']
        result.trailing_exits = stats['trailing']
        result.time_exits = stats['time_exit']
        result.target_exits = stats['target_exit']
        if trades:
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]
            result.win_rate = len(wins) / len(trades) * 100
            result.avg_trade_pnl = sum(t['pnl'] for t in trades) / len(trades)
            tw = sum(t['pnl'] for t in wins) if wins else 0
            tl = abs(sum(t['pnl'] for t in losses)) if losses else 0
            if tl > 0:
                result.profit_factor = tw / tl
        return result
    
    def run_all(self):
        self.compute_masks()
        total = 2 ** len(self.conditions) - 1
        print(f"\n{'='*60}")
        print(f"{self.signal_type.upper()} OPTIMIZATION - {total} COMBINATIONS")
        print(f"{'='*60}")
        start = datetime.now()
        for cid in range(1, total + 1):
            if cid % max(1, total // 10) == 0:
                el = (datetime.now() - start).total_seconds()
                rate = cid / el if el > 0 else 0
                rem = (total - cid) / rate if rate > 0 else 0
                print(f"Progress: {cid}/{total} ({rate:.1f}/s, ~{rem:.0f}s remaining)")
            self.results.append(self.run_combo(cid))
        print(f"\nCompleted in {(datetime.now() - start).total_seconds():.1f}s")
        return pd.DataFrame([{
            'combo_id': r.combo_id, 'binary': r.combo_binary, 'conditions': ', '.join(r.conditions_enabled),
            'num_conditions': r.num_conditions, 'total_return_%': round(r.total_return, 2),
            'cagr_%': round(r.cagr, 2), 'sharpe': round(r.sharpe_ratio, 2), 'max_dd_%': round(r.max_drawdown, 2),
            'volatility_%': round(r.volatility, 2), 'win_rate_%': round(r.win_rate, 1),
            'profit_factor': round(r.profit_factor, 2), 'total_trades': r.total_trades,
            'signals_generated': r.total_signals, 'signals_filled': r.signals_filled,
            'fill_rate_%': round(r.fill_rate, 1), 'stop_loss_exits': r.stop_loss_exits,
            'trailing_exits': r.trailing_exits, 'time_exits': r.time_exits,
            'target_exits': r.target_exits, 'trades_per_year': round(r.trades_per_year, 1),
            'avg_trade_pnl': round(r.avg_trade_pnl, 2),
        } for r in self.results])

# ============================================================
# EXPORT FUNCTION
# ============================================================

def export_to_excel(df, output_path, signal_type, conditions):
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Combinations', index=False)
        df.nlargest(50, 'total_return_%').to_excel(writer, sheet_name='Top_50_Return', index=False)
        if len(df[df['total_trades'] >= 10]) > 0:
            df[df['total_trades'] >= 10].nlargest(50, 'sharpe').to_excel(writer, sheet_name='Top_50_Sharpe', index=False)
        by_num = df.groupby('num_conditions').agg({'total_return_%': 'mean', 'total_trades': 'mean', 'fill_rate_%': 'mean', 'sharpe': 'mean'}).round(2)
        by_num.to_excel(writer, sheet_name='By_Num_Conditions')
        importance = []
        for i, cond in enumerate(conditions):
            mask = df['combo_id'].apply(lambda x: bool(x & (1 << i)))
            on_ret = df[mask]['total_return_%'].mean()
            off_ret = df[~mask]['total_return_%'].mean()
            importance.append({'condition': cond, 'avg_return_ON': round(on_ret, 2), 'avg_return_OFF': round(off_ret, 2), 'impact': round(on_ret - off_ret, 2)})
        pd.DataFrame(importance).sort_values('impact', ascending=False).to_excel(writer, sheet_name='Condition_Importance', index=False)
    print(f"Excel saved: {output_path}")

# ============================================================
# MAIN EXECUTION
# ============================================================

def run_optimization(signal_type, data_path, output_path):
    print("="*70)
    print(f"QTbtx OPTIMIZER - {signal_type.upper()}")
    print("="*70)
    print(f"Data: {data_path}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    optimizer = SignalOptimizer(signal_type)
    optimizer.load_data(data_path)
    results_df = optimizer.run_all()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_file = f'{output_path}/{signal_type}_optimization_{timestamp}.xlsx'
    export_to_excel(results_df, excel_file, signal_type, optimizer.conditions)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total combinations: {len(results_df)}")
    print(f"Positive return: {len(results_df[results_df['total_return_%'] > 0])}")
    print(f"Average fill rate: {results_df['fill_rate_%'].mean():.1f}%")
    
    best = results_df.nlargest(1, 'total_return_%').iloc[0]
    print(f"\nBEST BY RETURN:")
    print(f"  Combo #{best['combo_id']}: {best['total_return_%']:.2f}%")
    print(f"  Conditions: {best['conditions']}")
    
    print(f"\nResults saved to: {excel_file}")
    return results_df

# Run it!
if SIGNAL_TYPE.lower() == 'all':
    all_results = {}
    for sig in SIGNAL_CONDITIONS.keys():
        all_results[sig] = run_optimization(sig, DATA_PATH, OUTPUT_PATH)
else:
    results = run_optimization(SIGNAL_TYPE, DATA_PATH, OUTPUT_PATH)

