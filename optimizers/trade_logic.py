"""
Trade Logic - Entry and exit calculations
"""

import pandas as pd
import numpy as np
from typing import Tuple, Any, Optional


def calculate_true_range(data: pd.DataFrame) -> pd.Series:
    """Calculate True Range"""
    high = data['High']
    low = data['Low']
    close_prev = data['Close'].shift(1)
    tr = pd.Series(
        np.maximum.reduce([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ]),
        index=data.index
    )
    return tr


def calculate_atr(data: pd.DataFrame, period: int = 10) -> pd.Series:
    """Calculate Average True Range"""
    tr = calculate_true_range(data)
    return tr.rolling(window=period).mean()


def calculate_entry_limit(data: pd.DataFrame, signal_bar_idx: int = -2) -> Optional[float]:
    """
    Calculate entry limit price using T-1 data
    
    Entry Limit = Low[T-1] - TR[T-2] / 4
    """
    if len(data) < 3:
        return None
    
    if 'entry_limit' in data.columns:
        entry_limit = data['entry_limit'].iloc[signal_bar_idx]
        if not pd.isna(entry_limit):
            return round(entry_limit, 2)
    
    low_t_minus_1 = data['Low'].iloc[signal_bar_idx]
    
    if 'tr' in data.columns:
        tr_t_minus_2 = data['tr'].iloc[signal_bar_idx - 1]
    else:
        tr = calculate_true_range(data)
        tr_t_minus_2 = tr.iloc[signal_bar_idx - 1]
    
    if pd.isna(low_t_minus_1) or pd.isna(tr_t_minus_2):
        return None
    
    return round(low_t_minus_1 - (tr_t_minus_2 / 4), 2)


def check_exit_conditions(
    position: dict,
    current_data: pd.DataFrame,
    trade_params: dict
) -> Tuple[bool, str, Any]:
    """
    Check all exit conditions in priority order
    
    Priority:
    1. Hard stop loss
    2. Trailing stop
    3. Time-based exit
    4. Profit target exit
    """
    if len(current_data) < 3:
        return False, "", 0.0
    
    close_t_minus_1 = current_data['Close'].iloc[-2]
    entry_price = position['entry_price']
    runup = position['runup']
    entry_date = position['entry_date']
    
    if pd.isna(close_t_minus_1):
        return False, "", 0.0
    
    bars_held = (current_data.index[-2] - entry_date).days
    stop_level = entry_price * (1 - trade_params['stop_loss_pct'])
    
    # Priority 1: Hard stop loss
    if close_t_minus_1 < stop_level:
        return True, "Stop Loss", "NEXT_OPEN"
    
    # Priority 2: Trailing stop
    if runup > trade_params['trailing_trigger']:
        historical_data = current_data.iloc[:-1]
        high_since_entry = historical_data.loc[historical_data.index >= entry_date, 'High'].max()
        
        if runup >= 30.0:
            trailing_stop_pct = 0.20
        elif runup >= 25.0:
            trailing_stop_pct = 0.15
        elif runup >= 20.0:
            trailing_stop_pct = 0.10
        else:
            trailing_stop_pct = (high_since_entry - entry_price) / high_since_entry
        
        trailing_stop_level = high_since_entry * (1 - trailing_stop_pct)
        
        if close_t_minus_1 < trailing_stop_level:
            return True, "Trailing Stop", "NEXT_OPEN"
    
    # Priority 3: Time-based exit
    if bars_held > trade_params['exitbarslong']:
        historical_data = current_data.iloc[:-1]
        time_exit_level = None
        
        if 'time_exit' in historical_data.columns:
            time_exit_level = historical_data['time_exit'].iloc[-1]
        
        if time_exit_level is None or pd.isna(time_exit_level):
            if len(historical_data) >= 5:
                swing_value = historical_data.get('swing_value', pd.Series()).iloc[-1] if 'swing_value' in historical_data.columns else None
                if swing_value is not None and not pd.isna(swing_value):
                    if 'atr_5' in historical_data.columns:
                        atr5 = historical_data['atr_5'].iloc[-1]
                    else:
                        atr5 = calculate_atr(historical_data, 5).iloc[-1]
                    if not pd.isna(atr5):
                        time_exit_level = round(swing_value + atr5, 2)
        
        if time_exit_level is not None and not pd.isna(time_exit_level):
            return True, "Time Exit", round(time_exit_level, 2)
        return True, "Time Exit", round(close_t_minus_1, 2)
    
    # Priority 4: Profit target exit
    historical_data = current_data.iloc[:-1]
    target_price = None
    
    if 'target_exit' in historical_data.columns:
        target_price = historical_data['target_exit'].iloc[-1]
    
    if target_price is None or pd.isna(target_price):
        if len(historical_data) >= 10:
            swing_value = historical_data.get('swing_value', pd.Series()).iloc[-1] if 'swing_value' in historical_data.columns else None
            if swing_value is not None and not pd.isna(swing_value):
                if 'atr_10' in historical_data.columns:
                    atr10 = historical_data['atr_10'].iloc[-1]
                else:
                    atr10 = calculate_atr(historical_data, 10).iloc[-1]
                if not pd.isna(atr10):
                    target_price = swing_value + atr10
    
    if target_price is not None and not pd.isna(target_price):
        if close_t_minus_1 >= target_price:
            limit_price = historical_data['High'].iloc[-1]
            return True, "Target Exit", round(limit_price, 2)
    
    return False, "", 0.0

