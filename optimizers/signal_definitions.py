"""
Signal Definitions - Condition masks for all trading signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List

# All possible conditions across all signals
ALL_CONDITIONS = [
    'cyber_negative',       # CyberCycle < 0
    'value_gt_vindex',      # Swing Value > VIndex
    'rsl_itrend_w_weak',    # RSL iTrend Weekly < 100
    'rsl_itrend_d_weak',    # RSL iTrend Daily < 100
    'rsl_close_weak',       # RSL Close < 100
    'lower_low',            # Low < Low[T-1]
    'lower_close',          # Close < Close[T-1]
    'lower_close_prev',     # Close[T-1] < Close[T-2]
    'below_weekly_low',     # Close < Weekly Low
]

# Signal-specific conditions
SIGNAL_CONDITIONS = {
    'mount': ['cyber_negative', 'value_gt_vindex'],
    'climb': ['cyber_negative', 'value_gt_vindex'],
    'arrow': ['lower_close', 'cyber_negative', 'value_gt_vindex', 'rsl_itrend_w_weak', 'rsl_itrend_d_weak'],
    'collect': ['lower_close', 'cyber_negative', 'value_gt_vindex', 'rsl_itrend_w_weak', 'rsl_itrend_d_weak'],
    'solid': ALL_CONDITIONS,
    'resist': ALL_CONDITIONS,
}

# Trade parameters for each signal
TRADE_PARAMS = {
    'mount': {'exitbarslong': 10, 'stop_loss_pct': 0.05, 'trailing_trigger': 20.0},
    'climb': {'exitbarslong': 10, 'stop_loss_pct': 0.05, 'trailing_trigger': 20.0},
    'arrow': {'exitbarslong': 10, 'stop_loss_pct': 0.05, 'trailing_trigger': 20.0},
    'collect': {'exitbarslong': 10, 'stop_loss_pct': 0.05, 'trailing_trigger': 20.0},
    'solid': {'exitbarslong': 10, 'stop_loss_pct': 0.07, 'trailing_trigger': 12.0},
    'resist': {'exitbarslong': 10, 'stop_loss_pct': 0.05, 'trailing_trigger': 20.0},
}


def get_signal_config(signal_type: str) -> Dict:
    """Get configuration for a specific signal type"""
    signal_type = signal_type.lower()
    if signal_type not in SIGNAL_CONDITIONS:
        raise ValueError(f"Unknown signal type: {signal_type}. Available: {list(SIGNAL_CONDITIONS.keys())}")
    
    conditions = SIGNAL_CONDITIONS[signal_type]
    return {
        'name': signal_type.upper(),
        'conditions': conditions,
        'num_conditions': len(conditions),
        'total_combinations': 2 ** len(conditions) - 1,
        'trade_params': TRADE_PARAMS[signal_type],
    }


def compute_condition_mask(
    ticker_data: pd.DataFrame,
    condition: str,
    config: Dict
) -> pd.Series:
    """
    Compute boolean mask for a single condition
    
    Args:
        ticker_data: DataFrame with OHLCV and indicators
        condition: Condition name
        config: Configuration dict with thresholds
    
    Returns:
        Boolean Series
    """
    rsl_threshold = config.get('rsl_threshold', 100)
    cyber_threshold = config.get('cyber_threshold', 0)
    
    if condition == 'cyber_negative':
        cyber = ticker_data.get('cyber_cycle', pd.Series(np.nan, index=ticker_data.index))
        return (cyber < cyber_threshold).fillna(False)
    
    elif condition == 'value_gt_vindex':
        swing = ticker_data.get('swing_value', pd.Series(np.nan, index=ticker_data.index))
        vindex = ticker_data.get('vindex', pd.Series(np.nan, index=ticker_data.index))
        return (swing > vindex).fillna(False)
    
    elif condition == 'rsl_itrend_w_weak':
        rsl_w = ticker_data.get('rsl_itrend_w', pd.Series(np.nan, index=ticker_data.index))
        return (rsl_w < rsl_threshold).fillna(False)
    
    elif condition == 'rsl_itrend_d_weak':
        rsl_d = ticker_data.get('rsl_itrend_d', pd.Series(np.nan, index=ticker_data.index))
        return (rsl_d < rsl_threshold).fillna(False)
    
    elif condition == 'rsl_close_weak':
        rsl_c = ticker_data.get('rsl_close', pd.Series(np.nan, index=ticker_data.index))
        return (rsl_c < rsl_threshold).fillna(False)
    
    elif condition == 'lower_low':
        low = ticker_data['Low']
        return (low < low.shift(1)).fillna(False)
    
    elif condition == 'lower_close':
        close = ticker_data['Close']
        return (close < close.shift(1)).fillna(False)
    
    elif condition == 'lower_close_prev':
        close = ticker_data['Close']
        return (close.shift(1) < close.shift(2)).fillna(False)
    
    elif condition == 'below_weekly_low':
        close = ticker_data['Close']
        weekly_low = ticker_data.get('weekly_low', pd.Series(np.nan, index=ticker_data.index))
        return (close < weekly_low).fillna(False)
    
    else:
        raise ValueError(f"Unknown condition: {condition}")


def get_enabled_conditions(combo_id: int, conditions: List[str]) -> List[str]:
    """Get list of enabled conditions for a combo ID"""
    return [conditions[i] for i in range(len(conditions)) if combo_id & (1 << i)]

