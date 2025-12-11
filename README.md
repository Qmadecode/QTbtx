# QTbtx - Quantitative Trading Signal Optimizer

A comprehensive feature optimization framework for testing all combinations of trading signal conditions across multiple strategies.

## Supported Signals

| Signal | Conditions | Combinations |
|--------|------------|--------------|
| MOUNT | 2 | 3 |
| CLIMB | 2 | 3 |
| ARROW | 5 | 31 |
| COLLECT | 5 | 31 |
| SOLID | 9 | 511 |
| RESIST | 9 | 511 |

## Quick Start

### Google Colab (Recommended)

1. Upload your preprocessed data to Google Drive
2. Open `colab/run_optimizer.ipynb` in Colab
3. Set your `DATA_PATH` and `SIGNAL_TYPE`
4. Run all cells

### Google Cloud

```bash
# Clone the repo
git clone https://github.com/Qmadecode/QTbtx.git
cd QTbtx

# Deploy to Google Cloud
./cloud/deploy_gcp.sh
```

### Local

```bash
pip install -r requirements.txt
python run_optimizer.py --signal resist --data /path/to/data
```

## Project Structure

```
QTbtx/
├── README.md
├── requirements.txt
├── run_optimizer.py           # Main entry point
├── config/
│   └── signals.yaml           # Signal configurations
├── optimizers/
│   ├── __init__.py
│   ├── base_optimizer.py      # Core optimization logic
│   ├── signal_definitions.py  # Signal conditions
│   └── trade_logic.py         # Trade/exit logic
├── colab/
│   └── run_optimizer.ipynb    # Colab notebook
└── cloud/
    ├── Dockerfile
    ├── deploy_gcp.sh
    └── cloud_run.py
```

## Signal Conditions

### MOUNT / CLIMB (2 conditions)
- `cyber_negative`: CyberCycle < 0
- `value_gt_vindex`: Swing Value > VIndex

### ARROW / COLLECT (5 conditions)
- `lower_close`: Close < Close[T-1]
- `cyber_negative`: CyberCycle < 0
- `value_gt_vindex`: Swing Value > VIndex
- `rsl_itrend_w_weak`: RSL iTrend Weekly < 100
- `rsl_itrend_d_weak`: RSL iTrend Daily < 100

### SOLID / RESIST (9 conditions)
- `cyber_negative`: CyberCycle < 0
- `value_gt_vindex`: Swing Value > VIndex
- `rsl_itrend_w_weak`: RSL iTrend Weekly < 100
- `rsl_itrend_d_weak`: RSL iTrend Daily < 100
- `rsl_close_weak`: RSL Close < 100
- `lower_low`: Low < Low[T-1]
- `lower_close`: Close < Close[T-1]
- `lower_close_prev`: Close[T-1] < Close[T-2]
- `below_weekly_low`: Close < Weekly Low

## Output

Results are saved as Excel files with multiple sheets:
- `All_Combinations`: Complete results for all tested combinations
- `Top_50_Return`: Best performers by total return
- `Top_50_Sharpe`: Best risk-adjusted performers
- `By_Num_Conditions`: Average performance by condition count
- `Condition_Importance`: Impact of each individual condition

## License

MIT

