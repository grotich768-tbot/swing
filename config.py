"""
config.py  —  Central configuration for Always-In Trading Bot
All hyperparameters, paths, and constants live here.
"""
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data" / "raw"
MODEL_DIR  = BASE_DIR / "models" / "saved"
LOG_DIR    = BASE_DIR / "logs"
RESULT_DIR = BASE_DIR / "results"

for _d in [DATA_DIR, MODEL_DIR, LOG_DIR, RESULT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Symbols
# ─────────────────────────────────────────────
# MT5 symbol names (adjust to your broker's naming)
# The fetcher will try each candidate in order and use the first one found.
MT5_SYMBOLS = {
    "GOLD":   ["GOLD", "XAUUSD", "XAUUSDm", "XAUUSD."],
    "SILVER": ["SILVER", "XAGUSD", "XAGUSDm", "XAGUSD."],
    "EURUSD": ["EURUSD", "EURUSDm", "EURUSD."],
    "GBPUSD": ["GBPUSD", "GBPUSDm", "GBPUSD."],
    "USDJPY": ["USDJPY","USDJPYm"],
    "US30": ["US30Cash", "US30m", "US30."],
    "US100": ["US100Cash", "US100m", "US100."],
    "US500": ["US500Cash", "US500m", "US500."],
    "UK100": ["UK100Cash", "UK100m", "UK100."],
    "AUS200": ["AUS200Cash", "AUS200m", "AUS200."],
    "GER40": ["GER40Cash", "GER40m", "GER40."],
    "JP225": ["JP225Cash", "JN225m", "JN225."],
}

# yfinance fallback symbols (used on Codespaces/Linux)
YF_SYMBOLS = {
    "GOLD":   "GC=F",
    "SILVER": "SI=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "US30":   "^DJI",
    "US100":  "^NDX",
    "US500":  "^GSPC",
    "UK100":  "^FTSE",
    "AUS200": "^AXJO",
    "GER40":  "^GDAXI",
    "JP225":  "^N225",
}

SYMBOLS = list(MT5_SYMBOLS.keys())   # ["GOLD", "SILVER", "EURUSD", "GBPUSD"]

# Pip / point values for each symbol (used in PnL calculation)
PIP_VALUE = {
    "GOLD":   0.01,
    "SILVER": 0.001,
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "BTCUSD": 0.1,
    "ETHUSD": 0.1,
    "US30":   1.0,
    "US100":  1.0,
    "US500":  1.0,
    "UK100":  1.0,
    "AUS200": 1.0,
    "GER40":  1.0,
    "JP225":  1.0,
}

# Spread estimates in pips (used in reward shaping)
SPREAD_PIPS = {
    "GOLD":   32.0,
    "SILVER": 170.0,
    "EURUSD": 10.0,
    "GBPUSD": 11.1,
    "USDJPY": 11.4,
    "BTCUSD": 310.0,
    "ETHUSD": 57.5,
    "US30":   5.50,
    "US100":  2.60,
    "US500":  0.70,
    "UK100":  2.20,
    "AUS200": 5.54,
    "GER40":  2.30,
    "JP225":  8.0,
}

# ─────────────────────────────────────────────
# Data fetching — timeframes
# ─────────────────────────────────────────────
# Only these four are fetched and used in training.
# M15  →  entry refinement features  (intrabar momentum)
# H1   →  primary RL action timeframe
# H4   →  setup / structure context
# D1   →  daily bias context
FETCH_TIMEFRAMES = ["M15", "H1", "H4", "D1"]

PRIMARY_TF      = "H1"
LOOKBACK_BARS   = 2000
TRAIN_START = "2002-09-16"
TRAIN_END   = "2021-12-31"
TEST_START  = "2022-01-14"
TEST_END    = "2026-05-08"

# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────
FEATURE_LAGS    = [1, 2, 3, 5, 10, 20]   # Return lags
RSI_PERIOD      = 14
ATR_PERIOD      = 14
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
BB_PERIOD       = 20
EMA_PERIODS     = [8, 21, 50, 200]
FEATURE_WINDOW  = 20          # Rolling z-score normalisation window
N_REGIMES       = 4           # Number of market regimes

# ─────────────────────────────────────────────
# RL Environment
# ─────────────────────────────────────────────
LOOKBACK_STEPS  = 2           # How many past steps the agent sees
COMMISSION_PIPS = 0.5         # Commission per trade in pips (round-trip)
INITIAL_BALANCE = 10_000.0    # USD
MAX_POSITION_PCT = 0.02       # 2% account risk per trade
FLIP_PENALTY    = 0.5         # Extra pip-equivalent penalty per flip (discourages churn)
DRAWDOWN_PENALTY_SCALE = 1.5  # Multiplier on drawdown reward penalty
MAX_EPISODE_STEPS = 2000      # Max steps per training episode

# ─────────────────────────────────────────────
# Risk engine
# ─────────────────────────────────────────────
MAX_DRAWDOWN_PCT       = 0.08   # 8% max drawdown before circuit breaker
CIRCUIT_BREAKER_PCT    = 0.04   # Resume at 4% recovery
MAX_CORR_EXPOSURE_PCT  = 0.06   # Max combined exposure for correlated assets
SESSION_SIZE_REDUCTION = 0.5    # Size multiplier during illiquid sessions

# ─────────────────────────────────────────────
# Regime classifier (XGBoost)
# ─────────────────────────────────────────────
REGIME_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "random_state": 42,
}

# ─────────────────────────────────────────────
# Price direction predictor (LSTM)
# ─────────────────────────────────────────────
LSTM_SEQ_LEN    = 20       # Sequence length (shorter = faster, less RAM)
LSTM_HIDDEN     = 64       # Hidden units   (64 sufficient on CPU, was 128)
LSTM_LAYERS     = 1        # Single layer — avoids inter-layer dropout OOM on CPU
LSTM_DROPOUT    = 0.3
LSTM_LR         = 1e-3
LSTM_EPOCHS     = 30               # Max epochs — early stopping cuts this short
LSTM_BATCH      = 256              # Larger batch = faster on CPU
LSTM_EVAL_BATCH = 512              # Batched eval to avoid OOM
LSTM_MAX_SAMPLES = 40_000          # Cap sequences; randomly subsampled if exceeded
LSTM_EARLY_STOP_PAT = 6            # Stop if val loss stagnates for N epochs
DIRECTION_HORIZON = 5      # Predict direction N bars ahead

# ─────────────────────────────────────────────
# PPO (Stable Baselines 3)
# ─────────────────────────────────────────────
PPO_PARAMS = {
    "learning_rate":    3e-4,
    "n_steps":          2048,
    "batch_size":       64,
    "n_epochs":         10,
    "gamma":            0.99,
    "gae_lambda":       0.95,
    "clip_range":       0.2,
    "ent_coef":         0.01,
    "vf_coef":          0.5,
    "max_grad_norm":    0.5,
    "policy_kwargs": {
        "net_arch": [256, 256, 128],
    },
}
TOTAL_TIMESTEPS = 1_000_000   # Total RL training steps per symbol

# ─────────────────────────────────────────────
# Walk-forward validation
# ─────────────────────────────────────────────
WF_TRAIN_WINDOW = "18M"   # 18 months training
WF_TEST_WINDOW  = "3M"    # 3 months test
WF_PURGE_GAP    = "2W"    # 2 week gap between train and test
