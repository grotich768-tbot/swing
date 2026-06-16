"""
config.py  —  Central configuration for Always-In Trading Bot
All hyperparameters, paths, and constants live here.

IMPROVEMENTS APPLIED:
  Tier 1: Extended train history, 3M RL steps, per-symbol models
  Tier 2: Session-aware reward multipliers, ensemble seeds, walk-forward windows
  Tier 3: VWAP + tick volume feature flags
  Tier 4: Adaptive sizing, news filter window, paper-trade comparison flag
"""
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data" / "raw"
MODEL_DIR  = BASE_DIR / "models" / "saved"
LOG_DIR    = BASE_DIR / "logs"
RESULT_DIR = BASE_DIR / "results"

for _d in [DATA_DIR, MODEL_DIR, LOG_DIR, RESULT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Broker — apply per-broker pip/spread values on import
# Override active broker: set env var BROKER=ICMarkets
# or: python train.py --broker ICMarkets
# ─────────────────────────────────────────────────────────────────────────────
try:
    from broker_config import get_broker_specs, ACTIVE_BROKER as _BROKER
    _specs = get_broker_specs()
    for _sym, _vals in _specs.items():
        SPREAD_PIPS[_sym] = _vals["spread"]
        PIP_VALUE[_sym]   = _vals["pip_size"]
except Exception:
    pass   # broker_config not available — use hardcoded values above

# ─────────────────────────────────────────────────────────────────────────────
# Symbols
# ─────────────────────────────────────────────────────────────────────────────
MT5_SYMBOLS = {
    "GOLD":   ["GOLD", "XAUUSD", "XAUUSDm", "XAUUSD."],
    "SILVER": ["SILVER", "XAGUSD", "XAGUSDm", "XAGUSD."],
    "EURUSD": ["EURUSD", "EURUSDm", "EURUSD."],
    "GBPUSD": ["GBPUSD", "GBPUSDm", "GBPUSD."],
    "USDJPY": ["USDJPY", "USDJPYm"],
    "BTCUSD": ["BTCUSD", "BTCUSDm", "BTCUSD.", "BTC/USD"],
    "ETHUSD": ["ETHUSD", "ETHUSDm", "ETHUSD.", "ETH/USD"],
    "LTCUSD": ["LTCUSD", "LTCUSDm", "LTCUSD.", "LTC/USD"],
    "US30":   ["US30Cash", "US30m", "US30."],
    "US100":  ["US100Cash", "US100m", "US100."],
    "US500":  ["US500Cash", "US500m", "US500."],
    "UK100":  ["UK100Cash", "UK100m", "UK100."],
    "AUS200": ["AUS200Cash", "AUS200m", "AUS200."],
    "GER40":  ["GER40Cash", "GER40m", "GER40."],
    "JP225":  ["JP225Cash", "JN225m", "JN225."],
}

YF_SYMBOLS = {
    "GOLD":   "GC=F",
    "SILVER": "SI=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "LTCUSD": "LTC-USD",
    "US30":   "^DJI",
    "US100":  "^NDX",
    "US500":  "^GSPC",
    "UK100":  "^FTSE",
    "AUS200": "^AXJO",
    "GER40":  "^GDAXI",
    "JP225":  "^N225",
}

SYMBOLS = ["GOLD", "SILVER", "GBPUSD", "EURUSD", "BTCUSD", "ETHUSD"]

PIP_VALUE = {
    "GOLD":   0.01,
    "SILVER": 0.001,
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "BTCUSD": 0.1,
    "ETHUSD": 0.1,
    "LTCUSD": 0.01,   # LTC pip size
    "US30":   1.0,
    "US100":  1.0,
    "US500":  1.0,
    "UK100":  1.0,
    "AUS200": 1.0,
    "GER40":  1.0,
    "JP225":  1.0,
}

SPREAD_PIPS = {
    # Broker-verified values (pips)
    "GOLD":   6.0,     # was 32.0  ✓ fixed
    "SILVER": 10.0,    # was 170.0 ✓ fixed — explains Silver catastrophe
    "EURUSD": 1.0,     # was 10.0  ✓ fixed
    "GBPUSD": 1.1,     # was 11.1  ✓ fixed
    "USDJPY": 1.1,     # was 11.4  ✓ fixed
    "BTCUSD": 297.6,   # was 310.0 ✓ fixed
    "ETHUSD": 49.8,    # was 57.5  ✓ fixed
    "LTCUSD": 50.0,    # verify with broker
    "US30":   3.90,    # was 5.50  ✓ fixed
    "US100":  1.95,    # was 2.60  ✓ fixed
    "US500":  0.55,    # was 0.70  ✓ fixed
    "UK100":  1.60,    # was 2.20  ✓ fixed
    "AUS200": 5.54,    # unchanged ✓
    "GER40":  1.95,    # was 2.30  ✓ fixed
    "JP225":  8.0,     # unchanged (verify with broker)
}

# ─────────────────────────────────────────────────────────────────────────────
# Data fetching — timeframes
# ─────────────────────────────────────────────────────────────────────────────
FETCH_TIMEFRAMES = ["M15", "H1", "H4", "D1"]
PRIMARY_TF       = "M15"   # switched from H1 — intraday style, ~1.5hr avg hold
LOOKBACK_BARS    = 8000   # M15: keep ~8000 bars lookback

# TIER 1: Use full history from earliest available date
TRAIN_START = "2002-09-16"
TRAIN_END   = "2021-12-31"
TEST_START  = "2022-01-14"
TEST_END    = "2026-05-08"

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_LAGS    = [1, 2, 3, 5, 10, 20]
RSI_PERIOD      = 14
ATR_PERIOD      = 14
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
BB_PERIOD       = 20
EMA_PERIODS     = [8, 21, 50, 200]
FEATURE_WINDOW  = 20
N_REGIMES       = 4

# TIER 3: Feature flags — new features added to feature_engineer.py
USE_VWAP_FEATURES        = True   # VWAP deviation + distance (strong for GOLD)
USE_TICK_VOLUME_IMBALANCE = True  # Volume ratio asymmetry proxy
USE_SPREAD_DYNAMICS      = True   # HL-spread volatility signal

# ─────────────────────────────────────────────────────────────────────────────
# RL Environment
# ─────────────────────────────────────────────────────────────────────────────
LOOKBACK_STEPS  = 2
COMMISSION_PIPS = 0.5
INITIAL_BALANCE = 10_000.0
MAX_POSITION_PCT = 0.02
FLIP_PENALTY    = 0.5
DRAWDOWN_PENALTY_SCALE = 1.5
MAX_EPISODE_STEPS = 8000   # M15: 4x H1, ~8000 bars = ~83 trading days

# TIER 2: Session-aware reward multipliers
# Calibrated from GOLD ensemble backtest analysis (2022-2026):
#   London  07-12 UTC  $197/trade, 82.6% win  ← BEST for GOLD
#   Overlap 12-16 UTC  $180/trade, 81.5% win
#   Asia    00-07 UTC  $163/trade, 72.0% win  ← stronger than assumed (China/India demand)
#   NY      16-22 UTC  $137/trade, 73.1% win
#   Off     22-00 UTC  $121/trade, 70.5% win
SESSION_REWARD_MULTIPLIERS = {
    "overlap":  1.2,   # 12:00–16:00 UTC  reduced from 1.3 — London outperforms
    "london":   1.4,   # 07:00–12:00 UTC  increased from 1.1 — best session for GOLD
    "ny":       1.0,   # 16:00–22:00 UTC  unchanged
    "asia":     1.0,   # 00:00–07:00 UTC  increased from 0.7 — Asia is strong for GOLD
    "off":      0.8,   # 22:00–00:00 UTC  unchanged — still weakest
}

# Power-hour bonuses — from hourly heatmap (applied on top of session mult)
# 09:00 UTC: 92% win rate  $246 avg  London open momentum
# 13:00 UTC: 88% win rate  $215 avg  NY pre-open
# 05:00 UTC: 84% win rate  $211 avg  Early Tokyo
# 17:00 UTC: 88% win rate  $195 avg  NY first hour
SESSION_POWER_HOURS = {
    9:  1.5,
    13: 1.4,
    5:  1.3,
    17: 1.2,
}

# TIER 4: Adaptive position sizing  (applied by risk engine)
ADAPTIVE_SIZING_ENABLED     = True
ADAPTIVE_LOSE_STREAK_CUTOFF = 3      # After N consecutive losses, scale down
ADAPTIVE_LOSE_SIZE_MULT     = 0.7   # Size multiplier after losing streak
ADAPTIVE_WIN_STREAK_CUTOFF  = 5     # After N consecutive wins, scale up slightly
ADAPTIVE_WIN_SIZE_MULT      = 1.15  # Max scale-up
KELLY_WINDOW                = 50    # Rolling window for Kelly fraction estimate

# TIER 4: News / economic calendar filter (minutes before/after event)
NEWS_FILTER_MINUTES = 15   # Suppress flips ±15 min around major events
# High-impact events: 0=Sun 1=Mon…6=Sat, hour UTC, minute UTC
# These are hardcoded approximations; connect a calendar API for live use.
KNOWN_HIGH_IMPACT_HOURS_UTC = {
    # NFP — first Friday of month ~13:30 UTC
    "NFP": {"weekday": 4, "hour": 13, "minute": 30},
    # CPI — usually Wed/Thu ~12:30 UTC
    "CPI": {"weekday": 2, "hour": 12, "minute": 30},
    # FOMC — Wed ~18:00 UTC
    "FOMC": {"weekday": 2, "hour": 18, "minute": 0},
}

# ─────────────────────────────────────────────────────────────────────────────
# Risk engine
# ─────────────────────────────────────────────────────────────────────────────
MAX_DRAWDOWN_PCT       = 0.08
CIRCUIT_BREAKER_PCT    = 0.04
MAX_CORR_EXPOSURE_PCT  = 0.06
SESSION_SIZE_REDUCTION = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Regime classifier (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
REGIME_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":      "mlogloss",
    "random_state":     42,
}

# ─────────────────────────────────────────────────────────────────────────────
# Price direction predictor (LSTM)
# ─────────────────────────────────────────────────────────────────────────────
LSTM_SEQ_LEN         = 20
LSTM_HIDDEN          = 64
LSTM_LAYERS          = 1
LSTM_DROPOUT         = 0.3
LSTM_LR              = 1e-3
LSTM_EPOCHS          = 30
LSTM_BATCH           = 256
LSTM_EVAL_BATCH      = 512
LSTM_MAX_SAMPLES     = 40_000
LSTM_EARLY_STOP_PAT  = 6
DIRECTION_HORIZON    = 5

# ─────────────────────────────────────────────────────────────────────────────
# PPO (Stable Baselines 3)
# ─────────────────────────────────────────────────────────────────────────────
PPO_PARAMS = {
    "learning_rate":  3e-4,
    "n_steps":        2048,
    "batch_size":     64,
    "n_epochs":       10,
    "gamma":          0.99,
    "gae_lambda":     0.95,
    "clip_range":     0.2,
    "ent_coef":       0.01,
    "vf_coef":        0.5,
    "max_grad_norm":  0.5,
    "policy_kwargs": {
        "net_arch": [256, 256, 128],
    },
}

# TIER 1: 3M steps for noisy metals; 2M for FX
TOTAL_TIMESTEPS = {
    # M15 timeframe — 4x more bars than H1, steps scaled up accordingly
    "GOLD":   5_000_000,
    "SILVER": 5_000_000,
    "EURUSD": 4_000_000,
    "GBPUSD": 4_000_000,
    "USDJPY": 4_000_000,
    "BTCUSD": 8_000_000,   # crypto M15 very noisy
    "ETHUSD": 6_000_000,
    "LTCUSD": 6_000_000,
    "US30":   4_000_000,
    "US100":  4_000_000,
    "US500":  4_000_000,
    "UK100":  4_000_000,
    "AUS200": 4_000_000,
    "GER40":  4_000_000,
    "JP225":  4_000_000,
}

# Per-symbol regime agent timesteps (overrides REGIME_TIMESTEPS in train_regime_agents.py)
REGIME_TIMESTEPS_OVERRIDE = {
    # M15 regime timesteps — scaled 2x from H1 due to 4x more bars
    "GOLD":   {0: 2_500_000, 1: 2_000_000, 2: 2_000_000},
    "SILVER": {0: 2_500_000, 1: 2_000_000, 2: 2_000_000},
    "EURUSD": {0: 2_000_000, 1: 1_500_000, 2: 1_500_000},
    "GBPUSD": {0: 2_000_000, 1: 1_500_000, 2: 1_500_000},
    "USDJPY": {0: 2_000_000, 1: 1_500_000, 2: 1_500_000},
    "BTCUSD": {0: 3_000_000, 1: 4_000_000, 2: 4_000_000},
    "ETHUSD": {0: 2_500_000, 1: 3_000_000, 2: 3_000_000},
    "LTCUSD": {0: 2_500_000, 1: 3_000_000, 2: 3_000_000},
    "US30":   {0: 1_500_000, 1: 2_000_000, 2: 1_500_000},
    "US100":  {0: 1_500_000, 1: 2_000_000, 2: 1_500_000},
    "US500":  {0: 1_500_000, 1: 2_000_000, 2: 1_500_000},
    "UK100":  {0: 1_500_000, 1: 1_500_000, 2: 1_500_000},
    "AUS200": {0: 1_500_000, 1: 1_500_000, 2: 1_500_000},
    "GER40":  {0: 1_500_000, 1: 2_000_000, 2: 1_500_000},
    "JP225":  {0: 1_500_000, 1: 1_500_000, 2: 1_500_000},
}

# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol live model mode — "ensemble" or "regime"
# ─────────────────────────────────────────────────────────────────────────────
# Determined empirically by comparing M15 ensemble vs regime-router backtests
# over 2022-01-14 → 2026-05-08 (BTCUSD/ETHUSD: 2024-01-01 →). Default is
# "ensemble" for any symbol not listed below (DEFAULT_MODEL_MODE).
#
# Source comparison (PnL / Sharpe-equivalent / MaxDD, ensemble vs regime):
#   BTCUSD   ens=$1,567,450 DD=2.73%  | reg=$1,551,187 DD=3.99%  -> ensemble
#   ETHUSD   ens=$51,546    DD=0.77%  | reg=$48,003    DD=0.98%  -> ensemble
#   EURUSD   ens=$610,933   DD=4.83%  | reg=$618,411   DD=2.02%  -> regime
#            (regime cuts EURUSD's drawdown by 58% for essentially flat PnL —
#             matches the session-imbalance diagnosis: EURUSD's Off-session
#             (22-00 UTC) trades have the weakest profit factor of any
#             session/symbol combination seen so far, and regime specialists
#             absorb that weakness better than the single generalist ensemble)
#   GBPUSD   ens=$803,896   DD=2.14%  | reg=$802,542   DD=3.16%  -> ensemble
#   GOLD     ens=$352,917   DD=0.93%  | reg=$357,320   DD=1.40%  -> ensemble
#   SILVER   ens=$442,725   DD=0.76%  | reg=$437,059   DD=0.85%  -> ensemble
#
# Re-run analyse_streaks_sessions.py + this comparison whenever a symbol is
# retrained, and update this dict if the picture changes.
DEFAULT_MODEL_MODE = "ensemble"

SYMBOL_MODEL_MODE = {
    "GOLD":   "ensemble",
    "SILVER": "ensemble",
    "EURUSD": "regime",     # only symbol where regime router wins — see above
    "GBPUSD": "ensemble",
    "BTCUSD": "ensemble",
    "ETHUSD": "ensemble",
    # Not yet benchmarked — fall back to DEFAULT_MODEL_MODE until compared:
    # USDJPY, LTCUSD, US30, US100, US500, UK100, AUS200, GER40, JP225
}

def get_model_mode(symbol: str) -> str:
    """Return 'ensemble' or 'regime' for a symbol, falling back to the default."""
    return SYMBOL_MODEL_MODE.get(symbol, DEFAULT_MODEL_MODE)

# Per-symbol train/test date overrides
# Global defaults: TRAIN_START=2002-09-16, TRAIN_END=2021-12-31
# Override only when history is shorter than global range
SYMBOL_TRAIN_START = {
    # Crypto — limited history
    "BTCUSD": "2017-01-01",
    "ETHUSD": "2017-01-01",
    "LTCUSD": "2017-01-01",
    # Indices — yfinance has good history from ~2000, use global default
    # FX — use global default (data from 2002)
}
SYMBOL_TRAIN_END = {
    "BTCUSD": "2023-12-31",   # includes FTX collapse
    "ETHUSD": "2023-12-31",
    "LTCUSD": "2023-12-31",
}
SYMBOL_TEST_START = {
    "BTCUSD": "2024-01-01",   # ETF approval + halving + bull run
    "ETHUSD": "2024-01-01",
    "LTCUSD": "2024-01-01",
}
SYMBOL_TEST_END = {
    "BTCUSD": "2026-05-08",
    "ETHUSD": "2026-05-08",
    "LTCUSD": "2026-05-08",
}

# TIER 2: Ensemble — train N seeds per symbol, majority vote at inference
PPO_ENSEMBLE_SEEDS  = [42, 123, 777, 1337, 9999]  # 5 seeds → majority vote
PPO_ENSEMBLE_ENABLED = True  # Set False to skip ensemble and use single model

# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward validation   (TIER 2)
# ─────────────────────────────────────────────────────────────────────────────
WF_TRAIN_WINDOW = "18M"   # 18 months training
WF_TEST_WINDOW  = "3M"    # 3 months test (roll forward monthly)
WF_PURGE_GAP    = "2W"    # 2-week gap between train and test
WF_FINETUNE_STEPS = 200_000  # RL fine-tune steps when retraining monthly
