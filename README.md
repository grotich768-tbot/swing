# Always-In RL/ML Trading Bot — Multi-Asset (V3)

An always-in position trading bot using Reinforcement Learning (PPO) +
supervised ML (XGBoost regime classifier + LSTM direction predictor)
for **GOLD, SILVER, EURUSD, GBPUSD, BTCUSD, and ETHUSD**.

The agent is **never flat** — it holds Long or Short at all times,
and flips when the policy decides.

---

## Architecture

```
MT5 Terminal  (open on Windows)
      │
      │  python -m data.mt5_fetcher          ← no arguments needed
      │  auto-detects terminal + broker symbols
      │  saves  data/raw/GOLD_M15.parquet
      │         data/raw/SILVER_M15.parquet  (+ other timeframes)
      ↓
Feature Engineering
  returns · RSI · ATR · MACD · Bollinger · EMA stack
  volume ratio · session sin/cos · day-of-week sin/cos
      │
      ├─► Regime Classifier  (XGBoost)
      │     4 regimes: range-low-vol / trend-up / trend-down / breakout
      │
      └─► Direction Predictor  (2-layer LSTM)
            P(up move in next 5 bars)
                    │
                    ▼
         RL Environment  (Gymnasium  AlwaysInEnv)
           State  : features + ML outputs + position + PnL context
           Action : {HOLD, FLIP}
           Reward : ATR-norm PnL − costs − drawdown penalty + Sharpe bonus
                    │
                    ▼
            PPO Agent  (Stable-Baselines3)
              MLP 256 → 256 → 128  |  Actor + Critic heads
                    │
                    ▼
         Risk Engine
           circuit breaker · GOLD+SILVER correlation cap
           session sizing  · metals rollover guard
```

---

## Quickstart

### On Windows (with MT5 open)

```bash
# 1  Install dependencies
pip install -r requirements.txt

# 2  Fetch data — just open MT5, log in, then run:
python -m data.mt5_fetcher

# 3  Train everything
python train.py
```

### On GitHub Codespaces / Linux (yfinance fallback)

```bash
# 1  Open repo in Codespaces  (devcontainer auto-installs requirements)
# 2  Run — yfinance fetches GC=F and SI=F automatically
python train.py

# Quick smoke test (fewer steps)
python train.py --timesteps 50000
```

---

## MT5 Data Fetcher

### Basic usage — no arguments

```bash
python -m data.mt5_fetcher
```

Opens whichever MT5 terminal is already running, auto-detects the broker's
symbol name for GOLD and SILVER, and saves Parquet files to `data/raw/`.

**Symbol candidates tried (in order):**

| Logical | Tries |
|---------|-------|
| GOLD    | `GOLD` → `XAUUSD` → `XAUUSDm` → `XAUUSD.` |
| SILVER  | `SILVER` → `XAGUSD` → `XAGUSDm` → `XAGUSD.` |

If your broker uses a different name, add it to `MT5_SYMBOLS` in `config.py`.

### Options

```bash
# Custom date range
python -m data.mt5_fetcher --from 2020-01-01 --to 2024-12-31

# Single symbol / timeframe
python -m data.mt5_fetcher --symbol GOLD --tf M15

# Specific MT5 installation path
python -m data.mt5_fetcher --path "C:/Program Files/MetaTrader 5/terminal64.exe"

# Explicit login (only needed if MT5 is NOT already logged in)
python -m data.mt5_fetcher --login 123456 --password yourpass --server Broker-Live
```

### Output files

```
data/raw/
  GOLD_M1.parquet   GOLD_M5.parquet   GOLD_M15.parquet
  GOLD_H1.parquet   GOLD_H4.parquet   GOLD_D1.parquet
  SILVER_M1.parquet ...
```

Check what's been downloaded:
```bash
python -m data.data_loader    # prints a status table
```

---

## Training pipeline

```bash
# Full pipeline
python train.py

# Skip supervised training if already done
python train.py --skip-supervised

# Custom timesteps
python train.py --timesteps 500000

# Single symbol
python train.py --symbols GOLD

# Custom date split
python train.py --from 2019-01-01 --to 2023-06-30 \
                --test-from 2023-07-01 --test-to 2024-01-01
```

### Steps executed by `train.py`

| Step | What happens |
|------|-------------|
| 0    | Optional MT5 fetch (`--fetch-mt5` flag) |
| 1    | Train XGBoost regime classifier + LSTM direction model |
| 2    | Train PPO agent per symbol (1 M steps default) |
| 3    | Evaluate on test set, print results table, save JSON |

---

## Project structure

```
always_in_bot/
├── .devcontainer/
│   └── devcontainer.json       # GitHub Codespaces config
│
├── data/
│   ├── mt5_fetcher.py          # MT5 downloader (Windows)
│   ├── data_loader.py          # Parquet loader + yfinance fallback
│   └── feature_engineer.py     # All technical indicators + normalisation
│
├── env/
│   └── trading_env.py          # Gymnasium AlwaysInEnv
│
├── models/
│   ├── regime_classifier.py    # XGBoost  (4 regimes)
│   ├── price_predictor.py      # LSTM     (directional probability)
│   └── saved/                  # Auto-created; holds .pkl / .pt files
│
├── risk/
│   └── risk_engine.py          # Circuit breaker · corr cap · session sizing
│
├── training/
│   ├── train_supervised.py     # Trains XGBoost + LSTM
│   └── train_rl.py             # Trains PPO, EvalCallback, TensorBoard
│
├── logs/                       # TensorBoard logs + per-symbol metrics JSON
├── results/                    # final_results.json
│
├── config.py                   # ← ALL hyperparameters live here
├── train.py                    # Main entry point
└── requirements.txt
```

---

## Key config knobs (`config.py`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `TOTAL_TIMESTEPS` | `1_000_000` | PPO training steps per symbol |
| `FLIP_PENALTY` | `0.5` | Pip-equiv penalty per reversal (stops churn) |
| `MAX_DRAWDOWN_PCT` | `8 %` | Circuit breaker threshold |
| `LSTM_SEQ_LEN` | `30` | LSTM lookback window (bars) |
| `LSTM_EPOCHS` | `50` | LSTM training epochs |
| `N_REGIMES` | `4` | Regime count (range/up/down/breakout) |
| `TRAIN_START` | `2018-01-01` | Training start |
| `TEST_START` | `2024-01-01` | Out-of-sample start |

---

## Reward function

```
reward =  (price_Δ × position) / ATR          # normalised step PnL
        −  spread_cost − commission            # transaction costs
        −  FLIP_PENALTY      (if FLIP)         # churn deterrent
        −  1.5 × drawdown    (if DD > 2%)      # drawdown penalty
        +  0.1 × Sharpe₂₀                     # rolling consistency bonus
```

---

## Monitoring

```bash
tensorboard --logdir logs/tensorboard
# → open http://localhost:6006
```

Tracks: episode reward, policy/value/entropy loss, eval mean reward per symbol.

---

## Risk rules (applied every bar)

| Rule | Trigger | Action |
|------|---------|--------|
| Circuit breaker | Drawdown > 8 % | Block all flips until DD < 4 % |
| Correlation cap | GOLD + SILVER both active | Halve size on both |
| Session filter | Outside London/NY (07–17 UTC) | 50 % size reduction |
| Rollover guard | 21:30–22:05 UTC (metals) | Block flips, avoid swap cost |

---

## Notes

- **yfinance intraday cap**: free tier limits 1h data to ~730 days.
  The MT5 fetcher has no such limit — use it for full history.
- **GPU**: LSTM auto-detects CUDA. PPO (MLP) trains fine on CPU.
- **Broker symbols**: if `python -m data.mt5_fetcher` can't find your symbol,
  open MT5 → View → Symbols → search "gold", copy the exact name,
  and add it to `MT5_SYMBOLS` in `config.py`.
