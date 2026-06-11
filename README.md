# swing

# Swing Always-In Trading Bot

An autonomous "always-in" trading agent powered by Reinforcement Learning (PPO) and supervised Machine Learning (XGBoost + LSTM) designed to trade **Stock Indices** and **Gold** via MetaTrader 5.

The agent is **never flat** — it holds Long or Short at all times and relies on advanced regime classification to decide when to flip its bias.

---

## Key Features

- **Multi-Asset Portfolio**: Trades `GOLD`, `US500`, `US100`, `US30`, `UK100`, `GER40`, `AUS200`, and `JP225`.
- **Live Interactive HUD**: A custom `textual` Terminal UI (TUI) providing live PnL, dynamic spreads, margin exposure, and interactive scrolling event logs.
- **Deep RL Engine**: Uses Stable-Baselines3 PPO to optimize risk-adjusted returns (Sharpe ratio) over millions of simulated market bars.
- **Robust Risk Guard**: Features a circuit breaker to halt trading during drawdowns, nightly rollover guards to prevent swap slippage, and session time filtering.

---

## Live Trading Dashboard

The bot features a beautiful, fully interactive live dashboard built using `Textual`:

![Live Dashboard Interface]

The dashboard displays:
- **Capital Metrics**: Balance, Equity, Floating PnL, Drawdown.
- **System Health**: Circuit breaker status, MT5 connection states, Loaded Models.
- **Positions**: Live updating table with exact spreads, lots, and dynamic colored PnL.
- **Event Log**: A scrollable console tracking all MT5 execution logs and errors natively in the terminal.

---

## Architecture

```
MT5 Terminal  (open on Windows)
      │
Feature Engineering (Live/Backtest)
  returns · RSI · ATR · MACD · Bollinger · EMA stack
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
         Live Execution Engine / Terminal UI
           MT5 Bridge · Circuit breaker · Session limits
```

---

## Installation & Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   Copy `.env.example` to `.env` and configure your broker's exact symbol suffixes.
   ```bash
   cp .env.example .env
   ```

3. **Fetch Historical Data**
   Make sure MT5 is open and logged into your broker, then run:
   ```bash
   python download_raw_data.py
   ```

4. **Train the Models**
   Train the Supervised Models (LSTM/XGBoost) and the RL PPO Agent.
   ```bash
   python train.py
   ```

---

## Running the Bot

### Backtesting
The models have been trained and validated on historical MT5 data with the following splits:
- **Training Period:** `2002-09-16` to `2021-12-31`
- **Testing (Out-of-Sample) Period:** `2022-01-14` to `2026-05-08`

You can run backtests on the trained models using:
```bash
python backtest.py --symbol US500Cash --from 2024-01-01 --to 2024-12-31 --export-csv --no-plots
```
*Results and trade logs will be saved to the `results/` directory.*

### Live Trading
Start the interactive live trading engine. Ensure your MT5 terminal is open and `TRADING_MODE=DEMO` or `LIVE` is set correctly in `.env`.
```bash
python live_run.py
```
Press `Q` or `Ctrl+C` to gracefully shut down the bot.

---

## Disclaimer
This project is for educational and research purposes. Live trading incurs significant financial risk. Always run bots in `DEMO` mode for at least a month before committing real capital.
