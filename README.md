# swing - Version 2

# Swing Z+ Advanced Trading Bot (V2)

An autonomous "always-in" trading agent powered by Reinforcement Learning (PPO) and supervised Machine Learning (XGBoost + LSTM). Version 2 introduces massive architectural upgrades including **Model Ensembling**, **Adaptive Position Sizing**, and **Live Multi-Broker Support**.

---

## What's New in Version 2?

Version 2 represents a major leap in the bot's robustness and feature set. Here is the full breakdown of features successfully implemented into the architecture:

### ✅ Implemented Features

1. **H4 Regime Gate** (`feature_engineer.py` + `trading_env.py`)
   - Flip penalty scales dynamically: **×2 during ranging markets** (discourages chop) and **×0.5 during trending markets** (encourages riding trends).
2. **Session-Aware Reward Shaping** (`trading_env.py`)
   - The PPO agent is penalized/rewarded differently depending on the active trading session (e.g., NY/London overlap multiplier = 1.3×, Asia session multiplier = 0.7×).
3. **VWAP Features** (`feature_engineer.py`)
   - Added Volume Weighted Average Price indicators: `vwap_dev`, `vwap_dist_atr`, and `value_area_pos`.
4. **Volume Imbalance Proxy** (`feature_engineer.py`)
   - Added `vol_imbalance` and Chaikin Money Flow (`cmf_14`) to proxy volume asymmetry.
5. **Spread Dynamics** (`feature_engineer.py`)
   - Incorporated `hl_spread_ratio` and `spread_z` to track high/low spread volatility signals.
6. **Ensemble Training** (`train_rl.py` + `ensemble_predictor.py`)
   - Trains **5 independent PPO models** per symbol using different random seeds. Live execution uses a Majority Rule vote.
7. **Walk-Forward Retraining Pipeline** (`walkforward.py`)
   - Automated monthly retraining pipeline to keep models adapted to recent market regimes.
8. **Adaptive Position Sizing** (`risk_engine.py`)
   - Incorporates Kelly fraction scaling. Position sizes dynamically scale up during winning streaks and scale down during drawdowns.
9. **News Event Filter** (`risk_engine.py`)
   - Automatically blocks bias flips ±15 minutes around major high-impact events (NFP, CPI, and FOMC).

---

### ❌ Planned (But Not Implemented)

The following features were architected but ultimately **not** included in the V2 release:
- **Regime-Specific Agents**: Training 4 separate PPO agents per symbol (range/trend-up/trend-down/breakout) and having XGBoost route them. *(Major architecture change)*
- **Real Tick Volume Features**: Pulling actual buy/sell volume directly from MT5 tick data. *(Proxies were used instead)*
- **Multi-Agent Portfolio Coordinator**: A single mega-PPO agent that sees all symbols simultaneously and outputs a `MultiDiscrete` action space. *(Major architecture change)*
- **Macro Event Features**: An observation feature tracking `hours_to_major_event` (clipped at 48h). *(The news filter blocks trades, but the RL agent is currently blind to the impending event schedule)*
- **Live Paper Trading Comparison**: A system to run a demo account and backtest simultaneously, raising alerts if divergence exceeds 15%.

---

## Live Trading Dashboard

The bot features a beautiful, fully interactive live dashboard built using `Textual`:
- **Capital Metrics**: Balance, Equity, Floating PnL, Drawdown.
- **Positions**: Live updating table with exact spreads, lots, and dynamic colored PnL.
- **Event Log**: A scrollable console tracking all MT5 execution logs and errors natively in the terminal.

*(Note: If running `live2.py` as a background worker, the UI is automatically suppressed to keep logs clean).*

---

## Installation & Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   Copy `.env.example` to `.env` and set your broker suffix (or use `broker_config.py` for auto-detect).
   ```bash
   cp .env.example .env
   ```

3. **Fetch Historical Data**
   Make sure MT5 is open and logged into your broker, then run:
   ```bash
   python download_raw_data.py
   ```

4. **Train the Models**
   Train the Supervised Models and the multi-seed PPO Ensembles.
   ```bash
   python train.py
   ```

---

## Training Periods
Models are trained on extensive historical MT5 data with the following default splits:
- **Training Period:** `2002-09-16` to `2021-12-31`
- **Testing (Out-of-Sample) Period:** `2022-01-14` to `2026-05-08`

*Note: Crypto assets (BTC, ETH, LTC) use an adjusted training period starting from `2017-01-01` due to limited historical data.*

---

## Running the Bot

### Live Trading
Start the interactive live trading engine. Ensure your MT5 terminal is open and `TRADING_MODE` is set correctly in `.env`.

```bash
python live_run.py
```

**How the Live Bot Operates Under the Hood:**
- **The Startup Sequence**: It connects to your MT5 terminal, fetches your live account balance, and initializes the LiveTrader engine.
- **Model Loading**: It strictly loads your 5-seed Ensemble (5 brains) for every symbol, bypassing any old regime routers or manifests.
- **The Live Loop**: Every time a new 1-hour bar closes, it:
  1. Fetches the latest live data from MT5.
  2. Runs the FeatureEngineer to calculate all the multi-timeframe indicators.
  3. Calculates the exact time of day, your current MT5 balance, and streak history.
  4. Assembles the perfect 137-data-point observation array.
- **The Vote**: It feeds that array into all 5 brains simultaneously. If 3 out of 5 (or more) agree that the market is reversing, it triggers a flip. Otherwise, it strictly follows the rule: "let it run til tp or sl" and ignores minor disagreements mid-trade.
- **Execution**: If a flip is triggered, it instantly sends the trade ticket to MT5 to close the current position and open the reverse one.

It will run infinitely in the terminal UI, actively managing all of your symbols at once. Press `Q` or `Ctrl+C` to gracefully shut down the bot.

---

## Disclaimer
This project is for educational and research purposes. Live trading incurs significant financial risk. Always run bots in `DEMO` mode for at least a month before committing real capital.
