"""
backtest.py  —  Always-In Trading Bot Backtester  (v3 — Fixed Position Sizing)
──────────────────────────────────────────────────────────────────────────────
Corrections applied:

  v1 → raw price units (Sharpe 17, PnL in pips not dollars)
  v2 → compounding lot sizes (balance grew → lots grew → $920k illusion)
  v3 → FIXED-BALANCE sizing (always risk 2% of INITIAL $10k per trade)
         • Lot size = $200 risk / (ATR_stop_pips × pip_usd_per_lot)
         • Does not grow with profits → honest strategy evaluation
         • Account equity still tracks real P&L, just lot size is fixed
         • Sharpe computed on % returns of INITIAL balance

Position sizing
───────────────
  risk_usd  = INITIAL_BALANCE × 2%                ($200, fixed)
  atr_stop  = ATR × 1.5                           (price units)
  lots      = risk_usd / (atr_stop_pips × pip$)   (varies with ATR only)
  max lots  = 2.0  (broker-realistic cap)
  min lots  = 0.01 (micro-lot floor)

Usage:
    python backtest.py
    python backtest.py --symbol GOLD
    python backtest.py --from 2024-01-01 --to 2024-12-31
    python backtest.py --walk-forward
    python backtest.py --save-plots
    python backtest.py --export-csv
"""

import sys
import json
import argparse
import os
import platform
import pathlib

# ── Windows compatibility fix ──────────────────────────────────────────────
# Models trained on Linux (Codespaces) contain PosixPath objects in their
# pickle data. Windows can't instantiate PosixPath, so we remap it.
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import print as rprint

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SYMBOLS, TEST_START, TEST_END, MODEL_DIR, LOG_DIR, RESULT_DIR,
    INITIAL_BALANCE, MAX_POSITION_PCT, PIP_VALUE, SPREAD_PIPS,
)
from data.data_loader import DataLoader
from data.feature_engineer import FeatureEngineer
from models.regime_classifier import RegimeClassifier
from models.price_predictor import LSTMDirectionModel
from env.trading_env import AlwaysInEnv

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Instrument constants
# ─────────────────────────────────────────────────────────────────────────────
# Dollar value of 1 pip move on 1 standard lot (account currency = USD)
#   GOLD   : pip=$0.01, contract=100oz  → $0.01 × 100  = $1.00 /pip/lot
#   SILVER : pip=$0.001, contract=5000oz → $0.001 × 5000 = $5.00 /pip/lot
#   EURUSD : pip=0.0001, contract=100k  → 0.0001 × 100000 = $10.00 /pip/lot
#   GBPUSD : pip=0.0001, contract=100k  → ~$10.00 /pip/lot
PIP_USD_PER_LOT = {
    # Broker-verified: USD per pip per 1.0 standard lot
    "GOLD":   10.00,
    "SILVER": 50.00,
    "EURUSD": 10.00,
    "GBPUSD": 10.00,
    "USDJPY": 6.24,
    "ETHUSD": 1.00,
    "BTCUSD": 1.00,
    "LTCUSD": 1.00,
    "US30":   1.00,
    "US100":  1.00,
    "US500":  1.00,
    "UK100":  1.00,
    "AUS200": 1.00,
    "GER40":  1.00,
    "JP225":  1.00,
}

# Round-trip commission in USD per standard lot
COMMISSION_USD = {
    "GOLD":   7.0,
    "SILVER": 3.5,
    "EURUSD": 7.0,
    "GBPUSD": 7.0,
    "USDJPY": 7.0,
    "ETHUSD": 0.0,
    "BTCUSD": 0.0,
    "US30":   0.0,
    "US100":  0.0,
    "US500":  0.0,
    "UK100":  0.0,
    "AUS200": 0.0,
    "GER40":  0.0,
    "JP225":  0.0,
}

ATR_STOP_MULT = 1.5
MAX_LOTS      = 0.5    # conservative cap — prevents over-leverage on $10k account
MIN_LOTS      = 0.01   # micro-lot floor


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-balance position sizer  (KEY FIX — uses INITIAL_BALANCE always)
# ─────────────────────────────────────────────────────────────────────────────
def _lot_size(symbol: str, atr: float) -> float:
    """
    ATR-based lot size using FIXED initial balance for risk calculation.
    This prevents compounding from inflating position sizes.

    risk_usd  = INITIAL_BALANCE × MAX_POSITION_PCT   (always $200 on $10k)
    atr_stop  = atr × ATR_STOP_MULT                  (price units)
    lots      = risk_usd / (atr_stop_pips × pip_usd)
    """
    pip         = PIP_VALUE[symbol]
    pip_usd     = PIP_USD_PER_LOT[symbol]
    risk_usd    = INITIAL_BALANCE * MAX_POSITION_PCT   # fixed, never changes
    atr_pips    = (atr * ATR_STOP_MULT) / pip
    lots        = risk_usd / (atr_pips * pip_usd + 1e-10)
    return float(np.clip(lots, MIN_LOTS, MAX_LOTS))


def _pnl_usd(symbol: str, position: int, price_change: float,
              lots: float, flipped: bool) -> float:
    """
    Convert raw price change to USD PnL for one bar.

    Spread handling:
      SPREAD_PIPS[symbol] is stored in PIPS (already converted from MT5 points).
      MT5 reports spread in points — conversion: spread_pips = spread_points / 10
      This is handled in broker_config.py and config.py at load time.

    PnL formula:
      price_change_pips = price_change / pip_size
      pnl = position × price_change_pips × pip_usd × lots

    Spread cost on flip:
      cost = spread_pips × pip_usd × lots
      e.g. GOLD: 6.0 pips × $0.10/pip × 0.239 lots = $0.143
      e.g. US30: 3.9 pips × $1.00/pip × 0.499 lots = $1.95
    """
    pip     = PIP_VALUE[symbol]
    pip_usd = PIP_USD_PER_LOT[symbol]
    pnl     = position * (price_change / pip) * pip_usd * lots
    if flipped:
        pnl -= SPREAD_PIPS[symbol] * pip_usd * lots   # spread in pips × $/pip × lots
        pnl -= COMMISSION_USD[symbol] * lots            # round-trip commission
    return pnl


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest runner
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_backtest_dates(symbol: str, date_from: str, date_to: str):
    """Respect per-symbol test date overrides from config."""
    try:
        from config import SYMBOL_TEST_START, SYMBOL_TEST_END
        date_from = SYMBOL_TEST_START.get(symbol, date_from)
        date_to   = SYMBOL_TEST_END.get(symbol,   date_to)
    except ImportError:
        pass
    return date_from, date_to


def run_backtest(symbol: str, date_from: str, date_to: str,
                 model_tag: str = "final",
                 use_ensemble: bool = False,
                 use_regime_router: bool = False) -> dict:

    # Resolve per-symbol date overrides (e.g. BTCUSD test from 2024)
    date_from, date_to = _resolve_backtest_dates(symbol, date_from, date_to)

    from stable_baselines3 import PPO
    from models.ensemble_predictor import EnsemblePredictor
    from models.regime_router import RegimeRouter

    # ── Load model ────────────────────────────────────────────────────────────
    if use_regime_router:
        model        = RegimeRouter.load_or_ensemble(symbol)
        _use_router  = True
        logger.info(f"[{symbol}] Loaded RegimeRouter: {model}")
    elif use_ensemble:
        model        = EnsemblePredictor.load_or_single(symbol)
        _use_router  = False
        logger.info(f"[{symbol}] Loaded ensemble: {len(model)} models")
    else:
        _use_router  = False
        final_path = MODEL_DIR / f"ppo_{symbol}_seed42_final.zip"
        if not final_path.exists():
            final_path = MODEL_DIR / f"ppo_{symbol}_final.zip"
        best_path  = MODEL_DIR / f"ppo_{symbol}" / "best_model.zip"
        if model_tag == "best" and best_path.exists():
            path = best_path
        elif final_path.exists():
            path = final_path
        elif best_path.exists():
            path = best_path
        else:
            raise FileNotFoundError(
                f"No model for {symbol}. Run: python train.py\n"
                f"  Looked for: {final_path}"
            )
        logger.info(f"[{symbol}] Loading: {path.name}")
        model = PPO.load(str(path))

    # ── Load full multi-TF features ───────────────────────────────────────────
    loader   = DataLoader()
    engineer = FeatureEngineer(normalise=True)

    raw_h1  = loader.load(symbol, "H1",  date_from, date_to)
    raw_d1  = loader.load(symbol, "D1",  date_from, date_to)
    raw_h4  = loader.load(symbol, "H4",  date_from, date_to)
    raw_m15 = loader.load(symbol, "M15", date_from, date_to)

    if raw_h1 is None or len(raw_h1) < 100:
        raise ValueError(f"[{symbol}] Not enough data for {date_from}→{date_to}")

    feats = engineer.transform_multi_tf(
        df_h1=raw_h1, df_d1=raw_d1, df_h4=raw_h4, df_m15=raw_m15, symbol=symbol
    )
    feat_cols = [c for c in feats.columns
                 if not c.startswith("_") and not c.startswith("target")]

    # ── ML outputs ────────────────────────────────────────────────────────────
    regime_proba = direction_proba = None
    for tag in ("shared", symbol):
        try:
            regime_proba = RegimeClassifier.load(tag).predict_proba(
                pd.DataFrame(feats[feat_cols].values, columns=feat_cols)
            )
            break
        except FileNotFoundError:
            continue

    n_feat = len(feat_cols)
    for tag in ("shared", symbol):
        try:
            direction_proba = LSTMDirectionModel.load(n_feat, tag).predict_proba(
                feats[feat_cols].values
            )
            break
        except FileNotFoundError:
            continue

    # ── Run episode ───────────────────────────────────────────────────────────
    env = AlwaysInEnv(
        features_df     = feats,
        symbol          = symbol,
        regime_proba    = regime_proba,
        direction_proba = direction_proba,
        mode            = "test",
    )
    obs, _ = env.reset()
    done   = False

    rows       = []
    balance    = INITIAL_BALANCE
    peak       = INITIAL_BALANCE
    prev_price = float(env._close[env._step_idx])

    current_pos = 0 # Start flat
    env._position = current_pos
    obs = env._get_obs()

    # TP/SL Parameters
    TP_ATR_MULT = 2.0
    SL_ATR_MULT = 1.0
    entry_price = 0.0
    entry_atr = 0.0

    while not done:
        idx = env._step_idx
        curr_price_now = float(env._close[min(idx, len(env._close)-1)])
        
        if hasattr(model, "models"):
            votes = [int(m.predict(obs, deterministic=True)[0]) for m in model.models]
            long_votes = votes.count(0)
            short_votes = votes.count(1)
            
            if long_votes == len(votes):
                model_target = 1
            elif short_votes == len(votes):
                model_target = -1
            else:
                model_target = 0 # Disagreement -> FLAT
        else:
            action, _ = model.predict(obs, deterministic=True)
            model_target = 1 if int(action) == 0 else -1
            
        # Hold trade until TP or SL is hit. Only follow models if FLAT.
        if current_pos == 0:
            target_pos = model_target
        else:
            target_pos = current_pos
            
        # Check TP / SL
        if current_pos != 0 and entry_atr > 0:
            if current_pos == 1:
                if curr_price_now >= entry_price + (entry_atr * TP_ATR_MULT): target_pos = 0
                elif curr_price_now <= entry_price - (entry_atr * SL_ATR_MULT): target_pos = 0
            elif current_pos == -1:
                if curr_price_now <= entry_price - (entry_atr * TP_ATR_MULT): target_pos = 0
                elif curr_price_now >= entry_price + (entry_atr * SL_ATR_MULT): target_pos = 0
            
        dummy_action = 0 if target_pos == 1 else 1
        _, _, done, truncated, info = env.step(dummy_action)
        done = done or truncated

        is_entry = (current_pos == 0 and target_pos != 0)
        is_flip = (current_pos == 1 and target_pos == -1) or (current_pos == -1 and target_pos == 1)
        flipped = is_entry or is_flip
        
        if is_entry or is_flip:
            entry_price = float(env._close[min(env._step_idx, len(env._close)-1)])
            entry_atr = float(env._atr[min(env._step_idx, len(env._atr)-1)])
        elif target_pos == 0:
            entry_price = 0.0
            entry_atr = 0.0

        env._position = target_pos
        current_pos = target_pos
        obs = env._get_obs()
        position = current_pos

        idx        = env._step_idx
        curr_price = float(env._close[min(idx, len(env._close)-1)])
        atr        = float(env._atr[min(idx,  len(env._atr)-1)])
        ts         = feats.index[min(idx, len(feats)-1)]

        # Fixed-balance lot size and USD PnL
        lots       = _lot_size(symbol, atr) if position != 0 else 0.0
        step_pnl   = _pnl_usd(symbol, position, curr_price - prev_price,
                               lots, flipped)

        balance    = balance + step_pnl
        peak       = max(peak, balance)
        dd_usd     = peak - balance
        dd_pct     = dd_usd / peak if peak > 0 else 0.0

        rows.append({
            "timestamp":    ts,
            "price":        curr_price,
            "atr":          atr,
            "position":     position,
            "action":       1 if flipped else 0,
            "lots":         lots,
            "pnl_usd":      step_pnl,
            "balance_usd":  balance,
            "drawdown_pct": dd_pct,
        })
        prev_price = curr_price

    log           = pd.DataFrame(rows).set_index("timestamp")
    log["flipped"] = log["action"] == 1

    metrics = _compute_metrics(log, symbol)
    metrics.update({
        "symbol":    symbol,
        "date_from": date_from,
        "date_to":   date_to,
        "n_bars":    len(log),
        "trade_log": log,
    })

    logger.info(
        f"[{symbol}]  "
        f"PnL=${metrics['total_pnl_usd']:+,.2f}  "
        f"Return={metrics['total_return']:.2%}  "
        f"Sharpe={metrics['sharpe']:.3f}  "
        f"MaxDD={metrics['max_drawdown']:.2%}  "
        f"WinRate={metrics['win_rate']:.2%}  "
        f"AvgLots={metrics['avg_lots']:.3f}  "
        f"Flips={metrics['n_flips']}"
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Metrics  (Sharpe on % of INITIAL_BALANCE — no compounding distortion)
# ─────────────────────────────────────────────────────────────────────────────
def _max_streak(mask: np.ndarray) -> int:
    max_s = cur = 0
    for v in mask:
        cur   = cur + 1 if v else 0
        max_s = max(max_s, cur)
    return max_s


def _compute_metrics(log: pd.DataFrame, symbol: str) -> dict:
    pnl_usd  = log["pnl_usd"].values
    balances = log["balance_usd"].values
    flips    = int(log["flipped"].sum())

    total_pnl_usd = float(pnl_usd.sum())
    total_return  = (balances[-1] - INITIAL_BALANCE) / INITIAL_BALANCE

    # ── Max drawdown ──────────────────────────────────────────────────────────
    peak       = np.maximum.accumulate(balances)
    dd_pct     = (peak - balances) / (peak + 1e-10)
    dd_usd     = peak - balances
    max_dd     = float(dd_pct.max())
    max_dd_usd = float(dd_usd.max())
    calmar     = total_return / (max_dd + 1e-10)

    # ── Bar-level win rate & profit factor ────────────────────────────────────
    win_rate = float((pnl_usd > 0).mean())
    gp       = pnl_usd[pnl_usd > 0].sum()
    gl       = abs(pnl_usd[pnl_usd < 0].sum())
    pf       = float(gp / (gl + 1e-10))
    avg_win  = float(pnl_usd[pnl_usd > 0].mean()) if (pnl_usd > 0).any() else 0.0
    avg_loss = float(pnl_usd[pnl_usd < 0].mean()) if (pnl_usd < 0).any() else 0.0

    # ── TRADE-LEVEL metrics (flip-to-flip) ────────────────────────────────────
    # Each "trade" = position held from one flip to the next.
    # This is the honest unit — removes serial correlation from bar holding.
    trade_pnls = []
    flip_idx   = [0] + list(np.where(log["flipped"].values)[0]) + [len(log)]
    for i in range(len(flip_idx) - 1):
        start, end   = flip_idx[i], flip_idx[i+1]
        trade_pnl    = log["pnl_usd"].iloc[start:end].sum()
        trade_pnls.append(trade_pnl)

    trade_pnls   = np.array(trade_pnls)
    n_trades     = len(trade_pnls)
    trade_wins   = float((trade_pnls > 0).mean())
    trade_avg_w  = float(trade_pnls[trade_pnls > 0].mean()) if (trade_pnls > 0).any() else 0.0
    trade_avg_l  = float(trade_pnls[trade_pnls < 0].mean()) if (trade_pnls < 0).any() else 0.0
    trade_expect = trade_wins * trade_avg_w + (1 - trade_wins) * trade_avg_l

    # Trade-level Sharpe (annualised by trade frequency)
    trades_per_yr = n_trades / max((balances[-1] != INITIAL_BALANCE), 1)
    # Estimate trades per year from actual frequency
    n_days   = (log.index[-1] - log.index[0]).days + 1
    tpy      = n_trades / max(n_days, 1) * 365
    t_mean   = trade_pnls.mean() / INITIAL_BALANCE
    t_std    = trade_pnls.std()  / INITIAL_BALANCE + 1e-10
    trade_sharpe = (t_mean / t_std) * np.sqrt(tpy)

    # ── DAILY Sharpe (aggregated, removes some serial correlation) ────────────
    daily_pnl = log["pnl_usd"].resample("D").sum()
    daily_pct = daily_pnl / INITIAL_BALANCE
    mean_d    = daily_pct.mean()
    std_d     = daily_pct.std() + 1e-10
    sharpe    = mean_d / std_d * np.sqrt(252)

    neg_d   = daily_pct[daily_pct < 0]
    sortino = (mean_d / (neg_d.std() + 1e-10) * np.sqrt(252)
               if len(neg_d) > 0 else 0.0)

    return {
        "total_pnl_usd":    total_pnl_usd,
        "total_return":     total_return,
        "final_balance":    float(balances[-1]),
        # Daily Sharpe (standard but inflated by serial correlation)
        "sharpe":           float(sharpe),
        "sortino":          float(sortino),
        "calmar":           float(calmar),
        "max_drawdown":     max_dd,
        "max_drawdown_usd": max_dd_usd,
        # Bar-level
        "win_rate":         win_rate,
        "profit_factor":    pf,
        "n_flips":          flips,
        "long_pct":         float((log["position"] ==  1).mean()),
        "short_pct":        float((log["position"] == -1).mean()),
        "avg_lots":         float(log["lots"].mean()),
        "avg_win_usd":      avg_win,
        "avg_loss_usd":     avg_loss,
        "expectancy_usd":   avg_win * win_rate + avg_loss * (1 - win_rate),
        "max_win_streak":   _max_streak(pnl_usd > 0),
        "max_loss_streak":  _max_streak(pnl_usd < 0),
        # Trade-level (honest — flip-to-flip)
        "n_trades":         n_trades,
        "trade_win_rate":   trade_wins,
        "trade_sharpe":     float(trade_sharpe),
        "trade_expectancy": trade_expect,
        "trade_avg_win":    trade_avg_w,
        "trade_avg_loss":   trade_avg_l,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward
# ─────────────────────────────────────────────────────────────────────────────
def run_walk_forward(symbols, overall_from, overall_to,
                     train_months=18, test_months=3, purge_weeks=2):
    from dateutil.relativedelta import relativedelta
    rows   = []
    start  = datetime.strptime(overall_from, "%Y-%m-%d")
    end    = datetime.strptime(overall_to,   "%Y-%m-%d")
    window = (start
              + relativedelta(months=train_months)
              + timedelta(weeks=purge_weeks))

    while window + relativedelta(months=test_months) <= end:
        wf_from = window.strftime("%Y-%m-%d")
        wf_to   = (window + relativedelta(months=test_months)).strftime("%Y-%m-%d")
        for sym in symbols:
            try:
                m = run_backtest(sym, wf_from, wf_to)
                rows.append({
                    "symbol":     sym,
                    "from":       wf_from,
                    "to":         wf_to,
                    "pnl_usd":    m["total_pnl_usd"],
                    "return_pct": m["total_return"],
                    "sharpe":     m["sharpe"],
                    "max_dd":     m["max_drawdown"],
                    "win_rate":   m["win_rate"],
                    "avg_lots":   m["avg_lots"],
                    "n_flips":    m["n_flips"],
                })
            except Exception as e:
                logger.warning(f"[{sym}] {wf_from}→{wf_to}: {e}")
        window += relativedelta(months=test_months)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(all_metrics: dict, save: bool = False):
    headless = (
        save
        or not os.environ.get("DISPLAY")
        or bool(os.environ.get("CODESPACES"))
        or os.environ.get("TERM_PROGRAM") == "vscode"
    )
    try:
        import matplotlib
        matplotlib.use("Agg" if headless else "TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not available — skipping plots.")
        return

    symbols = list(all_metrics.keys())
    n       = len(symbols)
    fig     = plt.figure(figsize=(18, 5 * n))
    fig.suptitle(
        "Always-In Bot — Backtest  (Fixed Position Sizing, USD PnL)",
        fontsize=14, fontweight="bold"
    )
    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.45, wspace=0.35)
    colours = {
        "GOLD":   "#FFD700",
        "SILVER": "#A8A9AD",
        "EURUSD": "#4A90D9",
        "GBPUSD": "#E74C3C",
    }

    for row, sym in enumerate(symbols):
        m   = all_metrics[sym]
        log = m["trade_log"]
        c   = colours.get(sym, "#7F8C8D")

        # ── Equity curve ──────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.plot(log.index, log["balance_usd"], color=c, linewidth=1.2, label=sym)
        ax1.axhline(INITIAL_BALANCE, color="grey", linestyle="--",
                    linewidth=0.8, alpha=0.6)
        ax1.set_title(f"{sym}  |  Equity (USD)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Balance ($)")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax1.grid(True, alpha=0.3)
        final  = log["balance_usd"].iloc[-1]
        colour = "green" if final >= INITIAL_BALANCE else "red"
        ax1.annotate(
            f"${final:,.0f}\n({m['total_return']:+.1%})",
            xy=(log.index[-1], final),
            fontsize=8, color=colour, fontweight="bold", ha="right", va="bottom",
        )

        # ── Drawdown ──────────────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[row, 1])
        dd  = log["drawdown_pct"].values * 100
        ax2.fill_between(log.index, -dd, 0, color="red", alpha=0.4)
        ax2.plot(log.index, -dd, color="darkred", linewidth=0.8)
        ax2.set_title(f"{sym}  |  Drawdown", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Drawdown (%)")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax2.grid(True, alpha=0.3)
        ax2.annotate(
            f"Max DD: {m['max_drawdown']:.2%}\n(${m['max_drawdown_usd']:,.0f})",
            xy=(0.02, 0.08), xycoords="axes fraction",
            fontsize=8, color="darkred",
        )

        # ── Price + position + flips ──────────────────────────────────────────
        ax3 = fig.add_subplot(gs[row, 2])
        price      = log["price"].values
        long_mask  = log["position"].values ==  1
        short_mask = log["position"].values == -1
        ax3.plot(log.index[long_mask],  price[long_mask],
                 color="green", lw=0.8, label="Long",  alpha=0.7)
        ax3.plot(log.index[short_mask], price[short_mask],
                 color="red",   lw=0.8, label="Short", alpha=0.7)
        flip_idx = log[log["flipped"]].index
        ax3.scatter(flip_idx, log.loc[flip_idx, "price"].values,
                    marker="^", color="black", s=10, zorder=5,
                    label=f"Flip ({m['n_flips']})", alpha=0.5)
        ax3.set_title(f"{sym}  |  Positions & Flips", fontsize=10, fontweight="bold")
        ax3.set_ylabel("Price")
        ax3.legend(fontsize=7, loc="upper left")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if headless or save:
        out = RESULT_DIR / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.success(f"Plot saved → {out}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def print_report(all_metrics: dict):
    # ── Table 1: Core performance ─────────────────────────────────────────────
    t1 = Table(
        title=(
            f"Backtest — Fixed Sizing  "
            f"(2% of ${INITIAL_BALANCE:,.0f}, max {MAX_LOTS} lots)"
        ),
        show_lines=True, header_style="bold cyan",
    )
    for name, style, just in [
        ("Symbol",          "cyan",   "left"),
        ("Period",          "white",  "left"),
        ("PnL (USD)",       "green",  "right"),
        ("Return %",        "green",  "right"),
        ("Daily Sharpe",    "yellow", "right"),
        ("Trade Sharpe",    "yellow", "right"),
        ("Max DD",          "red",    "right"),
        ("Win Rate (bar)",  "blue",   "right"),
        ("Win Rate (trade)","blue",   "right"),
        ("Trade Expect $",  "blue",   "right"),
        ("Avg Lots",        "white",  "right"),
        ("Trades",          "white",  "right"),
        ("Flips",           "white",  "right"),
    ]:
        t1.add_column(name, style=style, justify=just)

    for sym, m in all_metrics.items():
        p  = m["total_pnl_usd"]
        r  = m["total_return"]
        ds = m["sharpe"]
        ts = m["trade_sharpe"]
        d  = m["max_drawdown"]

        t1.add_row(
            sym,
            f"{m['date_from']} → {m['date_to']}",
            f"[{'green' if p>=0 else 'red'}]{'+' if p>=0 else ''}${p:,.2f}[/{'green' if p>=0 else 'red'}]",
            f"[{'green' if r>=0 else 'red'}]{r:+.2%}[/{'green' if r>=0 else 'red'}]",
            f"[yellow]{ds:.2f}[/yellow]",
            f"[{'green' if ts>1.5 else 'yellow' if ts>0 else 'red'}]{ts:.2f}[/{'green' if ts>1.5 else 'yellow' if ts>0 else 'red'}]",
            f"[{'red' if d>0.1 else 'yellow' if d>0.05 else 'green'}]{d:.2%}[/{'red' if d>0.1 else 'yellow' if d>0.05 else 'green'}]",
            f"{m['win_rate']:.2%}",
            f"[{'green' if m['trade_win_rate']>0.5 else 'red'}]{m['trade_win_rate']:.2%}[/{'green' if m['trade_win_rate']>0.5 else 'red'}]",
            f"[{'green' if m['trade_expectancy']>0 else 'red'}]${m['trade_expectancy']:+,.2f}[/{'green' if m['trade_expectancy']>0 else 'red'}]",
            f"{m['avg_lots']:.3f}",
            str(m["n_trades"]),
            str(m["n_flips"]),
        )

    console.print()
    console.print(t1)

    # ── Portfolio summary ─────────────────────────────────────────────────────
    pnls    = [m["total_pnl_usd"]    for m in all_metrics.values()]
    dsharp  = [m["sharpe"]           for m in all_metrics.values()]
    tsharp  = [m["trade_sharpe"]     for m in all_metrics.values()]
    returns = [m["total_return"]     for m in all_metrics.values()]
    dds     = [m["max_drawdown"]     for m in all_metrics.values()]
    texps   = [m["trade_expectancy"] for m in all_metrics.values()]

    console.print(
        f"\n[bold]Portfolio Summary[/bold]\n"
        f"  Combined PnL      : [green]${sum(pnls):+,.2f}[/green]\n"
        f"  Mean Daily Sharpe : [yellow]{np.mean(dsharp):.2f}[/yellow]"
        f"  ← inflated by serial correlation / trending year\n"
        f"  Mean Trade Sharpe : [yellow]{np.mean(tsharp):.2f}[/yellow]"
        f"  ← honest flip-to-flip measure\n"
        f"  Mean Return       : [green]{np.mean(returns):.2%}[/green]\n"
        f"  Worst DD          : [red]{max(dds):.2%}[/red]\n"
        f"  Mean Trade Expect : [green]${np.mean(texps):+,.2f}[/green] per trade\n"
    )

    console.print(
        "[dim]Daily Sharpe note: always-in strategies holding through trending "
        "bars show inflated daily Sharpe due to serial correlation. "
        "Trade Sharpe (flip-to-flip) is the honest benchmark. "
        "Sharpe > 2.0 trade-level is genuinely strong.[/dim]\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Backtest Always-In bot with fixed position sizing."
    )
    parser.add_argument("--symbols",      nargs="+", default=SYMBOLS)
    parser.add_argument("--from",         dest="date_from", default=None,
                        help="Test start — defaults to SYMBOL_TEST_START[symbol] or TEST_START")
    parser.add_argument("--to",           dest="date_to",   default=None,
                        help="Test end — defaults to SYMBOL_TEST_END[symbol] or TEST_END")
    parser.add_argument("--model",        choices=["final","best"], default="final")
    parser.add_argument("--ensemble",     action="store_true",
                        help="Use 5-seed ensemble majority vote")
    parser.add_argument("--regime-router",action="store_true", dest="regime_router",
                        help="Use regime-specific agents + ensemble safety net")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--train-months", type=int, default=18)
    parser.add_argument("--test-months",  type=int, default=3)
    parser.add_argument("--purge-weeks",  type=int, default=2)
    parser.add_argument("--save-plots",   action="store_true")
    parser.add_argument("--no-plots",     action="store_true")
    parser.add_argument("--export-csv",   action="store_true")
    args = parser.parse_args()

    rprint("""
[bold cyan]╔══════════════════════════════════════════════════════╗
║   Always-In Bot  —  Backtest (Fixed Position Sizing) ║
╚══════════════════════════════════════════════════════╝[/bold cyan]
""")

    if args.walk_forward:
        console.rule("[bold]Walk-Forward Evaluation")
        wf = run_walk_forward(
            args.symbols, args.date_from, args.date_to,
            args.train_months, args.test_months, args.purge_weeks,
        )
        console.print(wf.to_string(index=False))
        out = RESULT_DIR / "walk_forward.csv"
        wf.to_csv(out, index=False)
        logger.success(f"Walk-forward → {out}")
        return

    console.rule(f"[bold]Backtest  {args.date_from} → {args.date_to}")
    all_metrics = {}
    for sym in args.symbols:
        try:
            all_metrics[sym] = run_backtest(
                sym,
                args.date_from or TEST_START,
                args.date_to   or TEST_END,
                args.model,
                use_ensemble      = getattr(args, "ensemble",      False),
                use_regime_router = getattr(args, "regime_router", False),
            )
        except FileNotFoundError as e:
            logger.error(str(e))
        except Exception as e:
            logger.error(f"[{sym}] {e}", exc_info=True)

    if not all_metrics:
        logger.error("No results. Run: python train.py")
        return

    print_report(all_metrics)

    # ── Determine model type tag for filenames ────────────────────────────────
    model_tag = "single"
    if getattr(args, "regime_router", False):
        model_tag = "regime"
    elif getattr(args, "ensemble", False):
        model_tag = "ensemble"

    symbols_tag = "_".join(args.symbols) if len(args.symbols) <= 3 else f"{len(args.symbols)}syms"
    date_tag    = f"{(args.date_from or TEST_START)[:10]}_{(args.date_to or TEST_END)[:10]}"

    # ── Export CSV trade logs ──────────────────────────────────────────────────
    if args.export_csv:
        for sym, m in all_metrics.items():
            p = RESULT_DIR / f"tradelog_{sym}_{model_tag}_{date_tag}.csv"
            if "trade_log" in m and m["trade_log"] is not None:
                m["trade_log"].to_csv(p)
                logger.info(f"Trade log → {p}")

    # ── Save JSON metrics ──────────────────────────────────────────────────────
    summary = {
        s: {k: v for k, v in m.items() if k != "trade_log"}
        for s, m in all_metrics.items()
    }

    # Named by symbol + model type + date for easy comparison
    out = RESULT_DIR / f"backtest_{symbols_tag}_{model_tag}_{date_tag}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Metrics → {out}")

    # Also save a latest.json for quick access
    latest = RESULT_DIR / f"backtest_{symbols_tag}_latest.json"
    with open(latest, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Print save location ────────────────────────────────────────────────────
    console.rule("[dim]Saved")
    rprint(f"  [dim]JSON  → {out}[/dim]")
    rprint(f"  [dim]Latest → {latest}[/dim]")

    if not args.no_plots:
        plot_results(all_metrics, save=args.save_plots)


if __name__ == "__main__":
    main()
