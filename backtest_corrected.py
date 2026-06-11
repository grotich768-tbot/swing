"""
backtest.py  —  Corrected Walk-Forward Backtester for Always-In Trading Bot
──────────────────────────────────────────────────────────────────────────────
Key corrections vs the previous version:

  1. Dollar PnL  — raw price moves converted to actual USD using pip values
                   and ATR-based position sizing (2% account risk per trade)
  2. Sharpe      — computed on percentage returns, not raw price units
  3. Return %    — actual account return in USD, not price-unit ratio
  4. Drawdown    — in USD terms against the account equity curve
  5. Spread cost — in USD (pip value × lot size × spread pips)

Position sizing formula (per bar, dynamically sized):
    risk_usd  = balance × MAX_POSITION_PCT          (e.g. $200 on $10k)
    atr_stop  = ATR × 1.5                           (price units)
    lots      = risk_usd / (atr_stop / pip × pip_usd)
    pnl_usd   = position × (Δprice / pip) × pip_usd × lots

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
# Dollar value per pip per standard lot  (account currency = USD)
# ─────────────────────────────────────────────────────────────────────────────
# GOLD   : pip=$0.01, contract=100oz  → $0.01×100 = $1.00 per pip per lot
# SILVER : pip=$0.001, contract=5000oz → $0.001×5000 = $5.00 per pip per lot
# EURUSD : pip=0.0001, contract=100k  → 0.0001×100000 = $10.00 per pip per lot
# GBPUSD : pip=0.0001, contract=100k  → ~$10.00 per pip per lot (USD quote)
PIP_USD_PER_LOT = {
    "GOLD":   1.00,
    "SILVER": 5.00,
    "EURUSD": 10.00,
    "GBPUSD": 10.00,
    "USDJPY": 6.28,
    "ETHUSD": 0.10,
    "BTCUSD": 0.10,
    "US30":   1.00,
    "US100":  1.00,
    "US500":  1.00,
    "UK100":  1.25,
    "AUS200": 0.65,
    "GER40":  1.10,
    "JP225":  0.0065,
}

# ATR multiplier for the stop-loss used in position sizing
ATR_STOP_MULT = 1.5

# Commission in USD per lot (round-trip)
COMMISSION_USD_PER_LOT = {
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


# ─────────────────────────────────────────────────────────────────────────────
# Position sizer
# ─────────────────────────────────────────────────────────────────────────────
def _lot_size(symbol: str, atr: float, balance: float) -> float:
    """
    ATR-based position size in lots.

    risk_usd  = balance × MAX_POSITION_PCT
    atr_stop  = atr × ATR_STOP_MULT              (price units)
    lots      = risk_usd / (atr_stop_in_pips × pip_usd_per_lot)

    Capped at 10 lots; floored at 0.01 lots (micro-lot).
    """
    pip         = PIP_VALUE[symbol]
    pip_usd     = PIP_USD_PER_LOT[symbol]
    risk_usd    = balance * MAX_POSITION_PCT
    atr_stop    = atr * ATR_STOP_MULT
    atr_pips    = atr_stop / pip
    lots        = risk_usd / (atr_pips * pip_usd + 1e-10)
    return float(np.clip(lots, 0.01, 10.0))


def _step_pnl_usd(
    symbol:       str,
    position:     int,
    price_change: float,
    lots:         float,
    flipped:      bool,
) -> float:
    """
    Convert raw price change to USD PnL for one bar.

    pnl  = position × (Δprice / pip) × pip_usd × lots
    cost = (spread_pips + commission_pips) × pip_usd × lots  (on flip only)
    """
    pip     = PIP_VALUE[symbol]
    pip_usd = PIP_USD_PER_LOT[symbol]
    pnl     = position * (price_change / pip) * pip_usd * lots

    if flipped:
        spread_cost = SPREAD_PIPS[symbol] * pip_usd * lots
        comm_cost   = COMMISSION_USD_PER_LOT[symbol] * lots
        pnl        -= (spread_cost + comm_cost)

    return pnl


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest runner
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(symbol, date_from, date_to, model_tag="final"):
    from stable_baselines3 import PPO

    # ── Find model ─────────────────────────────────────────────────────────────
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
            f"No model found for {symbol}.\n"
            f"  Run: python train.py\n"
            f"  Looked for: {final_path}"
        )
    logger.info(f"[{symbol}] Loading model: {path.name}")
    model = PPO.load(str(path))

    # ── Load multi-TF features ─────────────────────────────────────────────────
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

    # ── ML model outputs ───────────────────────────────────────────────────────
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

    # ── Build env and run ──────────────────────────────────────────────────────
    env = AlwaysInEnv(
        features_df     = feats,
        symbol          = symbol,
        regime_proba    = regime_proba,
        direction_proba = direction_proba,
        mode            = "test",
    )

    obs, _ = env.reset()
    done   = False

    # ── Step-by-step collection with proper USD PnL ────────────────────────────
    rows    = []
    balance = INITIAL_BALANCE
    peak    = INITIAL_BALANCE

    prev_price = float(env._close[env._step_idx])

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(int(action))
        done = done or truncated

        idx        = env._step_idx
        curr_price = float(env._close[min(idx, len(env._close)-1)])
        atr        = float(env._atr[min(idx,  len(env._atr)-1)])
        ts         = feats.index[min(idx, len(feats)-1)]
        flipped    = (int(action) == 1)
        position   = info["position"]

        # Dollar PnL this bar
        lots    = _lot_size(symbol, atr, balance)
        pnl_usd = _step_pnl_usd(
            symbol, position, curr_price - prev_price, lots, flipped
        )

        balance  = balance + pnl_usd
        peak     = max(peak, balance)
        dd_usd   = peak - balance
        dd_pct   = dd_usd / peak if peak > 0 else 0.0

        rows.append({
            "timestamp":   ts,
            "price":       curr_price,
            "position":    position,
            "action":      int(action),
            "lots":        lots,
            "pnl_usd":     pnl_usd,
            "balance_usd": balance,
            "drawdown_pct":dd_pct,
        })
        prev_price = curr_price

    log = pd.DataFrame(rows).set_index("timestamp")
    log["flipped"] = log["action"] == 1

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = _compute_metrics(log, symbol, INITIAL_BALANCE)
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
        f"Flips={metrics['n_flips']}"
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — all computed on USD returns
# ─────────────────────────────────────────────────────────────────────────────
def _max_streak(mask):
    max_s = cur = 0
    for v in mask:
        cur   = cur + 1 if v else 0
        max_s = max(max_s, cur)
    return max_s


def _compute_metrics(log: pd.DataFrame, symbol: str, initial_balance: float) -> dict:
    pnl_usd  = log["pnl_usd"].values
    balances = log["balance_usd"].values
    flips    = log["flipped"].sum()

    total_pnl_usd = float(pnl_usd.sum())
    total_return  = (balances[-1] - initial_balance) / initial_balance

    # Percentage returns per bar — dimensionless, correct for Sharpe
    pct_returns  = pnl_usd / np.maximum(
        np.roll(balances, 1), initial_balance
    )
    pct_returns[0] = pnl_usd[0] / initial_balance

    # Annualised Sharpe on percentage returns (H1 bars → ~6000/year)
    bars_yr  = 252 * 24
    mean_r   = pct_returns.mean()
    std_r    = pct_returns.std() + 1e-10
    sharpe   = mean_r / std_r * np.sqrt(bars_yr)

    # Sortino — downside deviation only
    neg      = pct_returns[pct_returns < 0]
    sortino  = (mean_r / (neg.std() + 1e-10) * np.sqrt(bars_yr)
                if len(neg) > 0 else 0.0)

    # Max drawdown in USD and %
    peak     = np.maximum.accumulate(balances)
    dd_abs   = peak - balances
    dd_pct   = dd_abs / (peak + 1e-10)
    max_dd   = float(dd_pct.max())
    max_dd_usd = float(dd_abs.max())

    calmar   = total_return / (max_dd + 1e-10)

    # Win rate and profit factor on USD PnL
    win_rate = float((pnl_usd > 0).mean())
    gp       = pnl_usd[pnl_usd > 0].sum()
    gl       = abs(pnl_usd[pnl_usd < 0].sum())
    pf       = float(gp / (gl + 1e-10))

    long_b   = (log["position"] ==  1).sum()
    short_b  = (log["position"] == -1).sum()

    avg_lots = float(log["lots"].mean())
    avg_win  = float(pnl_usd[pnl_usd > 0].mean()) if (pnl_usd > 0).any() else 0.0
    avg_loss = float(pnl_usd[pnl_usd < 0].mean()) if (pnl_usd < 0).any() else 0.0

    return {
        "total_pnl_usd":    total_pnl_usd,
        "total_return":     total_return,
        "final_balance":    float(balances[-1]),
        "sharpe":           float(sharpe),
        "sortino":          float(sortino),
        "calmar":           float(calmar),
        "max_drawdown":     max_dd,
        "max_drawdown_usd": max_dd_usd,
        "win_rate":         win_rate,
        "profit_factor":    pf,
        "n_flips":          int(flips),
        "long_pct":         long_b  / max(len(log), 1),
        "short_pct":        short_b / max(len(log), 1),
        "avg_lots":         avg_lots,
        "avg_win_usd":      avg_win,
        "avg_loss_usd":     avg_loss,
        "expectancy_usd":   avg_win * win_rate + avg_loss * (1 - win_rate),
        "max_win_streak":   _max_streak(pnl_usd > 0),
        "max_loss_streak":  _max_streak(pnl_usd < 0),
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
    window = start + relativedelta(months=train_months) + timedelta(weeks=purge_weeks)

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
                    "n_flips":    m["n_flips"],
                })
            except Exception as e:
                logger.warning(f"[{sym}] {wf_from}→{wf_to}: {e}")
        window += relativedelta(months=test_months)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(all_metrics, save=False):
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
    fig.suptitle("Always-In Bot — Backtest (Corrected USD PnL)", fontsize=14, fontweight="bold")
    gs      = gridspec.GridSpec(n, 3, figure=fig, hspace=0.45, wspace=0.35)
    colours = {"GOLD": "#FFD700", "SILVER": "#C0C0C0",
               "EURUSD": "#4A90D9", "GBPUSD": "#E74C3C"}

    for row, sym in enumerate(symbols):
        m   = all_metrics[sym]
        log = m["trade_log"]
        c   = colours.get(sym, "#7F8C8D")

        # ── Equity curve (USD) ────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.plot(log.index, log["balance_usd"], color=c, linewidth=1.2)
        ax1.axhline(INITIAL_BALANCE, color="grey", linestyle="--",
                    linewidth=0.8, alpha=0.6, label=f"Start ${INITIAL_BALANCE:,.0f}")
        ax1.set_title(f"{sym}  |  Equity Curve (USD)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Account Balance ($)")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax1.grid(True, alpha=0.3)
        final  = log["balance_usd"].iloc[-1]
        colour = "green" if final >= INITIAL_BALANCE else "red"
        ax1.annotate(
            f"${final:,.0f}\n({m['total_return']:+.1%})",
            xy=(log.index[-1], final),
            fontsize=8, color=colour, fontweight="bold", ha="right", va="bottom"
        )

        # Shade profitable / losing bars
        pnl = log["pnl_usd"].values
        pos_mask = pnl > 0
        neg_mask = pnl < 0
        ax1.fill_between(log.index, INITIAL_BALANCE,
                         log["balance_usd"].where(pos_mask),
                         alpha=0.05, color="green")
        ax1.fill_between(log.index, INITIAL_BALANCE,
                         log["balance_usd"].where(neg_mask),
                         alpha=0.05, color="red")

        # ── Drawdown (USD %) ──────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[row, 1])
        dd  = log["drawdown_pct"].values * 100
        ax2.fill_between(log.index, -dd, 0, color="red", alpha=0.4)
        ax2.plot(log.index, -dd, color="darkred", linewidth=0.8)
        ax2.set_title(f"{sym}  |  Drawdown (%)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Drawdown (%)")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax2.grid(True, alpha=0.3)
        ax2.annotate(
            f"Max: {m['max_drawdown']:.2%}\n(${m['max_drawdown_usd']:,.0f})",
            xy=(0.02, 0.08), xycoords="axes fraction",
            fontsize=8, color="darkred"
        )

        # ── Position + flips on price ─────────────────────────────────────────
        ax3 = fig.add_subplot(gs[row, 2])
        price      = log["price"].values
        long_mask  = log["position"].values ==  1
        short_mask = log["position"].values == -1
        ax3.plot(log.index[long_mask],  price[long_mask],
                 color="green", linewidth=0.8, label="Long",  alpha=0.7)
        ax3.plot(log.index[short_mask], price[short_mask],
                 color="red",   linewidth=0.8, label="Short", alpha=0.7)
        flip_idx = log[log["flipped"]].index
        flip_px  = log.loc[flip_idx, "price"].values
        ax3.scatter(flip_idx, flip_px, marker="^", color="black",
                    s=10, zorder=5, label=f"Flip ({m['n_flips']})", alpha=0.5)
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
# Report table
# ─────────────────────────────────────────────────────────────────────────────
def print_report(all_metrics):
    table = Table(
        title="Backtest — Corrected USD Performance",
        show_lines=True, header_style="bold cyan"
    )
    for name, style, just in [
        ("Symbol",       "cyan",   "left"),
        ("Period",       "white",  "left"),
        ("PnL (USD)",    "green",  "right"),
        ("Return %",     "green",  "right"),
        ("Sharpe",       "yellow", "right"),
        ("Sortino",      "yellow", "right"),
        ("Calmar",       "yellow", "right"),
        ("Max DD",       "red",    "right"),
        ("Max DD $",     "red",    "right"),
        ("Win Rate",     "blue",   "right"),
        ("Prof. Factor", "blue",   "right"),
        ("Expectancy",   "blue",   "right"),
        ("Avg Lots",     "white",  "right"),
        ("Flips",        "white",  "right"),
        ("L/S %",        "white",  "right"),
    ]:
        table.add_column(name, style=style, justify=just)

    for sym, m in all_metrics.items():
        pnl_c = (f"[green]+${m['total_pnl_usd']:,.2f}[/green]"
                 if m["total_pnl_usd"] >= 0
                 else f"[red]-${abs(m['total_pnl_usd']):,.2f}[/red]")
        ret_c = (f"[green]+{m['total_return']:.2%}[/green]"
                 if m["total_return"] >= 0
                 else f"[red]{m['total_return']:.2%}[/red]")
        sh_c  = (f"[green]{m['sharpe']:.3f}[/green]" if m["sharpe"] > 1.5 else
                 f"[yellow]{m['sharpe']:.3f}[/yellow]" if m["sharpe"] > 0 else
                 f"[red]{m['sharpe']:.3f}[/red]")
        dd_c  = (f"[red]{m['max_drawdown']:.2%}[/red]"
                 if m["max_drawdown"] > 0.05
                 else f"[yellow]{m['max_drawdown']:.2%}[/yellow]"
                 if m["max_drawdown"] > 0.02
                 else f"[green]{m['max_drawdown']:.2%}[/green]")
        exp_c = (f"[green]+${m['expectancy_usd']:.3f}[/green]"
                 if m["expectancy_usd"] > 0
                 else f"[red]${m['expectancy_usd']:.3f}[/red]")

        table.add_row(
            sym,
            f"{m['date_from']} → {m['date_to']}",
            pnl_c, ret_c, sh_c,
            f"{m['sortino']:.3f}",
            f"{m['calmar']:.3f}",
            dd_c,
            f"${m['max_drawdown_usd']:,.2f}",
            f"{m['win_rate']:.2%}",
            f"{m['profit_factor']:.2f}",
            exp_c,
            f"{m['avg_lots']:.3f}",
            str(m["n_flips"]),
            f"{m['long_pct']:.0%} / {m['short_pct']:.0%}",
        )

    console.print()
    console.print(table)

    # Portfolio summary
    sharpes  = [m["sharpe"]         for m in all_metrics.values()]
    returns  = [m["total_return"]   for m in all_metrics.values()]
    dds      = [m["max_drawdown"]   for m in all_metrics.values()]
    pnls     = [m["total_pnl_usd"]  for m in all_metrics.values()]
    exp      = [m["expectancy_usd"] for m in all_metrics.values()]

    console.print(
        f"\n[bold]Portfolio Summary[/bold]\n"
        f"  Total PnL    : [{'green' if sum(pnls)>=0 else 'red'}]"
        f"${sum(pnls):+,.2f}[/{'green' if sum(pnls)>=0 else 'red'}]\n"
        f"  Mean Sharpe  : [yellow]{np.mean(sharpes):.3f}[/yellow]\n"
        f"  Mean Return  : [green]{np.mean(returns):.2%}[/green]\n"
        f"  Worst DD     : [red]{max(dds):.2%}[/red]\n"
        f"  Mean Expect. : [green]${np.mean(exp):.4f}[/green] per bar\n"
    )

    # Sanity check — warn if Sharpe looks unrealistic
    mean_sh = np.mean(sharpes)
    if mean_sh > 5.0:
        console.print(
            "[bold yellow]⚠  Sharpe > 5.0 detected.[/bold yellow]\n"
            "   Verify: (1) train/test dates do not overlap in config.py\n"
            "           (2) TRAIN_END < TEST_START\n"
            "           (3) No look-ahead in feature_engineer.py\n"
            "   A Sharpe above 3-4 OOS is very rare in live trading.\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Backtest Always-In bot with corrected USD PnL."
    )
    parser.add_argument("--symbols",      nargs="+", default=SYMBOLS)
    parser.add_argument("--from",         dest="date_from", default=TEST_START)
    parser.add_argument("--to",           dest="date_to",   default=TEST_END)
    parser.add_argument("--model",        choices=["final","best"], default="final")
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
║     Always-In Bot  —  Backtest (USD PnL Corrected)   ║
╚══════════════════════════════════════════════════════╝[/bold cyan]
""")

    if args.walk_forward:
        console.rule("[bold]Walk-Forward Evaluation")
        wf = run_walk_forward(
            args.symbols, args.date_from, args.date_to,
            args.train_months, args.test_months, args.purge_weeks
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
            all_metrics[sym] = run_backtest(sym, args.date_from, args.date_to, args.model)
        except FileNotFoundError as e:
            logger.error(str(e))
        except Exception as e:
            logger.error(f"[{sym}] {e}", exc_info=True)

    if not all_metrics:
        logger.error("No results — run python train.py first.")
        return

    print_report(all_metrics)

    if args.export_csv:
        for sym, m in all_metrics.items():
            p = RESULT_DIR / f"tradelog_{sym}_{args.date_from}_{args.date_to}.csv"
            m["trade_log"].to_csv(p)
            logger.info(f"Trade log → {p}")

    # Save JSON (exclude trade_log)
    summary = {s: {k: v for k, v in m.items() if k != "trade_log"}
               for s, m in all_metrics.items()}
    out = RESULT_DIR / f"backtest_{args.date_from}_{args.date_to}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Metrics → {out}")

    if not args.no_plots:
        plot_results(all_metrics, save=args.save_plots)


if __name__ == "__main__":
    main()
