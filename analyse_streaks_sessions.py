"""
analyse_streaks_sessions.py  —  Streak & Session Performance Analysis
──────────────────────────────────────────────────────────────────────────────
Analyses the backtest trade log to show:
  1. Average AND max win/loss streaks (trade-level)
  2. Session performance breakdown (Asia / London / Overlap / NY / Off)
  3. Day-of-week performance
  4. Hourly heatmap of PnL

Usage:
    python analyse_streaks_sessions.py --symbol GOLD
    python analyse_streaks_sessions.py --symbol GOLD --ensemble
    python analyse_streaks_sessions.py --symbol GOLD --from 2022-01-14 --to 2026-05-08
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import print as rprint

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TEST_START, TEST_END, SYMBOLS,
    MODEL_DIR,
)

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Session labeller
# ─────────────────────────────────────────────────────────────────────────────
def _session_label(hour: int) -> str:
    if 12 <= hour < 16: return "Overlap (12-16)"
    if  7 <= hour < 12: return "London  (07-12)"
    if 16 <= hour < 22: return "NY      (16-22)"
    if  0 <= hour <  7: return "Asia    (00-07)"
    return                     "Off     (22-00)"


SESSION_ORDER = [
    "Overlap (12-16)",
    "London  (07-12)",
    "NY      (16-22)",
    "Asia    (00-07)",
    "Off     (22-00)",
]

DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]


# ─────────────────────────────────────────────────────────────────────────────
# Streak helpers
# ─────────────────────────────────────────────────────────────────────────────
def _compute_streaks(mask: np.ndarray) -> list:
    """Return list of all streak lengths where mask is True."""
    streaks = []
    cur = 0
    for v in mask:
        if v:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)
    return streaks


def _streak_stats(streaks: list) -> dict:
    if not streaks:
        return {"max": 0, "avg": 0.0, "median": 0.0,
                "p75": 0.0, "p90": 0.0, "count": 0}
    arr = np.array(streaks)
    return {
        "max":    int(arr.max()),
        "avg":    float(arr.mean()),
        "median": float(np.median(arr)),
        "p75":    float(np.percentile(arr, 75)),
        "p90":    float(np.percentile(arr, 90)),
        "count":  int(len(arr)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build trade log by re-running backtest
# ─────────────────────────────────────────────────────────────────────────────
def _get_trade_log(symbol: str, date_from: str, date_to: str,
                   use_ensemble: bool) -> pd.DataFrame:
    """Run backtest and return the raw bar-level log."""
    from backtest import run_backtest

    metrics = run_backtest(
        symbol       = symbol,
        date_from    = date_from,
        date_to      = date_to,
        model_tag    = "final",
        use_ensemble = use_ensemble,
    )
    log = metrics["trade_log"]
    return log


# ─────────────────────────────────────────────────────────────────────────────
# Build trade-level summary (flip-to-flip)
# ─────────────────────────────────────────────────────────────────────────────
def _build_trades(log: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse bar-level log into flip-to-flip trades.
    Each row = one trade with entry time, exit time, PnL, session, DOW.
    """
    flip_idx = [0] + list(np.where(log["flipped"].values)[0]) + [len(log)]
    trades   = []

    for i in range(len(flip_idx) - 1):
        start = flip_idx[i]
        end   = flip_idx[i + 1]
        if start >= end:
            continue

        slice_      = log.iloc[start:end]
        entry_ts    = slice_.index[0]
        exit_ts     = slice_.index[-1]
        trade_pnl   = slice_["pnl_usd"].sum()
        position    = int(slice_["position"].iloc[0])
        n_bars      = len(slice_)

        trades.append({
            "entry_time":  entry_ts,
            "exit_time":   exit_ts,
            "pnl_usd":     trade_pnl,
            "position":    position,
            "n_bars":      n_bars,
            "won":         trade_pnl > 0,
            "session":     _session_label(entry_ts.hour),
            "dow":         DOW_NAMES[entry_ts.dayofweek],
            "hour":        entry_ts.hour,
        })

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────
def analyse_streaks(trades: pd.DataFrame, symbol: str):
    """Print average and max win/loss streaks at trade level."""
    won  = trades["won"].values
    lost = ~won

    win_streaks  = _compute_streaks(won)
    loss_streaks = _compute_streaks(lost)

    ws = _streak_stats(win_streaks)
    ls = _streak_stats(loss_streaks)

    console.rule(f"[bold cyan]Streak Analysis — {symbol} (Trade Level)")

    table = Table(show_lines=True)
    table.add_column("Metric",   style="cyan",   justify="left")
    table.add_column("Win",      style="green",  justify="right")
    table.add_column("Loss",     style="red",    justify="right")

    table.add_row("Max streak",       str(ws["max"]),           str(ls["max"]))
    table.add_row("Average streak",   f"{ws['avg']:.2f}",       f"{ls['avg']:.2f}")
    table.add_row("Median streak",    f"{ws['median']:.1f}",    f"{ls['median']:.1f}")
    table.add_row("75th percentile",  f"{ws['p75']:.1f}",       f"{ls['p75']:.1f}")
    table.add_row("90th percentile",  f"{ws['p90']:.1f}",       f"{ls['p90']:.1f}")
    table.add_row("Total streaks",    str(ws["count"]),          str(ls["count"]))

    console.print(table)

    # Distribution of streak lengths
    rprint("\n[bold]Win streak distribution:[/bold]")
    if win_streaks:
        for length in sorted(set(win_streaks)):
            count = win_streaks.count(length)
            bar   = "█" * min(count, 40)
            rprint(f"  {length:3d}x  {bar}  ({count})")

    rprint("\n[bold]Loss streak distribution:[/bold]")
    if loss_streaks:
        for length in sorted(set(loss_streaks)):
            count = loss_streaks.count(length)
            bar   = "█" * min(count, 40)
            rprint(f"  {length:3d}x  {bar}  ({count})")


def analyse_sessions(trades: pd.DataFrame, symbol: str):
    """Session-by-session performance breakdown."""
    console.rule(f"[bold cyan]Session Performance — {symbol}")

    table = Table(show_lines=True)
    table.add_column("Session",        style="cyan",    justify="left")
    table.add_column("Trades",         style="white",   justify="right")
    table.add_column("Win Rate",       style="green",   justify="right")
    table.add_column("Avg PnL",        style="yellow",  justify="right")
    table.add_column("Total PnL",      style="yellow",  justify="right")
    table.add_column("Avg Win",        style="green",   justify="right")
    table.add_column("Avg Loss",       style="red",     justify="right")
    table.add_column("Profit Factor",  style="magenta", justify="right")
    table.add_column("Avg Bars Held",  style="white",   justify="right")

    session_stats = []
    for sess in SESSION_ORDER:
        s = trades[trades["session"] == sess]
        if len(s) == 0:
            continue

        n          = len(s)
        win_rate   = s["won"].mean()
        avg_pnl    = s["pnl_usd"].mean()
        total_pnl  = s["pnl_usd"].sum()
        avg_win    = s.loc[s["won"],  "pnl_usd"].mean() if s["won"].any()  else 0.0
        avg_loss   = s.loc[~s["won"], "pnl_usd"].mean() if (~s["won"]).any() else 0.0
        gross_win  = s.loc[s["won"],  "pnl_usd"].sum()
        gross_loss = min(s.loc[~s["won"], "pnl_usd"].sum(), -1e-10)
        pf         = abs(gross_win / gross_loss) if gross_loss != 0 else float("inf")
        avg_bars   = s["n_bars"].mean()

        session_stats.append((sess, n, win_rate, avg_pnl, total_pnl, pf))

        win_col  = "green"  if win_rate >= 0.6 else "yellow" if win_rate >= 0.5 else "red"
        pnl_col  = "green"  if avg_pnl > 0     else "red"
        pf_col   = "green"  if pf >= 1.5        else "yellow" if pf >= 1.0 else "red"

        table.add_row(
            sess,
            str(n),
            f"[{win_col}]{win_rate:.1%}[/{win_col}]",
            f"[{pnl_col}]${avg_pnl:+.2f}[/{pnl_col}]",
            f"[{pnl_col}]${total_pnl:+,.0f}[/{pnl_col}]",
            f"${avg_win:+.2f}",
            f"${avg_loss:+.2f}",
            f"[{pf_col}]{pf:.2f}[/{pf_col}]",
            f"{avg_bars:.1f}",
        )

    console.print(table)

    # Best and worst session
    if session_stats:
        best  = max(session_stats, key=lambda x: x[3])
        worst = min(session_stats, key=lambda x: x[3])
        rprint(f"\n  [green]Best session:[/green]  {best[0]}  avg=${best[3]:+.2f}  win={best[2]:.1%}")
        rprint(f"  [red]Worst session:[/red] {worst[0]}  avg=${worst[3]:+.2f}  win={worst[2]:.1%}")


def analyse_day_of_week(trades: pd.DataFrame, symbol: str):
    """Day-of-week performance breakdown."""
    console.rule(f"[bold cyan]Day of Week Performance — {symbol}")

    table = Table(show_lines=True)
    table.add_column("Day",           style="cyan",   justify="left")
    table.add_column("Trades",        style="white",  justify="right")
    table.add_column("Win Rate",      style="green",  justify="right")
    table.add_column("Avg PnL",       style="yellow", justify="right")
    table.add_column("Total PnL",     style="yellow", justify="right")
    table.add_column("Profit Factor", style="magenta",justify="right")

    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        d = trades[trades["dow"] == day]
        if len(d) == 0:
            continue

        n         = len(d)
        win_rate  = d["won"].mean()
        avg_pnl   = d["pnl_usd"].mean()
        total_pnl = d["pnl_usd"].sum()
        gw        = d.loc[d["won"],  "pnl_usd"].sum()
        gl        = abs(d.loc[~d["won"], "pnl_usd"].sum())
        pf        = gw / (gl + 1e-10)

        win_col = "green" if win_rate >= 0.6 else "yellow" if win_rate >= 0.5 else "red"
        pnl_col = "green" if avg_pnl > 0 else "red"

        table.add_row(
            day,
            str(n),
            f"[{win_col}]{win_rate:.1%}[/{win_col}]",
            f"[{pnl_col}]${avg_pnl:+.2f}[/{pnl_col}]",
            f"[{pnl_col}]${total_pnl:+,.0f}[/{pnl_col}]",
            f"{pf:.2f}",
        )

    console.print(table)


def analyse_hourly(trades: pd.DataFrame, symbol: str):
    """Hourly PnL heatmap."""
    console.rule(f"[bold cyan]Hourly Performance Heatmap — {symbol}")

    hourly = trades.groupby("hour").agg(
        n        = ("pnl_usd", "count"),
        win_rate = ("won",     "mean"),
        avg_pnl  = ("pnl_usd", "mean"),
        total    = ("pnl_usd", "sum"),
    ).reset_index()

    max_abs = hourly["avg_pnl"].abs().max()

    for _, row in hourly.iterrows():
        h        = int(row["hour"])
        sess     = _session_label(h)
        bar_len  = int(abs(row["avg_pnl"]) / max_abs * 30) if max_abs > 0 else 0
        bar      = "█" * bar_len
        sign     = "+" if row["avg_pnl"] >= 0 else "-"
        color    = "green" if row["avg_pnl"] >= 0 else "red"
        rprint(
            f"  [{color}]{h:02d}:00[/{color}]  "
            f"[white]{sess:15s}[/white]  "
            f"n={int(row['n']):4d}  "
            f"wr={row['win_rate']:.0%}  "
            f"avg=[{color}]${row['avg_pnl']:+6.1f}[/{color}]  "
            f"[{color}]{sign}{bar}[/{color}]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Summary recommendation
# ─────────────────────────────────────────────────────────────────────────────
def _recommendations(trades: pd.DataFrame):
    console.rule("[bold yellow]Recommendations")

    session_pnl = trades.groupby("session")["pnl_usd"].mean()

    # Worst session
    worst_sess = session_pnl.idxmin()
    worst_val  = session_pnl.min()
    if worst_val < 0:
        rprint(f"  [red]✗[/red] [bold]{worst_sess}[/bold] is unprofitable "
               f"(avg ${worst_val:.2f}/trade) — "
               f"consider increasing flip penalty for this session in config")

    # Best session
    best_sess = session_pnl.idxmax()
    best_val  = session_pnl.max()
    rprint(f"  [green]✓[/green] [bold]{best_sess}[/bold] is your best session "
           f"(avg ${best_val:.2f}/trade) — "
           f"already rewarded at 1.3x in reward function")

    # Loss streak insight
    loss_streaks = _compute_streaks(~trades["won"].values)
    ls           = _streak_stats(loss_streaks)
    if ls["avg"] > 2.5:
        rprint(f"  [yellow]⚠[/yellow] Average loss streak {ls['avg']:.1f} — "
               f"regime-specific agents would target this directly")
    else:
        rprint(f"  [green]✓[/green] Average loss streak {ls['avg']:.1f} — "
               f"well controlled by ensemble")

    # DOW insight
    dow_pnl  = trades.groupby("dow")["pnl_usd"].mean()
    worst_d  = dow_pnl.idxmin()
    worst_dv = dow_pnl.min()
    if worst_dv < 0:
        rprint(f"  [red]✗[/red] [bold]{worst_d}[/bold] is unprofitable "
               f"(avg ${worst_dv:.2f}/trade) — "
               f"consider adding day-of-week multiplier to reward")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis(symbol: str, date_from: str, date_to: str,
                 use_ensemble: bool):

    rprint(f"""
[bold cyan]╔══════════════════════════════════════════════════════════════╗
║       Streak & Session Analysis  —  {symbol:6s}                  ║
╚══════════════════════════════════════════════════════════════╝[/bold cyan]
""")

    console.rule("Running backtest to collect trade log...")
    log    = _get_trade_log(symbol, date_from, date_to, use_ensemble)
    trades = _build_trades(log)

    rprint(f"\n  [cyan]Total trades:[/cyan]  {len(trades):,}")
    rprint(f"  [cyan]Period:[/cyan]        {date_from} → {date_to}")
    rprint(f"  [cyan]Ensemble:[/cyan]      {use_ensemble}\n")

    analyse_streaks(trades, symbol)
    analyse_sessions(trades, symbol)
    analyse_day_of_week(trades, symbol)
    analyse_hourly(trades, symbol)
    _recommendations(trades)

    # Export to CSV
    out = Path("results") / f"streak_session_{symbol}.csv"
    out.parent.mkdir(exist_ok=True)
    trades.to_csv(out, index=False)
    rprint(f"\n  [dim]Trade log saved → {out}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Streak & Session Performance Analysis"
    )
    parser.add_argument("--symbol",   type=str, default="GOLD")
    parser.add_argument("--from",     dest="date_from", default=TEST_START)
    parser.add_argument("--to",       dest="date_to",   default=TEST_END)
    parser.add_argument("--ensemble", action="store_true",
                        help="Use 5-seed ensemble")
    args = parser.parse_args()

    run_analysis(
        symbol       = args.symbol,
        date_from    = args.date_from,
        date_to      = args.date_to,
        use_ensemble = args.ensemble,
    )
