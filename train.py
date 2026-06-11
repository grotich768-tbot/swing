"""
train.py  —  Main entry point for Always-In Trading Bot
──────────────────────────────────────────────────────────────────────────────
Orchestrates the full pipeline:
  Step 1  →  Fetch data (MT5 or yfinance)
  Step 2  →  Train supervised models (regime classifier + LSTM)
  Step 3  →  Train RL agents (PPO, one per symbol)
  Step 4  →  Evaluate and report

Usage
─────
  # Full pipeline (will use yfinance if no Parquet files found)
  python train.py

  # Skip supervised training if models already saved
  python train.py --skip-supervised

  # Single symbol
  python train.py --symbols GOLD SILVER

  # Quick smoke-test (fewer timesteps)
  python train.py --timesteps 50000

  # Fetch from MT5 first (Windows only)
  python train.py --fetch-mt5
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import print as rprint

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SYMBOLS, TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    TOTAL_TIMESTEPS, LOG_DIR,
)

console = Console()


def banner():
    rprint("""
[bold cyan]╔══════════════════════════════════════════════════════╗
║          Always-In RL/ML Trading Bot                 ║
║   GOLD · SILVER · EURUSD · GBPUSD                   ║
╚══════════════════════════════════════════════════════╝[/bold cyan]
""")


def fetch_mt5(symbols, date_from, date_to):
    """Optional MT5 data fetch (Windows only)."""
    try:
        from data.mt5_fetcher import MT5Fetcher
        fetcher = MT5Fetcher()
        fetcher.fetch_all(symbols=symbols, date_from=date_from, date_to=date_to)
    except EnvironmentError as e:
        logger.error(f"MT5 fetch failed (expected on Linux/Codespaces): {e}")
        logger.info("Continuing with yfinance fallback...")


def run_pipeline(
    symbols:          list,
    timesteps:        int,
    date_from:        str,
    date_to:          str,
    test_from:        str,
    test_to:          str,
    skip_supervised:  bool,
    fetch_mt5_flag:   bool,
    per_symbol_models: bool,
):
    start = datetime.now()
    banner()

    # ── Step 0: MT5 fetch (optional) ──────────────────────────────────────────
    if fetch_mt5_flag:
        console.rule("[bold]Step 0 · Fetching data from MT5")
        fetch_mt5(symbols, date_from, date_to)

    # ── Step 1: Supervised training ───────────────────────────────────────────
    if not skip_supervised:
        console.rule("[bold]Step 1 · Training supervised models")
        from training.train_supervised import train_supervised
        train_supervised(
            symbols   = symbols,
            date_from = date_from,
            date_to   = date_to,
            shared    = not per_symbol_models,
        )
    else:
        console.rule("[dim]Step 1 · Skipping supervised training (--skip-supervised)")

    # ── Step 2: RL training ───────────────────────────────────────────────────
    console.rule("[bold]Step 2 · Training RL agents (PPO)")
    from training.train_rl import train_all
    results = train_all(
        symbols   = symbols,
        timesteps = timesteps,
        date_from = date_from,
        date_to   = date_to,
        test_from = test_from,
        test_to   = test_to,
    )

    # ── Step 3: Summary table ─────────────────────────────────────────────────
    console.rule("[bold]Results Summary")
    table = Table(title="Test Set Performance", show_lines=True)
    table.add_column("Symbol",    style="cyan",   justify="left")
    table.add_column("Mean PnL",  style="green",  justify="right")
    table.add_column("Sharpe",    style="yellow", justify="right")
    table.add_column("Win Rate",  style="blue",   justify="right")
    table.add_column("Avg Flips", style="white",  justify="right")
    table.add_column("Status",    style="bold",   justify="center")

    for sym, m in results.items():
        if "error" in m:
            table.add_row(sym, "–", "–", "–", "–", "[red]ERROR[/red]")
        else:
            pnl    = m.get("mean_pnl", 0)
            sharpe = m.get("mean_sharpe", 0)
            win    = m.get("win_rate", 0)
            flips  = m.get("mean_flips", 0)
            status = "[green]✓ PASS[/green]" if sharpe > 0.5 else "[yellow]⚠ REVIEW[/yellow]"
            table.add_row(
                sym,
                f"{pnl:+.4f}",
                f"{sharpe:.3f}",
                f"{win:.1%}",
                f"{flips:.0f}",
                status,
            )

    console.print(table)

    elapsed = (datetime.now() - start).total_seconds()
    logger.success(f"Full pipeline completed in {elapsed/60:.1f} minutes.")

    # Save results
    results_path = LOG_DIR / "final_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved → {results_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Always-In RL/ML Trading Bot — Full Training Pipeline"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=SYMBOLS,
        help=f"Symbols to train (default: {SYMBOLS})"
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"RL training steps per symbol (default: {TOTAL_TIMESTEPS:,})"
    )
    parser.add_argument("--from",      dest="date_from",  type=str, default=TRAIN_START)
    parser.add_argument("--to",        dest="date_to",    type=str, default=TRAIN_END)
    parser.add_argument("--test-from", dest="test_from",  type=str, default=TEST_START)
    parser.add_argument("--test-to",   dest="test_to",    type=str, default=TEST_END)
    parser.add_argument(
        "--skip-supervised", action="store_true",
        help="Skip supervised model training (use existing saved models)"
    )
    parser.add_argument(
        "--fetch-mt5", action="store_true",
        help="Fetch data from MT5 before training (Windows only)"
    )
    parser.add_argument(
        "--per-symbol-models", action="store_true",
        help="Train separate supervised models per symbol (default: shared)"
    )
    args = parser.parse_args()

    run_pipeline(
        symbols           = args.symbols,
        timesteps         = args.timesteps,
        date_from         = args.date_from,
        date_to           = args.date_to,
        test_from         = args.test_from,
        test_to           = args.test_to,
        skip_supervised   = args.skip_supervised,
        fetch_mt5_flag    = args.fetch_mt5,
        per_symbol_models = args.per_symbol_models,
    )
