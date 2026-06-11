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
    symbols:           list,
    timesteps:         int,
    date_from:         str,
    date_to:           str,
    test_from:         str,
    test_to:           str,
    skip_supervised:   bool,
    fetch_mt5_flag:    bool,
    per_symbol_models: bool,
    ensemble:          bool  = True,
    regime_agents:     bool  = False,
    walkforward:       bool  = False,
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
    ens_label = "ensemble (5 seeds)" if ensemble else "single model"
    console.rule(f"[bold]Step 2 · Training RL agents — {ens_label}")
    from training.train_rl import train_all
    results = train_all(
        symbols   = symbols,
        timesteps = timesteps,
        date_from = date_from,   # None = per-symbol override from config
        date_to   = date_to,
        test_from = test_from,
        test_to   = test_to,
        ensemble  = ensemble,
    )

    # ── Step 3: Regime agents (optional) ──────────────────────────────────────
    if regime_agents:
        console.rule("[bold]Step 3 · Training regime-specific agents")
        from training.train_regime_agents import train_all_regime_agents
        for sym in symbols:
            try:
                train_all_regime_agents(symbol=sym)
            except Exception as e:
                logger.error(f"[{sym}] Regime agents failed: {e}", exc_info=True)

    # ── Step 4: Walk-forward (optional) ───────────────────────────────────────
    if walkforward:
        console.rule("[bold]Step 4 · Walk-forward retraining pipeline")
        from training.walkforward import walkforward as run_wf
        for sym in symbols:
            try:
                run_wf(symbol=sym)
            except Exception as e:
                logger.error(f"[{sym}] Walk-forward failed: {e}", exc_info=True)

    # ── Step 3: Summary table ─────────────────────────────────────────────────
    console.rule("[bold]Results Summary")
    table = Table(title="Test Set Performance", show_lines=True)
    table.add_column("Symbol",    style="cyan",    justify="left")
    table.add_column("Mean PnL",  style="green",   justify="right")
    table.add_column("Sharpe",    style="yellow",  justify="right")
    table.add_column("Win Rate",  style="blue",    justify="right")
    table.add_column("Avg Flips", style="white",   justify="right")
    table.add_column("Ensemble",  style="magenta", justify="center")
    table.add_column("Status",    style="bold",    justify="center")

    for sym, m in results.items():
        if "error" in m:
            table.add_row(sym, "–", "–", "–", "–", "–", "[red]ERROR[/red]")
        else:
            pnl    = m.get("mean_pnl", 0)
            sharpe = m.get("mean_sharpe", 0)
            win    = m.get("win_rate", 0)
            flips  = m.get("mean_flips", 0)
            ens_n  = m.get("ensemble_size", 1)
            status = "[green]✓ PASS[/green]" if sharpe > 0.5 else "[yellow]⚠ REVIEW[/yellow]"
            table.add_row(
                sym,
                f"{pnl:+.4f}",
                f"{sharpe:.3f}",
                f"{win:.1%}",
                f"{flips:.0f}",
                str(ens_n),
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
        "--timesteps", type=int, default=None,
        help="RL training steps per symbol (overrides config dict)"
    )
    parser.add_argument(
        "--skip-supervised", action="store_true",
        help="Skip supervised model training (use existing saved models)"
    )
    parser.add_argument(
        "--fetch-mt5", action="store_true",
        help="Fetch data from MT5 before training (Windows only)"
    )
    parser.add_argument(
        "--shared-models", action="store_true",
        help="Train shared supervised models instead of per-symbol models"
    )
    parser.add_argument(
        "--ensemble", action="store_true", default=True,
        help="Train 5 seeds per symbol with majority-vote ensemble (default: on)"
    )
    parser.add_argument(
        "--no-ensemble", action="store_true",
        help="Force single-seed training (override ensemble default)"
    )
    parser.add_argument(
        "--regime-agents", action="store_true",
        help="Also train regime-specific agents after ensemble"
    )
    parser.add_argument(
        "--walkforward", action="store_true",
        help="Run monthly walk-forward retraining after initial training"
    )
    # Date defaults: None so per-symbol overrides in config apply
    parser.add_argument("--from",      dest="date_from",  type=str, default=None)
    parser.add_argument("--to",        dest="date_to",    type=str, default=None)
    parser.add_argument("--test-from", dest="test_from",  type=str, default=None)
    parser.add_argument("--test-to",   dest="test_to",    type=str, default=None)
    args = parser.parse_args()

    ens = not args.no_ensemble   # ensemble on by default, off with --no-ensemble

    run_pipeline(
        symbols           = args.symbols,
        timesteps         = args.timesteps,
        date_from         = args.date_from,
        date_to           = args.date_to,
        test_from         = args.test_from,
        test_to           = args.test_to,
        skip_supervised   = args.skip_supervised,
        fetch_mt5_flag    = args.fetch_mt5,
        per_symbol_models = not args.shared_models,
        ensemble          = ens,
        regime_agents     = args.regime_agents,
        walkforward       = args.walkforward,
    )
