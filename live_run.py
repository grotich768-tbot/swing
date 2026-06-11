"""
live_run.py  —  Entry point for the Always-In Live Trading Bot
──────────────────────────────────────────────────────────────────────────────
Requirements before running:
  1. Copy .env.example → .env  and fill in your settings
  2. Make sure MT5 is open and logged in
  3. Trained PPO models exist in models/saved/
  4. Run on DEMO first:  set TRADING_MODE=DEMO in .env

Usage:
    python live_run.py                    # uses .env settings
    python live_run.py --check            # validate .env without trading
    python live_run.py --status           # show current positions & risk state
"""

import sys
import argparse
import platform
import pathlib
from pathlib import Path

# Force UTF-8 encoding for Windows console to prevent emoji/checkmark crash
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Windows model-loading compatibility fix
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

from loguru import logger
from rich import print as rprint
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

console = Console()


def setup_logging(level: str, to_file: bool, rotation: str, ui=None):
    """Configure loguru logging.

    When the terminal dashboard is enabled, logs are routed into the UI so the
    CMD window stays clean and the dashboard becomes the primary interface.
    """
    logger.remove()

    if ui is None or not getattr(ui, "s", None) or not getattr(ui.s, "terminal_ui_enabled", True):
        logger.add(
            sys.stdout,
            level     = level,
            format    = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                        "<level>{level:<8}</level> | "
                        "<cyan>{name}:{function}:{line}</cyan> — {message}",
            colorize  = True,
        )
    else:
        def _ui_sink(message):
            record = message.record
            level_name = record["level"].name
            ui.emit(level_name, record["message"], ts=record["time"])

        logger.add(_ui_sink, level=level)

    if to_file:
        log_path = Path("logs") / "live_trading.log"
        log_path.parent.mkdir(exist_ok=True)
        logger.add(
            str(log_path),
            level    = level,
            rotation = rotation,
            retention= "1 month",
            format   = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
        )
        logger.info(f"Logging to {log_path}")


def cmd_check(settings):
    """Validate configuration and connectivity without trading."""
    rprint("\n[bold cyan]── Configuration Check ──────────────────────────────[/bold cyan]")

    # Print resolved settings
    table = Table(show_header=False, box=None)
    table.add_column("Key",   style="cyan",  width=30)
    table.add_column("Value", style="white")

    table.add_row("Trading mode",      f"[{'red' if settings.is_live else 'yellow'}]{settings.trading_mode}[/{'red' if settings.is_live else 'yellow'}]")
    table.add_row("Active symbols",    ", ".join(settings.active_symbols))
    table.add_row("Initial balance",   f"${settings.initial_balance:,.2f}")
    table.add_row("Risk per trade",    f"{settings.risk_pct:.1%}")
    table.add_row("Max lots",          str(settings.max_lots))
    table.add_row("Magic number",      str(settings.magic_number))
    table.add_row("Session filter",    str(settings.session_filter_enabled))
    table.add_row("Rollover guard",    str(settings.rollover_guard_enabled))
    table.add_row("Max drawdown",      f"{settings.max_drawdown_pct:.1%}")
    table.add_row("Max daily loss",    f"${settings.max_daily_loss_usd:,.2f}")
    table.add_row("Telegram",         "enabled" if settings.telegram_enabled else "disabled")

    console.print(table)

    rprint("\n[bold cyan]── Symbol Mapping ───────────────────────────────────[/bold cyan]")
    for sym in settings.active_symbols:
        explicit = settings.symbol_name_map.get(sym, sym)
        cands    = settings.symbol_candidates_map.get(sym, [])
        rprint(f"  [cyan]{sym:8s}[/cyan]  explicit=[white]{explicit}[/white]  candidates={cands}")

    rprint("\n[bold cyan]── MT5 Connection ───────────────────────────────────[/bold cyan]")
    try:
        from live.mt5_bridge import MT5Bridge
        bridge = MT5Bridge(settings)
        if bridge.connect():
            resolved = bridge.resolve_all()
            rprint(f"  [green]✓ Connected[/green]")
            for logical, broker in resolved.items():
                spread = bridge.get_current_spread_pips(logical)
                rprint(f"    [cyan]{logical:8s}[/cyan] → [white]{broker}[/white]  spread={spread:.1f} pips")
            balance = bridge.account_balance()
            rprint(f"  Balance: ${balance:,.2f}")
            bridge.disconnect()
        else:
            rprint("  [red]✗ Connection failed[/red]")
    except Exception as e:
        rprint(f"  [red]✗ {e}[/red]")

    rprint("\n[bold cyan]── Saved Models ─────────────────────────────────────[/bold cyan]")
    from config import MODEL_DIR
    for sym in settings.active_symbols:
        final = MODEL_DIR / f"ppo_{sym}_final.zip"
        best  = MODEL_DIR / f"ppo_{sym}" / "best_model.zip"
        if final.exists():
            rprint(f"  [green]✓[/green] [cyan]{sym:8s}[/cyan] {final.name}")
        elif best.exists():
            rprint(f"  [yellow]~[/yellow] [cyan]{sym:8s}[/cyan] {best.name}  (best, not final)")
        else:
            rprint(f"  [red]✗[/red] [cyan]{sym:8s}[/cyan] No model found — run: python train.py")

    rprint()


def cmd_status(settings):
    """Show current live positions and risk state."""
    try:
        from live.mt5_bridge import MT5Bridge
        from live.risk_guard import RiskGuard

        bridge = MT5Bridge(settings)
        if not bridge.connect():
            rprint("[red]Cannot connect to MT5[/red]")
            return

        rprint(f"\n[bold cyan]── Live Positions (magic={settings.magic_number}) ──[/bold cyan]")
        positions = bridge.get_positions()
        if positions:
            for sym, pos in positions.items():
                side = "[green]LONG  ▲[/green]" if pos["side"] == 1 else "[red]SHORT ▼[/red]"
                rprint(
                    f"  [cyan]{sym:8s}[/cyan]  {side}  "
                    f"{pos['lots']:.3f} lots  "
                    f"profit=${pos['profit']:+,.2f}  "
                    f"ticket={pos['ticket']}"
                )
        else:
            rprint("  No open positions from this bot.")

        balance = bridge.account_balance()
        equity  = bridge.account_equity()
        rprint(f"\n  Balance : ${balance:,.2f}")
        rprint(f"  Equity  : ${equity:,.2f}")

        bridge.disconnect()
    except Exception as e:
        rprint(f"[red]Status error: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(
        description="Always-In Trading Bot — Live Runner"
    )
    parser.add_argument(
        "--terminal", type=int, default=1,
        help="Terminal ID to run (1 or 2) for specific .env suffix"
    )
    parser.add_argument(
        "--check",  action="store_true",
        help="Validate .env config and MT5 connectivity without trading"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current live positions and exit"
    )
    args, _ = parser.parse_known_args()

    # Load settings
    from live.settings import load_settings
    settings = load_settings(terminal_id=args.terminal)

    ui = None
    if getattr(settings, "terminal_ui_enabled", True):
        from live.terminal_ui import TerminalUI
        ui = TerminalUI(settings)
        # We DO NOT call ui.start() here because ui.run() blocks the main thread.

    setup_logging(settings.log_level, settings.log_to_file, settings.log_rotation, ui=ui)

    if ui is not None:
        from rich.panel import Panel
        console.print(
            Panel(
                "Live dashboard is active. The terminal will now show a compact trading view.",
                title="Always-In RL/ML Trading Bot",
                border_style="cyan",
            )
        )
    else:
        rprint("""
[bold cyan]╔══════════════════════════════════════════════════════╗
║          Always-In RL/ML Trading Bot                 ║
║          Live Execution Engine                       ║
╚══════════════════════════════════════════════════════╝[/bold cyan]
""")

    if args.check:
        cmd_check(settings)
        if ui:
            ui.stop()
        return

    if args.status:
        cmd_status(settings)
        if ui:
            ui.stop()
        return

    # ── Start trading ─────────────────────────────────────────────────────────
    if settings.is_live:
        rprint("[bold red]  TRADING_MODE=LIVE — real money will be used.[/bold red]\n")
    else:
        rprint("[yellow]Running in DEMO mode — no real money at risk.[/yellow]\n")

    # ── Load symbol specs from MT5 before starting trader ────────────────────
    # Reads live pip_usd, spread, pip_size from MT5 terminal.
    # This is the single source of truth — no hardcoded values needed.
    try:
        from live.symbol_specs import get_specs, patch_live_trader_pip_usd
        import live.live_trader as _lt_module
        sym_specs = get_specs(symbols=settings.active_symbols, settings=settings)
        sym_specs.apply_to_config()          # -> config.SPREAD_PIPS + PIP_VALUE
        sym_specs.print_table()
        patch_live_trader_pip_usd(_lt_module) # → live_trader.PIP_USD_PER_LOT
        rprint("[green]✓ Symbol specs from MT5 applied[/green]")
    except Exception as _se:
        rprint(f"[yellow]⚠ MT5 symbol specs failed: {_se} — using broker_config.py fallback[/yellow]")

    from live.live_trader import LiveTrader
    trader = LiveTrader(settings, ui=ui)
    
    if ui is not None:
        import threading
        
        # Run trader in background thread
        trader_thread = threading.Thread(target=trader.start, daemon=True)
        trader_thread.start()
        
        try:
            # Run Textual UI on main thread (blocks until exit)
            ui.run()
        finally:
            trader.stop()
            trader_thread.join(timeout=5)
    else:
        try:
            trader.start()
        finally:
            pass


if __name__ == "__main__":
    main()
