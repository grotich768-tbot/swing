"""
live/terminal_ui.py — Textual interactive dashboard for the live trading bot.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional
import sys

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Header, Footer, Static, DataTable, RichLog
from rich.text import Text

CSS = """
Screen {
    layout: vertical;
}

#top_panels {
    height: 10;
    margin: 1 1;
}

#metrics_box {
    width: 1fr;
    border: round $accent;
    content-align: center middle;
    background: $surface;
}

#system_box {
    width: 30;
    border: round $secondary;
    background: $surface;
}

#positions_table {
    border: round $success;
    height: 10;
    margin: 0 1;
}

#log_view {
    border: round $warning;
    height: 1fr;
    margin: 1 1;
}
"""

class TerminalUI(App):
    """Interactive Textual dashboard."""

    CSS = CSS
    TITLE = "SWING Z+ Live Trader"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, settings, **kwargs):
        super().__init__(**kwargs)
        self.s = settings
        self._snapshot = {}
        self._started_at = datetime.now(timezone.utc)
        
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Horizontal(id="top_panels"):
            yield Static("Loading metrics...", id="metrics_box")
            yield Static("Loading system...", id="system_box")
            
        yield DataTable(id="positions_table")
        yield RichLog(id="log_view", wrap=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#positions_table", DataTable)
        table.add_columns("Symbol", "Side", "Lots", "Spread", "PnL", "Ticket")
        self.update_timer = self.set_interval(
            max(0.25, float(getattr(self.s, "terminal_ui_refresh_seconds", 1.0))), 
            self.refresh_ui
        )

    def refresh_ui(self) -> None:
        snap = self._snapshot

        # Update metrics box
        try:
            metrics = self.query_one("#metrics_box", Static)
        except Exception:
            return
            
        balance = snap.get("balance", 0.0)
        equity = snap.get("equity", 0.0)
        drawdown = snap.get("drawdown_pct", 0.0)
        daily_loss = snap.get("daily_loss_usd", 0.0)
        open_trades = snap.get("open_trades", 0)
        exposure = snap.get("exposure_lots", 0.0)
        risk_ready = snap.get("risk_ready", True)

        try:
            unrealized = float(equity) - float(balance)
        except Exception:
            unrealized = 0.0

        pnl_color = "green" if unrealized >= 0 else "red"
        pnl_sign = "+" if unrealized >= 0 else ""

        m_text = (
            f"[bold]Account HUD[/bold]\n\n"
            f"[dim]🏦 Balance:[/dim] [cyan]${balance:,.2f}[/cyan]    "
            f"[dim]📈 Equity:[/dim] [cyan]${equity:,.2f}[/cyan]    "
            f"[dim]💸 PnL:[/dim] [{pnl_color}]{pnl_sign}${unrealized:,.2f}[/]\n\n"
            f"[dim]📉 Drawdown:[/dim] [yellow]{drawdown:.2%}[/yellow]    "
            f"[dim]📦 Trades:[/dim] [magenta]{open_trades}[/magenta]    "
            f"[dim]⚖️ Exposure:[/dim] {exposure:.2f} lots\n\n"
            f"[dim]⚠️ Daily Loss:[/dim] [red]${daily_loss:,.2f}[/red]    "
            f"[dim]⚙️ Risk Engine:[/dim] {'[green]ACTIVE[/]' if risk_ready else '[yellow]STANDBY[/]'}"
        )
        metrics.update(m_text)

        # Update System Box
        sys_box = self.query_one("#system_box", Static)
        mode = snap.get("mode", getattr(self.s, "trading_mode", "DEMO"))
        conn = snap.get("connection", "DISCONNECTED")
        cb = snap.get("circuit_breaker_active", False)
        halt = snap.get("daily_halt", False)
        
        s_text = (
            f"[bold]System Status[/bold]\n\n"
            f"[dim]Mode:[/dim] [yellow]{mode}[/]\n"
            f"[dim]MT5 :[/dim] [{'green' if conn == 'CONNECTED' else 'red'}]{conn}[/]\n"
            f"[dim]CB Active:[/dim] [{'red' if cb else 'dim'}]YES[/]\n"
            f"[dim]Daily Halt:[/dim] [{'red' if halt else 'dim'}]YES[/]\n"
            f"[dim]Reconnects:[/dim] {snap.get('reconnects', 0)}\n"
            f"[dim]Models:[/dim] {snap.get('models_loaded', 0)}/{snap.get('models_expected', 0)}\n"
        )
        sys_box.update(s_text)

        # Update DataTable
        table = self.query_one("#positions_table", DataTable)
        positions = snap.get("positions", [])
        
        # Clear and repopulate cleanly
        table.clear()
        
        if not positions:
            pass
        else:
            for pos in positions:
                side_val = pos.get("side", 0)
                side_text = "⬆ LONG" if side_val == 1 else ("⬇ SHORT" if side_val == -1 else "—")
                side_style = "bold green" if side_val == 1 else "bold red"
                
                pnl = pos.get("profit", 0.0)
                p_color = "green" if pnl >= 0 else "red"
                p_sign = "+" if pnl >= 0 else ""
                
                table.add_row(
                    str(pos.get("symbol", "—")),
                    Text(side_text, style=side_style),
                    f"{float(pos.get('lots', 0.0)):.2f}",
                    f"{pos.get('spread', 0.0):.1f}",
                    Text(f"{p_sign}${pnl:,.2f}", style=p_color),
                    str(pos.get("ticket", "—"))
                )

    def start(self):
        # We don't call run() here because Textual blocks the main thread.
        # This keeps compatibility with code that previously called ui.start().
        # Actually, in live_run.py, we will call ui.run() directly on the main thread!
        pass

    def stop(self):
        try:
            self.exit()
        except Exception:
            pass

    def emit(self, level: str, message: str, ts: Optional[datetime] = None):
        if not getattr(self.s, "terminal_ui_enabled", True):
            return
        
        ts = ts or datetime.now(timezone.utc)
        ts_str = ts.strftime("%H:%M:%S")
        
        color_map = {
            "DEBUG": "dim",
            "INFO": "cyan",
            "SUCCESS": "bold green",
            "WARNING": "bold yellow",
            "ERROR": "bold red",
            "CRITICAL": "bold white on red",
        }
        color = color_map.get(level, "white")
        
        # Schedule the log write to happen on the main thread loop
        def _write():
            try:
                log = self.query_one("#log_view", RichLog)
                log.write(Text.assemble(
                    (f"[{ts_str}] ", "dim"),
                    (f"{level:<8} ", color),
                    (str(message), color if level in ("WARNING", "ERROR", "CRITICAL") else "white")
                ))
            except Exception:
                pass
                
        self.call_from_thread(_write)

    def update(self, snapshot: Dict):
        self._snapshot = dict(snapshot or {})
