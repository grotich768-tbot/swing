"""
live_utils.py  —  Live trading utilities
──────────────────────────────────────────────────────────────────────────────
Fixes:
  1. Real account balance used for lot sizing (not hardcoded INITIAL_BALANCE)
  2. Lot size scales proportionally with account size
  3. Drop-in replacement for the _lot_size function used in live execution

Usage — in your live_trader.py or wherever lots are computed:

    from live_utils import LiveSizer

    # On startup — fetch real balance once
    sizer = LiveSizer(mt5_bridge)

    # Each bar — compute lots using real balance
    lots = sizer.lot_size(symbol, atr)

    # Refresh balance every N bars (or after each trade)
    sizer.refresh()
"""

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PIP_VALUE, SPREAD_PIPS, MAX_POSITION_PCT,
    INITIAL_BALANCE, SYMBOLS,
    ADAPTIVE_SIZING_ENABLED,
)

# Must match backtest.py constants exactly for consistency
ATR_STOP_MULT = 1.5
MAX_LOTS      = 0.5
MIN_LOTS      = 0.01

# USD value per pip per 1.0 lot — must match backtest.py PIP_USD_PER_LOT
PIP_USD_PER_LOT = {
    # Broker-verified: USD value per pip per 1.0 standard lot
    "GOLD":   0.10,    # was 10.0  ✓ fixed
    "SILVER": 0.50,    # was 50.0  ✓ fixed
    "EURUSD": 0.10,    # was 10.0  ✓ fixed
    "GBPUSD": 0.10,    # was 10.0  ✓ fixed
    "USDJPY": 0.06,    # was 9.09  ✓ fixed
    "BTCUSD": 0.10,    # was 1.0   ✓ fixed
    "ETHUSD": 0.10,    # was 1.0   ✓ fixed
    "US30":   1.00,    # unchanged ✓
    "US100":  1.00,    # unchanged ✓
    "US500":  1.00,    # was 10.0  ✓ fixed
    "UK100":  1.00,    # unchanged ✓
    "AUS200": 1.00,    # unchanged ✓
    "GER40":  1.00,    # unchanged ✓
    "JP225":  1.00,    # verify with broker
}


class LiveSizer:
    """
    ATR-based position sizer that uses the REAL live account balance.

    This replaces the hardcoded INITIAL_BALANCE=$10,000 that caused
    identical lot sizes regardless of account size.

    Parameters
    ----------
    bridge           : MT5Bridge instance (must have .account_balance() method)
    risk_pct         : fraction of balance to risk per trade (default 2%)
    refresh_interval : refresh balance every N calls to lot_size()
    """

    def __init__(
        self,
        bridge,
        risk_pct:         float = MAX_POSITION_PCT,
        refresh_interval: int   = 50,
    ):
        self._bridge           = bridge
        self._risk_pct         = risk_pct
        self._refresh_interval = refresh_interval
        self._call_count       = 0
        self._balance          = self._fetch_balance()
        self._adaptive_mults   = {s: 1.0 for s in SYMBOLS}

        logger.info(
            f"[LiveSizer] Initialised  "
            f"balance=${self._balance:,.2f}  "
            f"risk={self._risk_pct:.1%}  "
            f"max_lots={MAX_LOTS}"
        )

    # ── Public API ────────────────────────────────────────────────────────────
    def lot_size(
        self,
        symbol:        str,
        atr:           float,
        size_mult:     float = 1.0,   # from risk_engine.position_size()
    ) -> float:
        """
        Compute lot size based on real account balance.

        Parameters
        ----------
        symbol    : trading symbol
        atr       : current ATR in price units
        size_mult : multiplier from risk engine (adaptive sizing, session filter)

        Returns
        -------
        float — lot size clipped to [MIN_LOTS, MAX_LOTS]
        """
        self._call_count += 1
        if self._call_count % self._refresh_interval == 0:
            self.refresh()

        pip        = PIP_VALUE.get(symbol, 0.0001)
        pip_usd    = PIP_USD_PER_LOT.get(symbol, 10.0)
        risk_usd   = self._balance * self._risk_pct   # scales with real balance
        atr_stop   = max(atr * ATR_STOP_MULT, pip)
        atr_pips   = atr_stop / pip
        base_lots  = risk_usd / (atr_pips * pip_usd + 1e-10)

        # Apply size multiplier from risk engine (adaptive sizing etc)
        adj_lots   = base_lots * size_mult * self._adaptive_mults.get(symbol, 1.0)

        lots = float(np.clip(adj_lots, MIN_LOTS, MAX_LOTS))

        logger.debug(
            f"[LiveSizer] {symbol}  "
            f"bal=${self._balance:,.0f}  "
            f"risk=${risk_usd:.0f}  "
            f"atr_pips={atr_pips:.1f}  "
            f"base={base_lots:.4f}  "
            f"adj={adj_lots:.4f}  "
            f"final={lots:.3f}"
        )
        return lots

    def refresh(self):
        """Re-fetch account balance from MT5."""
        new_balance = self._fetch_balance()
        if new_balance != self._balance:
            logger.info(
                f"[LiveSizer] Balance updated  "
                f"${self._balance:,.2f} → ${new_balance:,.2f}"
            )
        self._balance = new_balance

    def set_adaptive_mult(self, symbol: str, mult: float):
        """
        Set per-symbol adaptive size multiplier.
        Called by risk_engine after consecutive loss/win streaks.
        """
        self._adaptive_mults[symbol] = float(np.clip(mult, 0.1, 1.2))
        logger.debug(f"[LiveSizer] {symbol} adaptive_mult → {mult:.2f}")

    @property
    def balance(self) -> float:
        return self._balance

    # ── Private ───────────────────────────────────────────────────────────────
    def _fetch_balance(self) -> float:
        """Fetch real account balance from MT5 bridge."""
        try:
            balance = self._bridge.account_balance()
            if balance and balance > 0:
                return float(balance)
        except Exception as e:
            logger.warning(f"[LiveSizer] Could not fetch balance: {e}")
        # Fallback to config default — log a warning
        logger.warning(
            f"[LiveSizer] Using fallback INITIAL_BALANCE=${INITIAL_BALANCE:,.2f} "
            f"— check MT5 connection"
        )
        return float(INITIAL_BALANCE)

    def sizing_summary(self) -> dict:
        """Return current sizing state for diagnostics."""
        return {
            "live_balance":    self._balance,
            "config_balance":  INITIAL_BALANCE,
            "difference_pct":  (self._balance - INITIAL_BALANCE) / INITIAL_BALANCE,
            "risk_pct":        self._risk_pct,
            "adaptive_mults":  dict(self._adaptive_mults),
            "refresh_every":   self._refresh_interval,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Patch function — drop-in replacement for live_trader.py
# ─────────────────────────────────────────────────────────────────────────────
def patch_live_trader_sizing(live_trader_instance, mt5_bridge):
    """
    Monkey-patch a LiveTrader instance to use real account balance for sizing.

    Call this right after LiveTrader is instantiated in live_run.py:

        from live_utils import patch_live_trader_sizing
        trader = LiveTrader(settings, ui=ui)
        patch_live_trader_sizing(trader, trader._bridge)

    This replaces the internal _compute_lots method with LiveSizer.
    """
    sizer = LiveSizer(mt5_bridge)

    def _patched_lot_size(symbol, atr, size_mult=1.0):
        return sizer.lot_size(symbol, atr, size_mult)

    live_trader_instance._live_sizer   = sizer
    live_trader_instance._compute_lots = _patched_lot_size

    logger.info(
        f"[LiveSizer] Patched LiveTrader sizing  "
        f"live_balance=${sizer.balance:,.2f}  "
        f"(was hardcoded ${INITIAL_BALANCE:,.2f})"
    )
    return sizer


# ─────────────────────────────────────────────────────────────────────────────
# Standalone lot size calculator — for checking what lots will be
# ─────────────────────────────────────────────────────────────────────────────
def preview_lot_sizes(
    account_balance: float,
    atr_estimates:   dict = None,
    risk_pct:        float = MAX_POSITION_PCT,
):
    """
    Print a table of lot sizes for a given account balance.
    Useful for verifying sizing before going live.

    Usage:
        python live_utils.py --balance 50000
    """
    from rich.table import Table
    from rich.console import Console
    from rich import print as rprint

    # Default ATR estimates (approximate current values)
    if atr_estimates is None:
        atr_estimates = {
            "GOLD":   15.0,   # ~$15 ATR on H1
            "SILVER": 0.25,
            "EURUSD": 0.0008,
            "GBPUSD": 0.0010,
        }

    console = Console()
    rprint(f"\n[bold cyan]Lot Size Preview  —  Balance: ${account_balance:,.2f}  Risk: {risk_pct:.1%}[/bold cyan]\n")

    table = Table(show_lines=True)
    table.add_column("Symbol",   style="cyan")
    table.add_column("ATR",      style="white",  justify="right")
    table.add_column("Risk $",   style="yellow", justify="right")
    table.add_column("Lots",     style="green",  justify="right")
    table.add_column("vs $10k",  style="magenta",justify="right")

    for sym, atr in atr_estimates.items():
        pip      = PIP_VALUE.get(sym, 0.0001)
        pip_usd  = PIP_USD_PER_LOT.get(sym, 10.0)
        risk_usd = account_balance * risk_pct
        atr_pips = (atr * ATR_STOP_MULT) / pip
        lots     = float(np.clip(risk_usd / (atr_pips * pip_usd + 1e-10), MIN_LOTS, MAX_LOTS))

        # Compare vs $10k hardcoded
        risk_10k  = INITIAL_BALANCE * risk_pct
        lots_10k  = float(np.clip(risk_10k / (atr_pips * pip_usd + 1e-10), MIN_LOTS, MAX_LOTS))
        diff      = (lots - lots_10k) / (lots_10k + 1e-10)

        diff_str = f"{diff:+.0%}"
        diff_col = "green" if diff > 0 else "red" if diff < 0 else "white"

        table.add_row(
            sym,
            f"{atr:.4f}",
            f"${risk_usd:.0f}",
            f"{lots:.3f}",
            f"[{diff_col}]{diff_str}[/{diff_col}]",
        )

    console.print(table)
    rprint(f"[dim]Hardcoded $10k lots shown for comparison in 'vs $10k' column[/dim]\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preview live lot sizes")
    parser.add_argument("--balance", type=float, default=10_000,
                        help="Your actual account balance in USD")
    parser.add_argument("--risk",    type=float, default=MAX_POSITION_PCT,
                        help="Risk per trade as decimal (default 0.02 = 2%%)")
    args = parser.parse_args()
    preview_lot_sizes(args.balance, risk_pct=args.risk)
