"""
live/risk_guard.py  —  Live risk management layer
──────────────────────────────────────────────────────────────────────────────
All risk checks run BEFORE any order is sent to MT5.
The RL agent proposes an action — the risk guard approves or vetoes it.

Rules (in priority order):
  1. Daily loss limit    — stop all trading for today if exceeded
  2. Circuit breaker     — freeze flips per symbol on excessive drawdown
  3. Spread filter       — skip trade if spread is too wide
  4. Rollover guard      — block metal flips near daily close
  5. Session filter      — reduce size outside main session
  6. Correlation cap     — limit combined exposure on correlated pairs
"""

import sys
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PIP_VALUE
from live.settings import LiveSettings
from live.mt5_bridge import MT5Bridge

# Correlation groups — same macro driver
CORR_GROUPS = [
    {"GOLD", "SILVER"},
    {"EURUSD", "GBPUSD"},
]


class RiskGuard:
    """
    Stateful risk layer that sits between the RL agent and order execution.

    Call check_action() before every potential flip.
    Call update() after every bar to refresh state.
    """

    def __init__(self, bridge: MT5Bridge, settings: LiveSettings):
        self.bridge   = bridge
        self.s        = settings

        # Per-symbol state
        self._frozen:       Dict[str, bool]  = {s: False for s in settings.active_symbols}
        self._peak_equity:  Dict[str, float] = {}
        self._size_mult:    Dict[str, float] = {s: 1.0   for s in settings.active_symbols}

        # Daily loss tracking (equity-based to include floating P/L)
        self._daily_start_balance: float = 0.0
        self._daily_start_equity:   float = 0.0
        self._daily_halt:           bool  = False
        self._last_daily_reset:     date  = date.min

    # ── Primary gate — call before every potential flip ────────────────────────
    def check_action(
        self,
        symbol:    str,
        action:    int,          # 0 = HOLD, 1 = FLIP
        timestamp: Optional[datetime] = None,
    ) -> int:
        """
        Returns the (possibly overridden) action.
        Always returns 0 (HOLD) if any risk rule vetoes the flip.
        """
        if action == 0:
            return 0   # HOLDs never need risk checks

        ts = timestamp or datetime.now(tz=timezone.utc)

        # Rule 1 — daily halt
        if self._daily_halt:
            logger.warning(f"[RISK] {symbol}: FLIP blocked — daily loss limit reached")
            return 0

        # Rule 2 — circuit breaker
        if self._frozen.get(symbol, False):
            logger.warning(f"[RISK] {symbol}: FLIP blocked — circuit breaker active")
            return 0

        # Rule 3 — spread check
        spread = self.bridge.get_current_spread_pips(symbol)
        max_spread = self.s.max_spread_pips(symbol)
        if spread > max_spread:
            logger.warning(
                f"[RISK] {symbol}: FLIP blocked — spread {spread:.1f} pips "
                f"> max {max_spread:.1f} pips"
            )
            return 0

        # Rule 4 — rollover guard (metals only)
        if self.s.rollover_guard_enabled and symbol in ("GOLD", "SILVER"):
            if self._in_rollover_window(ts):
                logger.debug(f"[RISK] {symbol}: FLIP blocked — rollover window")
                return 0

        return 1   # Approved

    def get_size_multiplier(self, symbol: str) -> float:
        """Return position size multiplier for a symbol (0–1]."""
        if self._daily_halt:
            return 0.0
        return self._size_mult.get(symbol, 1.0)

    def get_status(self) -> dict:
        """Return a compact snapshot for dashboards and monitoring."""
        balance = self.bridge.account_balance()
        equity = self.bridge.account_equity()
        daily_loss = max(0.0, self._daily_start_equity - equity) if self._daily_start_equity else 0.0
        return {
            "balance": balance,
            "equity": equity,
            "daily_loss_usd": daily_loss,
            "daily_halt": self._daily_halt,
            "daily_start_balance": self._daily_start_balance,
            "daily_start_equity": self._daily_start_equity,
            "frozen_symbols": [sym for sym, frozen in self._frozen.items() if frozen],
            "size_multipliers": dict(self._size_mult),
        }

    # ── Update state after each bar ───────────────────────────────────────────
    def update(self, timestamp: Optional[datetime] = None):
        """
        Refresh all risk state. Call once per H1 bar before acting.
        """
        ts = timestamp or datetime.now(tz=timezone.utc)
        self._reset_daily_if_needed(ts)

        balance = self.bridge.account_balance()
        equity  = self.bridge.account_equity()

        # Track daily loss using equity so floating drawdown is protected too
        daily_loss = self._daily_start_equity - equity
        if daily_loss > self.s.max_daily_loss_usd and not self._daily_halt:
            self._daily_halt = True
            logger.error(
                f"[RISK] Daily loss limit hit: ${daily_loss:.2f} "
                f"(limit: ${self.s.max_daily_loss_usd:.2f}). "
                f"All trading halted for today."
            )

        # Per-symbol circuit breaker on equity drawdown
        for sym in self.s.active_symbols:
            if sym not in self._peak_equity:
                self._peak_equity[sym] = equity

            self._peak_equity[sym] = max(self._peak_equity[sym], equity)
            dd = (self._peak_equity[sym] - equity) / max(self._peak_equity[sym], 1.0)

            if not self._frozen[sym] and dd > self.s.max_drawdown_pct:
                self._frozen[sym] = True
                logger.warning(
                    f"[RISK] {sym} circuit breaker OPEN — "
                    f"drawdown={dd:.2%}  equity=${equity:,.2f}"
                )
            elif self._frozen[sym] and dd < self.s.circuit_breaker_recovery_pct:
                self._frozen[sym] = False
                logger.info(f"[RISK] {sym} circuit breaker CLOSED — drawdown={dd:.2%}")

        # Session size multiplier
        session_mult = self._session_multiplier(ts)

        # Correlation cap
        for sym in self.s.active_symbols:
            mult = session_mult
            # Check if in a correlated group with another active symbol
            for group in CORR_GROUPS:
                if sym in group:
                    active_in_group = [s for s in group if s in self.s.active_symbols
                                       and not self._frozen.get(s, True)]
                    if len(active_in_group) > 1:
                        mult *= 0.6   # reduce all group members by 40%
            self._size_mult[sym] = mult

    def status(self) -> dict:
        """Return current risk state for logging."""
        return {
            "daily_halt":  self._daily_halt,
            "frozen":      dict(self._frozen),
            "size_mult":   dict(self._size_mult),
            "peak_equity": {k: f"${v:,.2f}" for k, v in self._peak_equity.items()},
        }

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _reset_daily_if_needed(self, ts: datetime):
        today = ts.date()
        if today != self._last_daily_reset:
            self._last_daily_reset     = today
            self._daily_start_balance  = self.bridge.account_balance()
            self._daily_start_equity    = self.bridge.account_equity()
            self._daily_halt            = False
            logger.info(
                f"[RISK] Daily reset — balance=${self._daily_start_balance:,.2f} "
                f"equity=${self._daily_start_equity:,.2f}"
            )

    def _session_multiplier(self, ts: datetime) -> float:
        if not self.s.session_filter_enabled:
            return 1.0
        hour = ts.hour
        if self.s.session_full_size_start <= hour < self.s.session_full_size_end:
            return 1.0
        return self.s.session_reduced_multiplier

    def _in_rollover_window(self, ts: datetime) -> bool:
        h, m = ts.hour, ts.minute
        start = (self.s.rollover_start_hour, self.s.rollover_start_min)
        end   = (self.s.rollover_end_hour,   self.s.rollover_end_min)
        t     = (h, m)
        return start <= t < end
