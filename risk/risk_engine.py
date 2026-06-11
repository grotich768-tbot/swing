"""
risk/risk_engine.py  —  Portfolio-level risk management
──────────────────────────────────────────────────────────────────────────────
Sits between the RL agents and the execution layer.
Override agent decisions when risk limits are breached.

Rules applied (in priority order)
──────────────────────────────────
1. Circuit breaker  — if any symbol's DD > MAX_DRAWDOWN_PCT,
                      freeze all flips on that symbol until DD recovers.
2. Correlated cap   — GOLD + SILVER are correlated; cap combined notional.
3. Session filter   — reduce size in thin sessions.
4. Rollover guard   — flatten metals 30 min before daily close.
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SYMBOLS, MAX_DRAWDOWN_PCT, CIRCUIT_BREAKER_PCT,
    MAX_CORR_EXPOSURE_PCT, SESSION_SIZE_REDUCTION,
    INITIAL_BALANCE,
)

# GOLD and SILVER share the same primary driver (USD strength / real yields).
# EURUSD and GBPUSD are both inversely correlated with DXY.
# Cap combined exposure within each group.
CORR_GROUPS = [
    {"GOLD", "SILVER"},
    {"EURUSD", "GBPUSD"},
]


class RiskEngine:
    """
    Stateful risk engine; update() must be called every bar.

    Parameters
    ----------
    initial_balance : float
    """

    def __init__(self, initial_balance: float = INITIAL_BALANCE):
        self.initial_balance = initial_balance
        self._balances:   Dict[str, float] = {s: initial_balance for s in SYMBOLS}
        self._peaks:      Dict[str, float] = {s: initial_balance for s in SYMBOLS}
        self._frozen:     Dict[str, bool]  = {s: False           for s in SYMBOLS}
        self._positions:  Dict[str, int]   = {s: 1               for s in SYMBOLS}
        self._sizes:      Dict[str, float] = {s: 1.0             for s in SYMBOLS}

    # ── Main gate ─────────────────────────────────────────────────────────────
    def check_action(
        self,
        symbol:    str,
        action:    int,         # 0 = HOLD, 1 = FLIP
        timestamp: Optional[datetime] = None,
    ) -> int:
        """
        Gate an RL agent's proposed action through all risk rules.
        Returns (possibly overridden) action: 0 or 1.
        """
        if self._frozen.get(symbol, False):
            if action == 1:
                logger.debug(f"[RISK] {symbol}: FLIP blocked — circuit breaker active")
            return 0   # Force HOLD, no flips while frozen

        if self._rollover_guard(symbol, timestamp):
            return 0   # Force HOLD near daily close

        return action   # Pass through

    def position_size(self, symbol: str) -> float:
        """
        Returns a size multiplier in (0, 1] for position sizing.
        Applied on top of the base lot size defined in the execution layer.
        """
        return self._sizes.get(symbol, 1.0)

    # ── Update state after each bar ───────────────────────────────────────────
    def update(
        self,
        symbol:    str,
        pnl:       float,
        position:  int,
        timestamp: Optional[datetime] = None,
    ):
        """
        Call once per bar per symbol with the realised step PnL.
        """
        self._balances[symbol] += pnl
        self._peaks[symbol]     = max(self._peaks[symbol], self._balances[symbol])
        self._positions[symbol] = position

        dd = self._drawdown(symbol)

        # Circuit breaker
        if not self._frozen[symbol] and dd > MAX_DRAWDOWN_PCT:
            self._frozen[symbol] = True
            logger.warning(
                f"[RISK] {symbol} circuit breaker OPEN — "
                f"drawdown={dd:.2%}  balance={self._balances[symbol]:.2f}"
            )

        if self._frozen[symbol] and dd < CIRCUIT_BREAKER_PCT:
            self._frozen[symbol] = False
            logger.info(f"[RISK] {symbol} circuit breaker CLOSED — drawdown={dd:.2%}")

        # Session size
        self._sizes[symbol] = self._session_size(timestamp)

        # Correlation cap
        self._apply_corr_cap(symbol, timestamp)

    # ── Portfolio summary ─────────────────────────────────────────────────────
    def portfolio_summary(self) -> dict:
        total_balance = sum(self._balances.values())
        total_dd      = {s: self._drawdown(s) for s in SYMBOLS}
        return {
            "total_balance": total_balance,
            "per_symbol": {
                s: {
                    "balance":  self._balances[s],
                    "drawdown": total_dd[s],
                    "frozen":   self._frozen[s],
                    "size":     self._sizes[s],
                    "position": self._positions[s],
                }
                for s in SYMBOLS
            }
        }

    def reset(self):
        self.__init__(self.initial_balance)

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _drawdown(self, symbol: str) -> float:
        peak = self._peaks[symbol]
        return (peak - self._balances[symbol]) / max(peak, 1e-8)

    @staticmethod
    def _session_size(timestamp: Optional[datetime]) -> float:
        """
        Reduce size during illiquid sessions.
        Full size during London (07:00–16:00 UTC) and NY overlap (12:00–17:00 UTC).
        Reduced size during Asia-only (20:00–07:00 UTC).
        """
        if timestamp is None:
            return 1.0
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        hour = timestamp.hour
        if 7 <= hour < 17:
            return 1.0        # London / NY — full size
        return SESSION_SIZE_REDUCTION

    def _rollover_guard(self, symbol: str, timestamp: Optional[datetime]) -> bool:
        """
        Prevent flips on metals 30 minutes before daily close (21:30–22:00 UTC)
        to avoid large rollover swaps.
        """
        if symbol not in ("GOLD", "SILVER"):
            return False
        if timestamp is None:
            return False
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        h, m = timestamp.hour, timestamp.minute
        return (h == 21 and m >= 30) or (h == 22 and m < 5)

    def _apply_corr_cap(self, symbol: str, timestamp: Optional[datetime]):
        """
        If combined exposure in a correlation group exceeds the cap,
        halve the size for every member of that group.
        """
        for group in CORR_GROUPS:
            if symbol not in group:
                continue
            # Count how many members of the group are active (not frozen)
            active = [s for s in group if s in SYMBOLS and not self._frozen[s]]
            if len(active) > 1:
                # Each active member uses SESSION_SIZE; combined exposure check
                combined = len(active) * MAX_CORR_EXPOSURE_PCT / 2
                if combined > MAX_CORR_EXPOSURE_PCT:
                    for s in active:
                        self._sizes[s] = min(self._sizes[s], 0.5)
