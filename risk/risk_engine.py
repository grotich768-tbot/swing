"""
risk/risk_engine.py  —  Portfolio-level risk management
──────────────────────────────────────────────────────────────────────────────
IMPROVEMENTS APPLIED:

Tier 4 — Adaptive position sizing
  • Reduce size after ADAPTIVE_LOSE_STREAK_CUTOFF consecutive losing flips
  • Increase size slightly after sustained profitable period
  • Kelly fraction estimate on rolling 50-trade window

Tier 4 — News filter
  • _news_filter() blocks flips in ±NEWS_FILTER_MINUTES window of known events
  • Prevents the bot being on wrong side of 50-pip macro spikes

Tier 2 — Session-aware sizing preserved from original
  • Full size London/NY; reduced Asia
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SYMBOLS, MAX_DRAWDOWN_PCT, CIRCUIT_BREAKER_PCT,
    MAX_CORR_EXPOSURE_PCT, SESSION_SIZE_REDUCTION,
    INITIAL_BALANCE,
    ADAPTIVE_SIZING_ENABLED,
    ADAPTIVE_LOSE_STREAK_CUTOFF, ADAPTIVE_LOSE_SIZE_MULT,
    ADAPTIVE_WIN_STREAK_CUTOFF,  ADAPTIVE_WIN_SIZE_MULT,
    KELLY_WINDOW,
    NEWS_FILTER_MINUTES,
)

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

        # Tier 4: adaptive sizing state
        self._trade_returns:  Dict[str, List[float]] = {s: [] for s in SYMBOLS}
        self._consec_losses:  Dict[str, int]         = {s: 0  for s in SYMBOLS}
        self._consec_wins:    Dict[str, int]         = {s: 0  for s in SYMBOLS}
        self._adaptive_mult:  Dict[str, float]       = {s: 1.0 for s in SYMBOLS}

    # ── Main gate ─────────────────────────────────────────────────────────────
    def check_action(
        self,
        symbol:    str,
        action:    int,
        timestamp: Optional[datetime] = None,
    ) -> int:
        """
        Gate an RL agent's proposed action through all risk rules.
        Returns (possibly overridden) action: 0 or 1.
        """
        if self._frozen.get(symbol, False):
            if action == 1:
                logger.debug(f"[RISK] {symbol}: FLIP blocked — circuit breaker active")
            return 0

        if self._rollover_guard(symbol, timestamp):
            return 0

        # Tier 4: News filter — block flips near major macro events
        if action == 1 and self._news_filter(timestamp):
            logger.debug(f"[RISK] {symbol}: FLIP blocked — near high-impact event")
            return 0

        return action

    def position_size(self, symbol: str) -> float:
        """
        Returns a size multiplier in (0, 1.2] for position sizing.
        Combines session filter + adaptive sizing.
        """
        base = self._sizes.get(symbol, 1.0)
        if ADAPTIVE_SIZING_ENABLED:
            base *= self._adaptive_mult.get(symbol, 1.0)
        return float(np.clip(base, 0.1, 1.2))

    # ── Update state after each bar ───────────────────────────────────────────
    def update(
        self,
        symbol:    str,
        pnl:       float,
        position:  int,
        timestamp: Optional[datetime] = None,
        flipped:   bool = False,
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
        session_size = self._session_size(timestamp)
        self._sizes[symbol] = session_size

        # Correlation cap
        self._apply_corr_cap(symbol, timestamp)

        # Tier 4: Adaptive sizing — update on flip boundaries
        if flipped and pnl != 0:
            self._update_adaptive_sizing(symbol, pnl)

    # ── Portfolio summary ─────────────────────────────────────────────────────
    def portfolio_summary(self) -> dict:
        total_balance = sum(self._balances.values())
        total_dd      = {s: self._drawdown(s) for s in SYMBOLS}
        return {
            "total_balance": total_balance,
            "per_symbol": {
                s: {
                    "balance":        self._balances[s],
                    "drawdown":       total_dd[s],
                    "frozen":         self._frozen[s],
                    "size":           self.position_size(s),
                    "position":       self._positions[s],
                    "adaptive_mult":  self._adaptive_mult[s],
                }
                for s in SYMBOLS
            }
        }

    def reset(self):
        self.__init__(self.initial_balance)

    # ── Tier 4: Adaptive position sizing ──────────────────────────────────────
    def _update_adaptive_sizing(self, symbol: str, trade_pnl: float):
        """
        Adjust size multiplier based on recent trade performance.

        After ADAPTIVE_LOSE_STREAK_CUTOFF consecutive losing flips →
            scale down to ADAPTIVE_LOSE_SIZE_MULT.
        After ADAPTIVE_WIN_STREAK_CUTOFF consecutive winning flips →
            scale up to ADAPTIVE_WIN_SIZE_MULT (capped).
        Kelly fraction provides a secondary estimate on the rolling window.
        """
        returns = self._trade_returns[symbol]
        returns.append(trade_pnl)
        if len(returns) > KELLY_WINDOW:
            returns.pop(0)

        if trade_pnl < 0:
            self._consec_losses[symbol] += 1
            self._consec_wins[symbol]    = 0
        else:
            self._consec_wins[symbol]   += 1
            self._consec_losses[symbol]  = 0

        losses = self._consec_losses[symbol]
        wins   = self._consec_wins[symbol]

        if losses >= ADAPTIVE_LOSE_STREAK_CUTOFF:
            mult = ADAPTIVE_LOSE_SIZE_MULT
            if losses == ADAPTIVE_LOSE_STREAK_CUTOFF:
                logger.info(
                    f"[RISK] {symbol}: {losses} consecutive losses — "
                    f"size reduced to {mult:.0%}"
                )
        elif wins >= ADAPTIVE_WIN_STREAK_CUTOFF and len(returns) >= 10:
            # Kelly-informed scale-up
            kelly = self._kelly_fraction(returns)
            mult  = min(ADAPTIVE_WIN_SIZE_MULT, max(1.0, 1.0 + kelly * 0.5))
        else:
            mult = 1.0

        self._adaptive_mult[symbol] = float(np.clip(mult, 0.3, 1.2))

    @staticmethod
    def _kelly_fraction(returns: List[float]) -> float:
        """
        Simplified Kelly fraction: f = edge / odds.
        edge = mean return,  odds = mean win / mean |loss|
        """
        wins   = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        if not wins or not losses:
            return 0.0
        win_rate  = len(wins) / len(returns)
        avg_win   = np.mean(wins)
        avg_loss  = abs(np.mean(losses))
        if avg_loss < 1e-10:
            return 0.0
        odds = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / odds
        return float(np.clip(kelly, 0.0, 0.5))   # cap at 50% Kelly

    # ── Tier 4: News / economic calendar filter ────────────────────────────────
    @staticmethod
    def _news_filter(timestamp: Optional[datetime]) -> bool:
        """
        Returns True (block flip) if timestamp falls within
        NEWS_FILTER_MINUTES of a known high-impact macro event.

        Hardcoded approximate windows — for production, connect to
        an economic calendar API (e.g. Forex Factory JSON, MT5 calendar).
        """
        if timestamp is None or NEWS_FILTER_MINUTES <= 0:
            return False

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Check approximate weekly event windows
        weekday = timestamp.weekday()  # 0=Mon … 6=Sun
        h, m    = timestamp.hour, timestamp.minute
        now_min = h * 60 + m

        # NFP: first Friday 13:30 UTC ± filter window
        if weekday == 4:   # Friday
            nfp_min = 13 * 60 + 30
            if abs(now_min - nfp_min) <= NEWS_FILTER_MINUTES:
                return True

        # CPI / PPI: Wednesday 12:30 UTC ± filter window
        if weekday == 2:   # Wednesday
            cpi_min = 12 * 60 + 30
            if abs(now_min - cpi_min) <= NEWS_FILTER_MINUTES:
                return True

        # FOMC: Wednesday 18:00 UTC ± filter window (twice a year exact,
        # but cost of over-filtering one hour/month is negligible)
        if weekday == 2:
            fomc_min = 18 * 60
            if abs(now_min - fomc_min) <= NEWS_FILTER_MINUTES:
                return True

        return False

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _drawdown(self, symbol: str) -> float:
        peak = self._peaks[symbol]
        return (peak - self._balances[symbol]) / max(peak, 1e-8)

    @staticmethod
    def _session_size(timestamp: Optional[datetime]) -> float:
        if timestamp is None:
            return 1.0
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        hour = timestamp.hour
        if 7 <= hour < 17:
            return 1.0
        return SESSION_SIZE_REDUCTION

    def _rollover_guard(self, symbol: str, timestamp: Optional[datetime]) -> bool:
        if symbol not in ("GOLD", "SILVER"):
            return False
        if timestamp is None:
            return False
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        h, m = timestamp.hour, timestamp.minute
        return (h == 21 and m >= 30) or (h == 22 and m < 5)

    def _apply_corr_cap(self, symbol: str, timestamp: Optional[datetime]):
        for group in CORR_GROUPS:
            if symbol not in group:
                continue
            active = [s for s in group if s in SYMBOLS and not self._frozen[s]]
            if len(active) > 1:
                combined = len(active) * MAX_CORR_EXPOSURE_PCT / 2
                if combined > MAX_CORR_EXPOSURE_PCT:
                    for s in active:
                        self._sizes[s] = min(self._sizes[s], 0.5)
