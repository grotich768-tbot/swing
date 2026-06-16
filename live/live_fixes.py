"""
live/live_fixes.py  —  Three critical live trading fixes
──────────────────────────────────────────────────────────────────────────────
Fix 1: Remove 5% risk cap from settings
Fix 2: Re-entry guard — prevent same-direction re-entry on early position close
Fix 3: Spread guard — force flip when spread drops after being too wide
Fix 4: Correct MT5 spread conversion (points → pips)

Apply in live_trader.py:
    from live.live_fixes import LiveFixes
    self._fixes = LiveFixes(settings, bridge)

    # In your bar loop:
    spread_pips = self._fixes.get_spread_pips(symbol)
    if self._fixes.spread_too_wide(symbol, spread_pips):
        continue   # skip this bar — don't flip into wide spread
    if self._fixes.is_reentry_blocked(symbol, current_side):
        continue   # skip — same direction reentry too soon
    self._fixes.record_flip(symbol, current_side)
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Fix 4: Correct MT5 spread → pips conversion
# ─────────────────────────────────────────────────────────────────────────────
def mt5_spread_to_pips(spread_points: int, symbol: str, mt5_info=None) -> float:
    """
    Convert MT5 spread (in points) to pips correctly.

    MT5 always reports spread in POINTS (the smallest price increment).
    1 pip = 10 points for almost everything.

    The only exception is when the broker quotes indices/crypto as
    whole numbers (point=1.0, digits=0 or 1) — then 1 pip = 1 point.

    Verified conversions:
        GOLD    60 pts  × 0.01  / (0.01*10=0.10)  = 6.0 pips  ✓
        EURUSD  10 pts  × 0.00001/(0.00001*10)     = 1.0 pip   ✓
        BTCUSD  2976pts × 0.1   / (0.1*10=1.0)     = 297.6pips ✓
        US30    39 pts  × 1.0   / (1.0*10=10)       = 3.9 pips  ✓
          (US30: point=1.0, broker spread=39 points = 3.9 pips)
    """
    if mt5_info is not None:
        point  = mt5_info.point
        digits = mt5_info.digits

        # pip = point * 10 universally
        # The broker_config spread values already represent 10-point pips
        pip_size = point * 10

        if pip_size > 0:
            return round(spread_points * point / pip_size, 2)
        return float(spread_points)

    # Fallback without mt5_info
    try:
        from config import PIP_VALUE
        pip_size = PIP_VALUE.get(symbol, 0.0001)
        # pip_size from config = 1 pip in price units
        # point ≈ pip_size / 10 for all symbols
        # spread_pips = spread_points * point / pip_size
        #             = spread_points * (pip_size/10) / pip_size
        #             = spread_points / 10
        return spread_points / 10.0
    except Exception:
        return spread_points / 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Main fixes class
# ─────────────────────────────────────────────────────────────────────────────
class LiveFixes:
    """
    Encapsulates all three live trading fixes.

    Parameters
    ----------
    settings : live settings object
    bridge   : MT5Bridge instance
    """

    # Spread multiplier thresholds
    SPREAD_NORMAL_MULT  = 2.0   # spread > 2x typical → too wide to flip
    SPREAD_EXTREME_MULT = 5.0   # spread > 5x typical → extreme, hold regardless
    SPREAD_CLEAR_MULT   = 1.5   # spread must drop to < 1.5x typical to resume

    # Re-entry guard
    REENTRY_BLOCK_SECONDS = 3600   # block same-direction re-entry for 1 hour (1 bar)

    def __init__(self, settings, bridge):
        self._settings      = settings
        self._bridge        = bridge

        # Per-symbol state
        self._last_flip_time:  Dict[str, datetime] = {}
        self._last_flip_side:  Dict[str, int]       = {}   # +1 long, -1 short
        self._spread_blocked:  Dict[str, bool]       = {}
        self._typical_spreads: Dict[str, float]      = {}

        # Load typical spreads from broker_config / config
        self._load_typical_spreads()

        logger.info("[LiveFixes] Initialised")

    # ── Fix 1: Remove 5% risk cap ─────────────────────────────────────────────
    @staticmethod
    def remove_risk_cap(settings) -> object:
        """
        Remove the 5% risk cap from settings.
        Risk is controlled by MAX_POSITION_PCT in config (2%) and
        adaptive sizing in risk_engine — not a hard 5% cap.

        Call this right after load_settings():
            settings = load_settings()
            settings = LiveFixes.remove_risk_cap(settings)
        """
        removed = []

        # Common attribute names for risk cap across settings implementations
        for attr in ("max_risk_pct", "risk_cap", "max_risk", "max_position_pct_cap",
                     "session_risk_cap", "risk_ceiling"):
            if hasattr(settings, attr):
                val = getattr(settings, attr)
                if val is not None and val <= 0.06:   # was a 5-6% cap
                    setattr(settings, attr, 1.0)       # set to 100% = effectively removed
                    removed.append(f"{attr}={val:.1%} → removed")

        # Also remove any hard lot cap if it's based on the 5% figure
        if hasattr(settings, "max_lots") and settings.max_lots < 0.5:
            logger.warning(
                f"[LiveFixes] max_lots={settings.max_lots} — "
                f"check if this is intentional or a side-effect of the 5% cap"
            )

        if removed:
            logger.info(f"[LiveFixes] Risk cap removed: {removed}")
        else:
            logger.info(
                "[LiveFixes] No 5% risk cap found in settings attributes — "
                "check .env file for RISK_CAP or MAX_RISK_PCT variables"
            )

        return settings

    # ── Fix 2: Re-entry guard ─────────────────────────────────────────────────
    def is_reentry_blocked(self, symbol: str, proposed_side: int) -> bool:
        """
        Returns True if the proposed side is the same as the last flip
        and it happened within REENTRY_BLOCK_SECONDS ago.

        This prevents the bot from immediately re-entering the same direction
        when a position is closed before the next H1 bar completes.

        Parameters
        ----------
        symbol        : str  — e.g. "GOLD"
        proposed_side : int  — +1 long, -1 short

        Returns
        -------
        bool — True = block this entry
        """
        last_time = self._last_flip_time.get(symbol)
        last_side = self._last_flip_side.get(symbol)

        if last_time is None or last_side is None:
            return False   # no history — allow

        now     = datetime.now(timezone.utc)
        elapsed = (now - last_time).total_seconds()

        if elapsed < self.REENTRY_BLOCK_SECONDS and proposed_side == last_side:
            logger.info(
                f"[LiveFixes] {symbol}: re-entry BLOCKED  "
                f"side={'LONG' if proposed_side==1 else 'SHORT'}  "
                f"elapsed={elapsed:.0f}s  "
                f"(block={self.REENTRY_BLOCK_SECONDS}s)"
            )
            return True

        return False

    def record_flip(self, symbol: str, new_side: int):
        """Call this whenever a flip is executed."""
        self._last_flip_time[symbol] = datetime.now(timezone.utc)
        self._last_flip_side[symbol] = new_side
        logger.debug(
            f"[LiveFixes] {symbol}: flip recorded  "
            f"side={'LONG' if new_side==1 else 'SHORT'}"
        )

    def clear_reentry_block(self, symbol: str):
        """Manually clear re-entry block (e.g. after a full bar completes)."""
        self._last_flip_time.pop(symbol, None)
        self._last_flip_side.pop(symbol, None)

    # ── Fix 3: Spread guard ───────────────────────────────────────────────────
    def get_spread_pips(self, symbol: str) -> float:
        """
        Get current spread in pips — correctly converted from MT5 points.

        This fixes the MT5 spread-in-points issue.
        """
        try:
            from live.symbol_specs import get_specs
            specs    = get_specs()
            mt5_name = specs.mt5_name(symbol)
            if mt5_name:
                import MetaTrader5 as mt5
                info = mt5.symbol_info(mt5_name)
                if info:
                    return mt5_spread_to_pips(info.spread, symbol, info)
        except Exception as e:
            logger.debug(f"[LiveFixes] spread fetch failed: {e}")

        # Fallback to bridge method if available
        try:
            if hasattr(self._bridge, "get_current_spread_pips"):
                return float(self._bridge.get_current_spread_pips(symbol))
        except Exception:
            pass

        # Final fallback: typical spread
        return self._typical_spreads.get(symbol, 5.0)

    def spread_too_wide(self, symbol: str, current_spread_pips: float = None) -> bool:
        """
        Returns True if spread is too wide to safely flip.

        When spread is >2x typical:
          - Don't flip (cost too high, wrong-direction run is cheaper)
          - Wait for spread to normalize

        When spread drops back to <1.5x typical:
          - Re-evaluate and flip if model says so

        Parameters
        ----------
        symbol              : str
        current_spread_pips : float | None — pass in or auto-fetched

        Returns
        -------
        bool — True = spread too wide, hold current position
        """
        if current_spread_pips is None:
            current_spread_pips = self.get_spread_pips(symbol)

        typical = self._typical_spreads.get(symbol, current_spread_pips)
        ratio   = current_spread_pips / max(typical, 1e-6)

        was_blocked = self._spread_blocked.get(symbol, False)

        if ratio >= self.SPREAD_NORMAL_MULT:
            # Spread too wide — block flip
            if not was_blocked:
                logger.warning(
                    f"[LiveFixes] {symbol}: spread WIDE  "
                    f"{current_spread_pips:.1f} pips ({ratio:.1f}x typical={typical:.1f})  "
                    f"— flips blocked until spread normalises"
                )
            self._spread_blocked[symbol] = True
            return True

        if was_blocked and ratio >= self.SPREAD_CLEAR_MULT:
            # Spread improving but not clear yet — remain blocked
            logger.debug(
                f"[LiveFixes] {symbol}: spread still elevated "
                f"{current_spread_pips:.1f} pips ({ratio:.1f}x) — still blocked"
            )
            return True

        if was_blocked and ratio < self.SPREAD_CLEAR_MULT:
            # Spread cleared — unblock
            logger.info(
                f"[LiveFixes] {symbol}: spread NORMALISED  "
                f"{current_spread_pips:.1f} pips ({ratio:.1f}x)  "
                f"— flips resumed"
            )
            self._spread_blocked[symbol] = False

        return False

    def spread_ratio(self, symbol: str) -> float:
        """Current spread as a multiple of typical spread."""
        spread  = self.get_spread_pips(symbol)
        typical = self._typical_spreads.get(symbol, spread)
        return spread / max(typical, 1e-6)

    # ── Internal ──────────────────────────────────────────────────────────────
    def _load_typical_spreads(self):
        """Load typical spreads from broker_config / config."""
        try:
            from broker_config import get_broker_specs
            for sym, vals in get_broker_specs().items():
                self._typical_spreads[sym] = vals["spread"]
        except Exception:
            try:
                from config import SPREAD_PIPS
                self._typical_spreads = dict(SPREAD_PIPS)
            except Exception:
                pass
        logger.debug(f"[LiveFixes] Typical spreads: {self._typical_spreads}")
