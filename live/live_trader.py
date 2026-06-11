"""
live/live_trader.py  —  Main always-in trading loop  (with trailing SL)
──────────────────────────────────────────────────────────────────────────────
Stop-loss behaviour
───────────────────
On every position open:
  → SL placed immediately at  current_price ± (ATR × TRAIL_STOP_ATR_MULT)
  → No TP ever set (always-in — model decides when to exit)

Every H1 bar (before running the model):
  1. Check if any expected position was closed (by SL or externally)
  2. If SL was hit → re-enter same direction immediately (always-in)
  3. Trail SL for all open positions (never move SL backwards)
  4. Run model → HOLD or FLIP

Circuit breaker
───────────────
On drawdown > MAX_DRAWDOWN_PCT:
  → Close ALL positions immediately (stop bleeding)
  → Halt — no new positions
  → Resume when equity recovers to CIRCUIT_BREAKER_RECOVERY_PCT

Shutdown (Ctrl+C / SIGTERM)
────────────────────────────
CLOSE_ON_STOP=true  → close all positions, then exit
CLOSE_ON_STOP=false → leave open (only if you have broker-side stops)
"""

import sys
import time
import signal
import platform
import pathlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR, PIP_VALUE
from live.settings import LiveSettings, load_settings
from live.mt5_bridge import MT5Bridge
from live.feature_builder import FeatureBuilder
from live.risk_guard import RiskGuard
from live.notifier import Notifier

PIP_USD_PER_LOT = {
    "GOLD":   1.00,
    "SILVER": 5.00,
    "EURUSD": 10.00,
    "GBPUSD": 10.00,
    "USDJPY": 6.28,
    "ETHUSD": 0.10,
    "BTCUSD": 0.10,
    "US30":   1.00,
    "US100":  1.00,
    "US500":  1.00,
    "UK100":  1.25,
    "AUS200": 0.65,
    "GER40":  1.10,
    "JP225":  0.0065,
}


class SymbolState:
    def __init__(self, symbol: str, initial_balance: float):
        self.symbol       = symbol
        self.position     = 0       # 0=none, +1=long, -1=short
        self.ticket       = None    # MT5 ticket of current position
        self.steps_held   = 0
        self.n_flips      = 0
        self.daily_pnl    = 0.0
        self.peak_balance = initial_balance
        self.last_bar_time: Optional[datetime] = None




def _pip_distance_to_price(symbol: str, pips: float) -> float:
    return float(PIP_VALUE[symbol] * pips)

class LiveTrader:
    """
    Orchestrates all live trading.
    Usage:  LiveTrader().start()
    """

    def __init__(self, settings: Optional[LiveSettings] = None, ui=None):
        self.s        = settings or load_settings()
        self.bridge   = MT5Bridge(self.s)
        self.notifier = Notifier(self.s)
        self.risk     = None
        self.builder  = None
        self.models   = {}
        self.states: Dict[str, SymbolState] = {}
        self._running = False
        self.ui = ui
        self._reconnect_count = 0
        self._started_at: Optional[datetime] = None

        # Circuit breaker state
        self._cb_active         = False
        self._cb_halt_equity    = 0.0
        self._cb_close_balance  = 0.0
        self._cb_cooldown_until: Optional[datetime] = None

        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        self._mt5_was_down = False

    def _ensure_mt5_connection(self) -> bool:
        """Keep the MT5 bridge alive; reconnect and resync after outages."""
        if self.bridge.ensure_connection():
            if self._mt5_was_down:
                self._mt5_was_down = False
                self._reconnect_count += 1
                logger.info("MT5 connection restored — reconciling positions.")
                try:
                    self._reconcile_positions()
                except Exception as exc:
                    logger.error(f"Reconcile after reconnect failed: {exc}", exc_info=True)
                self._push_ui_update("mt5_restored")
            return True

        if not self._mt5_was_down:
            self._mt5_was_down = True
            logger.warning("MT5 connection lost — pausing trading and retrying.")
            self._push_ui_update("mt5_lost")
        return False

    def _push_ui_update(self, reason: str = ""):
        """Refresh the terminal dashboard if it is enabled."""
        if self.ui is None:
            return
        try:
            self.ui.update(self._build_ui_snapshot(reason=reason))
        except Exception as exc:
            logger.debug(f"UI update skipped: {exc}")

    def _build_ui_snapshot(self, reason: str = "") -> dict:
        """Collect live runtime state for the terminal dashboard."""
        now = datetime.now(tz=timezone.utc)
        balance = self.bridge.account_balance()
        equity = self.bridge.account_equity()
        positions = self.bridge.get_positions()
        exposure = sum(float(p.get("lots", 0.0)) for p in positions.values())
        open_trades = len(positions)

        if self.states:
            peak = max((state.peak_balance for state in self.states.values()), default=equity)
        else:
            peak = max(balance, equity, 1.0)
        drawdown = (peak - equity) / max(peak, 1.0)

        risk_status = {}
        if self.risk is not None and hasattr(self.risk, "get_status"):
            try:
                risk_status = self.risk.get_status()
            except Exception:
                risk_status = {}

        pos_rows = []
        for sym in self.s.active_symbols:
            pos = positions.get(sym)
            if pos:
                spread = 0.0
                try:
                    spread = self.bridge.get_current_spread_pips(sym)
                except Exception:
                    spread = 0.0
                pos_rows.append({
                    "symbol": sym,
                    "side": pos["side"],
                    "side_text": "LONG" if pos["side"] == 1 else "SHORT",
                    "lots": float(pos["lots"]),
                    "profit": float(pos.get("profit", 0.0)),
                    "ticket": pos["ticket"],
                    "spread": float(spread),
                })

        snapshot = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M UTC"),
            "mode": self.s.trading_mode.upper(),
            "connection": "CONNECTED" if self.bridge.is_connected() else "DISCONNECTED",
            "mt5_down": self._mt5_was_down,
            "uptime": self._format_uptime(now),
            "balance": balance,
            "equity": equity,
            "drawdown_pct": drawdown,
            "daily_loss_usd": risk_status.get("daily_loss_usd", 0.0),
            "daily_halt": risk_status.get("daily_halt", False),
            "circuit_breaker_active": self._cb_active,
            "open_trades": open_trades,
            "exposure_lots": exposure,
            "models_loaded": len(self.models),
            "models_expected": len(self.s.active_symbols),
            "symbols": list(self.s.active_symbols),
            "reconnects": self._reconnect_count,
            "risk_ready": self.risk is not None,
            "positions": pos_rows,
            "reason": reason,
        }
        if "daily_start_equity" in risk_status:
            snapshot["daily_start_equity"] = risk_status["daily_start_equity"]
        if "daily_start_balance" in risk_status:
            snapshot["daily_start_balance"] = risk_status["daily_start_balance"]
        if "last_connect_error" in risk_status:
            snapshot["last_connect_error"] = risk_status["last_connect_error"]
        if hasattr(self.bridge, "last_connect_error"):
            snapshot["last_connect_error"] = getattr(self.bridge, "last_connect_error")
        return snapshot

    def _format_uptime(self, now: Optional[datetime] = None) -> str:
        now = now or datetime.now(tz=timezone.utc)
        started = getattr(self, "_started_at", None)
        if started is None:
            self._started_at = now
            started = now
        delta = now - started
        total = int(delta.total_seconds())
        hours, rem = divmod(total, 3600)
        mins, secs = divmod(rem, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    # ── Entry point ────────────────────────────────────────────────────────────
    def start(self):
        logger.info("=" * 60)
        logger.info("  Always-In Bot  —  Starting")
        logger.info(f"  Mode          : {self.s.trading_mode}")
        logger.info(f"  Symbols       : {self.s.active_symbols}")
        logger.info(f"  Trailing SL   : {'enabled  mult=' + str(self.s.trail_stop_atr_mult) if self.s.trail_stop_enabled else 'disabled'}")
        logger.info(f"  Close on stop : {self.s.close_on_stop}")
        logger.info("=" * 60)

        while not self.bridge.ensure_connection():
            logger.warning("Waiting for MT5 terminal to become available...")
            time.sleep(max(1, int(self.s.retry_delay_seconds)))

        if not self.bridge.resolve_all():
            raise RuntimeError("No symbols resolved. Check .env.")

        balance      = self.bridge.account_balance()
        self.risk    = RiskGuard(self.bridge, self.s)
        self.builder = FeatureBuilder(self.bridge, self.s)

        self._load_models()

        for sym in self.s.active_symbols:
            self.states[sym] = SymbolState(sym, balance)

        self._reconcile_positions()
        self.notifier.startup(self.s.trading_mode, self.s.active_symbols, balance)
        self._push_ui_update("startup")

        logger.success("Startup complete — entering trading loop")
        self._running = True
        self._main_loop()

    # ── Main loop ──────────────────────────────────────────────────────────────
    def _main_loop(self):
        last_summary_date = None

        while self._running:
            now = datetime.now(tz=timezone.utc)

            if not self._ensure_mt5_connection():
                self._push_ui_update("loop")
                time.sleep(self.s.bar_check_interval_sec)
                continue

            if (now.hour == self.s.telegram_daily_summary_hour
                    and now.date() != last_summary_date):
                self._send_daily_summary()
                last_summary_date = now.date()

            self._check_circuit_breaker(now)

            if not self._cb_active:
                for sym in self.s.active_symbols:
                    try:
                        if self._new_bar_available(sym, now):
                            self._process_bar(sym, now)
                    except Exception as e:
                        logger.error(f"[{sym}] Bar error: {e}", exc_info=True)
                        self.notifier.error(f"{sym}: {e}")
            else:
                if now.second < self.s.bar_check_interval_sec:
                    recovery_target = self._cb_halt_equity * (
                        1 - self.s.circuit_breaker_recovery_pct
                    )
                    logger.info(
                        f"[CB] HALTED — waiting for equity "
                        f"recovery to ${recovery_target:,.2f}"
                    )

            self._push_ui_update("loop")
            time.sleep(self.s.bar_check_interval_sec)

    # ── Per-bar processing ─────────────────────────────────────────────────────
    def _new_bar_available(self, symbol: str, now: datetime) -> bool:
        state            = self.states[symbol]
        current_bar_open = now.replace(minute=0, second=0, microsecond=0)
        trigger_time     = current_bar_open + timedelta(seconds=self.s.exec_delay_sec)
        if now < trigger_time:
            return False
        return state.last_bar_time != current_bar_open

    def _process_bar(self, symbol: str, now: datetime):
        state            = self.states[symbol]
        current_bar_open = now.replace(minute=0, second=0, microsecond=0)

        if not self._ensure_mt5_connection():
            return

        self.risk.update(now)

        # ── Step 1: Detect SL-triggered close ─────────────────────────────────
        if state.position != 0 and state.ticket is not None:
            if not self.bridge.position_exists(symbol, state.ticket):
                logger.warning(
                    f"[{symbol}] Position ticket={state.ticket} no longer exists "
                    f"— likely closed by SL or external action. "
                    f"Re-entering same direction to maintain always-in."
                )
                self.notifier.error(
                    f"{symbol}: SL or external close detected — re-entering "
                    f"{'LONG' if state.position==1 else 'SHORT'}"
                )
                self._reenter_position(symbol, state.position)
                state.last_bar_time = current_bar_open
                self._push_ui_update("reenter")
                return

        # ── Step 2: Trail SL on open positions ────────────────────────────────
        if (self.s.trail_stop_enabled
                and state.position != 0
                and state.ticket is not None):
            atr = self.builder.get_last_atr(symbol)
            self.bridge.trail_stop_loss(
                logical   = symbol,
                ticket    = state.ticket,
                side      = state.position,
                atr       = atr,
                atr_mult  = self.s.trail_stop_atr_mult,
            )

        # ── Step 3: Build observation ──────────────────────────────────────────
        obs = self.builder.build_observation(
            symbol     = symbol,
            position   = state.position if state.position != 0 else 1,
            balance    = self.bridge.account_balance(),
            peak       = state.peak_balance,
            n_flips    = state.n_flips,
            steps_held = state.steps_held,
        )

        if obs is None:
            logger.warning(f"[{symbol}] Cannot build obs — skipping bar")
            state.last_bar_time = current_bar_open
            return

        # ── Step 4: Enter initial position if needed ───────────────────────────
        if state.position == 0:
            self._initialise_position(symbol, obs)
            state.last_bar_time = current_bar_open
            return

        # ── Step 5: Run model ──────────────────────────────────────────────────
        model = self.models.get(symbol)
        if model is None:
            state.last_bar_time = current_bar_open
            return

        action, _ = model.predict(obs, deterministic=True)
        action     = int(action)
        action     = self.risk.check_action(symbol, action, now)

        if action == 1:
            self._execute_flip(symbol)
            state.steps_held = 0
        else:
            state.steps_held += 1

        state.last_bar_time = current_bar_open

    # ── Position management ────────────────────────────────────────────────────
    def _initialise_position(self, symbol: str, obs: np.ndarray):
        model = self.models.get(symbol)
        if model is None:
            return
        action, _ = model.predict(obs, deterministic=True)
        new_side   = 1 if int(action) == 0 else -1
        self._open_with_sl(symbol, new_side)

    def _execute_flip(self, symbol: str):
        state    = self.states[symbol]
        new_side = -state.position
        self._open_with_sl(symbol, new_side)

    def _reenter_position(self, symbol: str, side: int):
        """Re-open a position after SL close — same direction, always-in."""
        self._open_with_sl(symbol, side, is_reentry=True)

    def _initial_stop_distance(self, symbol: str, atr: float) -> float:
        atr_distance = atr * self.s.trail_stop_atr_mult if self.s.trail_stop_enabled else 0.0
        emergency_distance = _pip_distance_to_price(symbol, self.s.emergency_stop_pips) if self.s.emergency_stop_pips > 0 else 0.0
        return max(atr_distance, emergency_distance)

    def _set_initial_stop(self, symbol: str, ticket: int, side: int, ref_price: float, atr: float) -> bool:
        distance = self._initial_stop_distance(symbol, atr)
        if distance <= 0 or ref_price <= 0:
            return False
        sl_price = ref_price - distance if side == 1 else ref_price + distance
        sl_price = self.bridge.calc_sl_price(symbol, side, ref_price, distance, 1.0)
        return self.bridge.set_stop_loss(symbol, ticket, sl_price)

    def _open_with_sl(self, symbol: str, new_side: int, is_reentry: bool = False):
        """
        Open (or flip) a position then immediately set the trailing SL.
        This is the single point of entry for all position opens.
        """
        state     = self.states[symbol]
        old_side  = state.position
        lots      = self._calculate_lots(symbol)
        size_mult = self.risk.get_size_multiplier(symbol)
        lots      = max(self.s.min_lots, lots * size_mult)

        # Execute the flip/open
        ticket = self.bridge.flip_position(symbol, new_side, lots)

        if ticket is None:
            logger.error(f"[{symbol}] Order failed — position unchanged")
            self._push_ui_update("order_failed")
            return

        # Update internal state
        state.position   = new_side
        state.ticket     = ticket
        state.n_flips   += 1
        state.steps_held = 0
        self._push_ui_update("position_opened")

        label = "RE-ENTRY" if is_reentry else ("INITIAL" if old_side == 0 else "FLIP")
        logger.success(
            f"[{symbol}] {label} → {'LONG' if new_side==1 else 'SHORT'}  "
            f"{lots:.3f} lots  ticket={ticket}  flips={state.n_flips}"
        )

        # Set initial SL immediately after opening
        atr = self.builder.get_last_atr(symbol)
        try:
            tick = self.bridge.get_tick(symbol)
            ref_price = (tick.ask if new_side == 1 else tick.bid) if tick else 0.0
        except Exception:
            ref_price = 0.0

        if ref_price > 0 and (self.s.trail_stop_enabled or self.s.emergency_stop_pips > 0):
            distance = self._initial_stop_distance(symbol, atr)
            if distance > 0:
                sl_price = ref_price - distance if new_side == 1 else ref_price + distance
                sl_price = self.bridge.calc_sl_price(symbol, new_side, ref_price, distance, 1.0)
                ok = self.bridge.set_stop_loss(symbol, ticket, sl_price)
                if ok:
                    stop_label = "ATR" if self.s.trail_stop_enabled and distance == atr * self.s.trail_stop_atr_mult else "EMERGENCY"
                    logger.info(
                        f"[{symbol}] SL set at {sl_price:.5f}  "
                        f"({'below' if new_side==1 else 'above'} entry by "
                        f"{abs(ref_price - sl_price):.5f})  [{stop_label}]"
                    )

        # Notify flip (not for initial/reentry)
        if old_side != 0 and not is_reentry:
            try:
                tick = self.bridge.get_tick(symbol)
                price = (tick.ask if new_side == 1 else tick.bid) if tick else 0.0
            except Exception:
                price = 0.0
            self.notifier.flip(
                symbol=symbol, old_side=old_side, new_side=new_side,
                lots=lots, price=price,
                balance=self.bridge.account_balance(),
            )

    # ── Circuit breaker ────────────────────────────────────────────────────────
    def _check_circuit_breaker(self, now: datetime):
        """
        Fixed circuit breaker — prevents death spiral by requiring:
          1. Cooldown period to expire before re-entering (default 60 min)
          2. Balance to actually recover above post-close level (real money back)
          3. Drawdown to subside below recovery threshold
        """
        equity  = self.bridge.account_equity()
        balance = self.bridge.account_balance()

        for state in self.states.values():
            state.peak_balance = max(state.peak_balance, equity)
        peak   = max((s.peak_balance for s in self.states.values()), default=equity)
        dd_pct = (peak - equity) / max(peak, 1.0)

        # ── Trigger ────────────────────────────────────────────────────────────
        if not self._cb_active and dd_pct > self.s.max_drawdown_pct:
            self._cb_active      = True
            self._cb_halt_equity = equity
            cooldown_mins        = getattr(self.s, "cb_cooldown_minutes", 60)
            self._cb_cooldown_until = now + timedelta(minutes=cooldown_mins)

            logger.error(
                f"[CB] FIRED — drawdown={dd_pct:.2%}  equity=${equity:,.2f}\n"
                f"     No re-entry until: "
                f"{self._cb_cooldown_until.strftime('%H:%M UTC')} "
                f"AND balance recovers"
            )
            self.notifier.circuit_breaker("PORTFOLIO", dd_pct, frozen=True)
            self._close_all_positions("circuit breaker")
            self._push_ui_update("circuit_breaker_fired")

            # Record balance AFTER closing — recovery must exceed this
            self._cb_close_balance = self.bridge.account_balance()
            logger.info(
                f"[CB] Positions closed. Balance=${self._cb_close_balance:,.2f}  "
                f"(must recover above this before re-entering)"
            )

        # ── Recovery ───────────────────────────────────────────────────────────
        elif self._cb_active:
            cooldown_expired = (self._cb_cooldown_until is not None
                                and now >= self._cb_cooldown_until)
            equity_recovered = balance > self._cb_close_balance
            dd_subsided      = dd_pct < self.s.circuit_breaker_recovery_pct

            if cooldown_expired and equity_recovered and dd_subsided:
                self._cb_active = False
                logger.info(
                    f"[CB] CLEARED — balance=${balance:,.2f}  "
                    f"dd={dd_pct:.2%}"
                )
                self.notifier.circuit_breaker("PORTFOLIO", dd_pct, frozen=False)
                self._push_ui_update("circuit_breaker_recovered")
                for state in self.states.values():
                    state.position      = 0
                    state.ticket        = None
                    state.last_bar_time = None
            elif now.second < self.s.bar_check_interval_sec:
                # Log status once per minute
                reasons = []
                if not cooldown_expired and self._cb_cooldown_until:
                    rem = max(0, int((self._cb_cooldown_until - now).total_seconds() // 60))
                    reasons.append(f"cooldown {rem}m left")
                if not equity_recovered:
                    gap = self._cb_close_balance - balance
                    reasons.append(f"need +${gap:,.2f} more")
                if not dd_subsided:
                    reasons.append(f"dd {dd_pct:.2%} still high")
                logger.info(f"[CB] HALTED — {' | '.join(reasons)}")

    # ── Close all positions ────────────────────────────────────────────────────
    def _close_all_positions(self, reason: str = ""):
        if not self.bridge.ensure_connection():
            logger.error(f"[CLOSE-ALL] Cannot close positions because MT5 is disconnected ({reason})")
            return
        positions = self.bridge.get_positions()
        if not positions:
            logger.info(f"[CLOSE-ALL] No open positions ({reason})")
            return

        logger.info(f"[CLOSE-ALL] Closing {len(positions)} position(s)  [{reason}]")
        total_pnl = 0.0
        for sym, pos in positions.items():
            closed = self.bridge.close_position(
                sym, pos["ticket"], pos["lots"], pos["side"]
            )
            if closed:
                pnl = pos.get("profit", 0.0)
                total_pnl += pnl
                self.states[sym].position = 0
                self.states[sym].ticket   = None
                logger.info(
                    f"  ✓ {sym}  "
                    f"{'LONG' if pos['side']==1 else 'SHORT'}  "
                    f"{pos['lots']:.3f} lots  profit=${pnl:+,.2f}"
                )
            else:
                logger.error(f"  ✗ Failed to close {sym} — manual action needed")

        logger.info(f"[CLOSE-ALL] Total realised: ${total_pnl:+,.2f}")

    # ── Position sizing ────────────────────────────────────────────────────────
    def _calculate_lots(self, symbol: str) -> float:
        """
        ATR-based lot sizing using the LARGER of H1 and M15 ATR.

        During fast markets (exactly when CB fires), the last completed H1 bar
        underestimates current volatility because the spike is in the forming bar.
        Using max(H1_ATR, M15_ATR×4) gives a real-time volatility estimate and
        prevents lot sizes from ballooning when the market is moving hard.
        """
        try:
            atr_h1  = self.builder.get_last_atr(symbol)           # 14-bar H1 ATR
            atr_m15 = self.builder.get_last_atr_tf(symbol, "M15") # 14-bar M15 ATR
            # Scale M15 ATR to H1 equivalent (4 M15 bars per H1)
            atr_m15_scaled = atr_m15 * 4.0
            # Use the more conservative (larger) estimate
            atr = max(atr_h1, atr_m15_scaled)

            pip      = PIP_VALUE[symbol]
            pip_usd  = PIP_USD_PER_LOT[symbol]
            balance  = self.bridge.account_balance()
            risk     = balance * self.s.risk_pct
            atr_pips = atr * self.s.atr_stop_mult / pip
            lots     = risk / (atr_pips * pip_usd + 1e-10)
            lots     = self.bridge.normalise_lots(symbol, lots)
            logger.debug(
                f"[{symbol}] Sizing: H1_ATR={atr_h1:.4f}  "
                f"M15_ATR×4={atr_m15_scaled:.4f}  "
                f"using={atr:.4f}  lots={lots:.3f}"
            )
            return lots
        except Exception as e:
            logger.error(f"[{symbol}] Lot calc error: {e}")
            return self.s.min_lots

    # ── Startup helpers ────────────────────────────────────────────────────────
    def _load_models(self):
        from stable_baselines3 import PPO
        for sym in self.s.active_symbols:
            final = MODEL_DIR / f"ppo_{sym}_final.zip"
            best  = MODEL_DIR / f"ppo_{sym}" / "best_model.zip"
            path  = final if final.exists() else (best if best.exists() else None)
            if path is None:
                logger.error(f"[{sym}] No model found")
                continue
            try:
                self.models[sym] = PPO.load(str(path))
                logger.info(f"[{sym}] Model loaded: {path.name}")
            except Exception as e:
                logger.error(f"[{sym}] Load failed: {e}")

    def _reconcile_positions(self):
        existing = self.bridge.get_positions()
        if not existing:
            logger.info("No existing positions to reconcile.")
            return
        for sym, pos in existing.items():
            if sym in self.states:
                self.states[sym].position = pos["side"]
                self.states[sym].ticket   = pos["ticket"]
                logger.info(
                    f"  Reconciled {sym}: "
                    f"{'LONG' if pos['side']==1 else 'SHORT'}  "
                    f"{pos['lots']:.3f} lots  ticket={pos['ticket']}"
                )
                # Ensure SL is set on reconciled positions
                current_sl = self.bridge.get_position_sl(sym, pos["ticket"])
                if current_sl == 0.0 and (self.s.trail_stop_enabled or self.s.emergency_stop_pips > 0):
                    atr = self.builder.get_last_atr(sym)
                    try:
                        tick = self.bridge.get_tick(sym)
                        ref = tick.bid if pos["side"] == 1 else tick.ask if tick else 0.0
                    except Exception:
                        ref = 0.0
                    if ref > 0:
                        distance = self._initial_stop_distance(sym, atr)
                        sl = ref - distance if pos["side"] == 1 else ref + distance
                        sl = self.bridge.calc_sl_price(sym, pos["side"], ref, distance, 1.0)
                        self.bridge.set_stop_loss(sym, pos["ticket"], sl)
                        logger.info(
                            f"  [SL] Set missing SL on {sym} "
                            f"ticket={pos['ticket']}  sl={sl:.5f}"
                        )

    def _send_daily_summary(self):
        stats   = {}
        balance = self.bridge.account_balance()
        for sym, state in self.states.items():
            stats[sym] = {
                "daily_pnl": state.daily_pnl,
                "n_flips":   state.n_flips,
                "position":  state.position,
            }
            state.daily_pnl = 0.0
        stats["_balance"] = balance
        self.notifier.daily_summary(stats)
        self._push_ui_update("daily_summary")

    # ── Shutdown ───────────────────────────────────────────────────────────────
    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received — stopping cleanly...")
        self._running = False

        if self.s.close_on_stop:
            logger.info("Closing all positions before exit...")
            self._close_all_positions("clean shutdown")
        else:
            positions = self.bridge.get_positions()
            if positions:
                logger.warning(
                    f"CLOSE_ON_STOP=false — "
                    f"leaving {len(positions)} position(s) open: "
                    f"{list(positions.keys())}"
                )

        self.notifier.shutdown(
            "Positions closed" if self.s.close_on_stop else "Positions left open"
        )
        self._push_ui_update("shutdown")
        self.bridge.disconnect()
        if self.ui is not None:
            self.ui.stop()
        logger.info("Bot stopped.")

    def stop(self):
        self._handle_shutdown(None, None)
