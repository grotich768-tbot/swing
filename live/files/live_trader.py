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


class LiveTrader:
    """
    Orchestrates all live trading.
    Usage:  LiveTrader().start()
    """

    def __init__(self, settings: Optional[LiveSettings] = None):
        self.s        = settings or load_settings()
        self.bridge   = MT5Bridge(self.s)
        self.notifier = Notifier(self.s)
        self.risk     = None
        self.builder  = None
        self.models   = {}
        self.states: Dict[str, SymbolState] = {}
        self._running = False

        # Circuit breaker state
        self._cb_active      = False
        self._cb_halt_equity = 0.0

        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    # ── Entry point ────────────────────────────────────────────────────────────
    def start(self):
        logger.info("=" * 60)
        logger.info("  Always-In Bot  —  Starting")
        logger.info(f"  Mode          : {self.s.trading_mode}")
        logger.info(f"  Symbols       : {self.s.active_symbols}")
        logger.info(f"  Trailing SL   : {'enabled  mult=' + str(self.s.trail_stop_atr_mult) if self.s.trail_stop_enabled else 'disabled'}")
        logger.info(f"  Close on stop : {self.s.close_on_stop}")
        logger.info("=" * 60)

        if not self.bridge.connect():
            raise ConnectionError("Failed to connect to MT5.")

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

        logger.success("Startup complete — entering trading loop")
        self._running = True
        self._main_loop()

    # ── Main loop ──────────────────────────────────────────────────────────────
    def _main_loop(self):
        last_summary_date = None

        while self._running:
            now = datetime.now(tz=timezone.utc)

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
            return

        # Update internal state
        state.position   = new_side
        state.ticket     = ticket
        state.n_flips   += 1
        state.steps_held = 0

        label = "RE-ENTRY" if is_reentry else ("INITIAL" if old_side == 0 else "FLIP")
        logger.success(
            f"[{symbol}] {label} → {'LONG' if new_side==1 else 'SHORT'}  "
            f"{lots:.3f} lots  ticket={ticket}  flips={state.n_flips}"
        )

        # Set initial SL immediately after opening
        if self.s.trail_stop_enabled:
            atr = self.builder.get_last_atr(symbol)
            try:
                import MetaTrader5 as mt5
                broker = self.bridge.resolve(symbol)
                tick   = mt5.symbol_info_tick(broker)
                ref_price = (tick.ask if new_side == 1 else tick.bid) if tick else 0.0
            except Exception:
                ref_price = 0.0

            if ref_price > 0:
                sl_price = self.bridge.calc_sl_price(
                    symbol, new_side, ref_price, atr, self.s.trail_stop_atr_mult
                )
                ok = self.bridge.set_stop_loss(symbol, ticket, sl_price)
                if ok:
                    logger.info(
                        f"[{symbol}] SL set at {sl_price:.5f}  "
                        f"({'below' if new_side==1 else 'above'} entry by "
                        f"{abs(ref_price - sl_price):.5f}  "
                        f"={self.s.trail_stop_atr_mult:.1f}×ATR)"
                    )

        # Notify flip (not for initial/reentry)
        if old_side != 0 and not is_reentry:
            try:
                import MetaTrader5 as mt5
                broker = self.bridge.resolve(symbol)
                tick   = mt5.symbol_info_tick(broker)
                price  = (tick.ask if new_side == 1 else tick.bid) if tick else 0.0
            except Exception:
                price = 0.0
            self.notifier.flip(
                symbol=symbol, old_side=old_side, new_side=new_side,
                lots=lots, price=price,
                balance=self.bridge.account_balance(),
            )

    # ── Circuit breaker ────────────────────────────────────────────────────────
    def _check_circuit_breaker(self, now: datetime):
        equity = self.bridge.account_equity()

        for state in self.states.values():
            state.peak_balance = max(state.peak_balance, equity)

        peak   = max((s.peak_balance for s in self.states.values()), default=equity)
        dd_pct = (peak - equity) / max(peak, 1.0)

        if not self._cb_active and dd_pct > self.s.max_drawdown_pct:
            self._cb_active      = True
            self._cb_halt_equity = equity
            logger.error(
                f"[CB] FIRED — drawdown={dd_pct:.2%}  "
                f"equity=${equity:,.2f}"
            )
            self.notifier.circuit_breaker("PORTFOLIO", dd_pct, frozen=True)
            self._close_all_positions("circuit breaker")

        elif self._cb_active:
            recovery_target = self._cb_halt_equity * (
                1 - self.s.circuit_breaker_recovery_pct
            )
            if equity >= recovery_target:
                self._cb_active = False
                logger.info(
                    f"[CB] CLEARED — equity=${equity:,.2f}  "
                    f"target=${recovery_target:,.2f}"
                )
                self.notifier.circuit_breaker("PORTFOLIO", dd_pct, frozen=False)
                for state in self.states.values():
                    state.position      = 0
                    state.ticket        = None
                    state.last_bar_time = None

    # ── Close all positions ────────────────────────────────────────────────────
    def _close_all_positions(self, reason: str = ""):
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
        try:
            atr      = self.builder.get_last_atr(symbol)
            pip      = PIP_VALUE[symbol]
            pip_usd  = PIP_USD_PER_LOT[symbol]
            risk     = self.s.initial_balance * self.s.risk_pct
            atr_pips = atr * self.s.atr_stop_mult / pip
            lots     = risk / (atr_pips * pip_usd + 1e-10)
            return self.bridge.normalise_lots(symbol, lots)
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
                if self.s.trail_stop_enabled:
                    current_sl = self.bridge.get_position_sl(sym, pos["ticket"])
                    if current_sl == 0.0:
                        atr = self.builder.get_last_atr(sym)
                        try:
                            import MetaTrader5 as mt5
                            broker = self.bridge.resolve(sym)
                            tick   = mt5.symbol_info_tick(broker)
                            ref    = tick.bid if pos["side"] == 1 else tick.ask
                        except Exception:
                            ref = 0.0
                        if ref > 0:
                            sl = self.bridge.calc_sl_price(
                                sym, pos["side"], ref, atr,
                                self.s.trail_stop_atr_mult
                            )
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
        self.bridge.disconnect()
        logger.info("Bot stopped.")

    def stop(self):
        self._handle_shutdown(None, None)
