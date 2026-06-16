"""
live/mt5_bridge.py  —  MT5 low-level operations for live trading
──────────────────────────────────────────────────────────────────────────────
Wraps MetaTrader5 API with:
  • Auto symbol resolution (prefix/suffix/candidates)
  • Spread validation before execution
  • Retry logic on order failures
  • Clean position open/close/flip
"""

import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import MetaTrader5 as mt5
    MT5_OK = True
except ImportError:
    MT5_OK = False
    mt5 = None

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PIP_VALUE
from live.settings import LiveSettings


class MT5Bridge:
    """
    Manages all communication with the MT5 terminal for live trading.

    Parameters
    ----------
    settings : LiveSettings
    """

    def __init__(self, settings: LiveSettings):
        if not MT5_OK:
            raise EnvironmentError(
                "MetaTrader5 package not found.\n"
                "Install: pip install MetaTrader5\n"
                "Requires Windows + MT5 terminal."
            )
        self.s = settings
        self._resolved: dict = {}   # logical → broker symbol
        self._connected: bool = False
        self._last_connect_error: str = ""

    @property
    def last_connect_error(self) -> str:
        return self._last_connect_error


    def is_connected(self) -> bool:
        """Best-effort health check for the MT5 terminal connection."""
        try:
            info = mt5.terminal_info()
            account = mt5.account_info()
            return bool(info and account)
        except Exception:
            return False

    def ensure_connection(self, reconnect: bool = True) -> bool:
        """Ensure MT5 is connected; optionally attempt reconnects."""
        if self._connected and self.is_connected():
            return True
        if not reconnect:
            return False
        return self.reconnect()

    def reconnect(self) -> bool:
        """Reconnect to MT5 with a small retry loop."""
        try:
            mt5.shutdown()
        except Exception:
            pass
        self._connected = False

        attempts = max(1, int(getattr(self.s, "max_retries", 3)))
        delay = float(getattr(self.s, "retry_delay_seconds", 5))
        for attempt in range(1, attempts + 1):
            if self.connect():
                if self._resolved:
                    self._reselect_cached_symbols()
                return True
            self._last_connect_error = mt5.last_error()
            logger.warning(
                f"MT5 reconnect attempt {attempt}/{attempts} failed: {self._last_connect_error}"
            )
            if attempt < attempts:
                time.sleep(delay)
        return False

    def _reselect_cached_symbols(self):
        """Re-select previously resolved broker symbols after reconnect."""
        for broker in self._resolved.values():
            try:
                mt5.symbol_select(broker, True)
            except Exception:
                pass

    # ── Connection ────────────────────────────────────────────────────────────
    def connect(self) -> bool:
        kwargs = {}
        if self.s.mt5_path:
            kwargs["path"] = self.s.mt5_path

        if not mt5.initialize(**kwargs):
            self._connected = False
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False

        if self.s.mt5_login:
            if not mt5.login(
                self.s.mt5_login,
                password=self.s.mt5_password,
                server=self.s.mt5_server,
            ):
                self._connected = False
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False

        info    = mt5.terminal_info()
        account = mt5.account_info()
        self._connected = True
        logger.success(
            f"MT5 connected  build={info.build}  "
            f"account={account.login}  "
            f"broker={account.company}  "
            f"balance=${account.balance:,.2f}  "
            f"mode={'LIVE' if account.trade_mode == 0 else 'DEMO'}"
        )
        if self.s.is_live and account.trade_mode != 0:
            logger.warning("TRADING_MODE=LIVE but MT5 is on a DEMO account!")
        return True

    def disconnect(self):
        try:
            mt5.shutdown()
        finally:
            self._connected = False
        logger.info("MT5 disconnected.")

    def account_balance(self) -> float:
        if not self.ensure_connection(reconnect=False):
            return 0.0
        info = mt5.account_info()
        return float(info.balance) if info else 0.0

    def account_equity(self) -> float:
        if not self.ensure_connection(reconnect=False):
            return 0.0
        info = mt5.account_info()
        return float(info.equity) if info else 0.0

    # ── Symbol resolution ─────────────────────────────────────────────────────
    def resolve(self, logical: str) -> str:
        """
        Return the broker symbol name for a logical name (e.g. "GOLD").
        Tries the explicit name first, then candidates list.
        Caches the result after first resolution.
        """
        if logical in self._resolved:
            return self._resolved[logical]

        if not self.ensure_connection():
            raise ConnectionError("MT5 is not connected")

        all_syms = {s.name for s in (mt5.symbols_get() or [])}

        # 1. Try explicit name from .env
        explicit = self.s.symbol_name_map.get(logical, logical)
        if explicit in all_syms:
            mt5.symbol_select(explicit, True)
            self._resolved[logical] = explicit
            logger.info(f"  {logical:8s}  →  {explicit}  (explicit .env name)")
            return explicit

        # 2. Try candidates list
        candidates = self.s.symbol_candidates_map.get(logical, [logical])
        for cand in candidates:
            if cand in all_syms:
                mt5.symbol_select(cand, True)
                self._resolved[logical] = cand
                logger.info(f"  {logical:8s}  →  {cand}  (from candidates)")
                return cand

        raise ValueError(
            f"Cannot find broker symbol for '{logical}'.\n"
            f"  Tried explicit : {explicit}\n"
            f"  Tried candidates: {candidates}\n"
            f"  Fix: set '{logical}=YourBrokerName' in .env"
        )

    def resolve_all(self) -> dict:
        """Resolve all active symbols. Returns {logical: broker_name}."""
        logger.info("Resolving broker symbol names...")
        result = {}
        for sym in self.s.active_symbols:
            try:
                result[sym] = self.resolve(sym)
            except ValueError as e:
                logger.error(str(e))
        return result

    # ── Market data ───────────────────────────────────────────────────────────
    def get_bars(
        self,
        logical:   str,
        timeframe: str,
        n_bars:    int,
    ) -> pd.DataFrame:
        """
        Get the last `n_bars` completed OHLCV bars.
        Always excludes the currently-forming bar (index 0 is the forming bar
        in MT5, so we request n_bars+1 and drop index 0).
        """
        if not self.ensure_connection():
            return pd.DataFrame()
        broker = self.resolve(logical)
        tf_map = {
            "M15": mt5.TIMEFRAME_M15,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
        }
        tf = tf_map.get(timeframe)
        if tf is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        # pos=0 is the forming bar — request +1 and skip it
        rates = mt5.copy_rates_from_pos(broker, tf, 0, n_bars + 1)
        if rates is None or len(rates) < 2:
            logger.error(f"No data for {broker} {timeframe}: {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates[:-1])   # drop forming bar
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.index.name = "datetime"
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df[["open","high","low","close","volume","spread"]]

    def get_current_spread_pips(self, logical: str) -> float:
        """
        Return current spread in pips for the symbol.
        Uses live pip_size from SymbolSpecs if available, otherwise falls back to point * 10.
        """
        if not self.ensure_connection():
            return 999.0
        broker = self.resolve(logical)
        info   = mt5.symbol_info(broker)
        if info is None:
            return 999.0
            
        try:
            from live.symbol_specs import get_specs
            specs = get_specs()
            pip_size = specs.pip_size(logical)
        except Exception:
            pip_size = info.point * 10
            
        if pip_size > 0:
            conversion_ratio = pip_size / info.point
            return round(float(info.spread / conversion_ratio), 2)
        return float(info.spread)

    def get_symbol_info(self, logical: str) -> dict:
        """Return symbol metadata needed for position sizing."""
        if not self.ensure_connection():
            raise RuntimeError("MT5 is not connected")
        broker = self.resolve(logical)
        info   = mt5.symbol_info(broker)
        if info is None:
            raise RuntimeError(f"Cannot get symbol info for {broker}")
        return {
            "name":       broker,
            "digits":     info.digits,
            "point":      info.point,
            "trade_tick_size":  info.trade_tick_size,
            "trade_tick_value": info.trade_tick_value,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step":info.volume_step,
        }

    def normalise_lots(self, logical: str, lots: float) -> float:
        """Round lots to the broker's volume step and clamp to min/max."""
        if not self.ensure_connection():
            return max(self.s.min_lots, min(self.s.max_lots, lots))
        broker = self.resolve(logical)
        info   = mt5.symbol_info(broker)
        if info is None:
            return max(self.s.min_lots, min(self.s.max_lots, lots))
        step = info.volume_step
        lots = round(round(lots / step) * step, 8)
        lots = max(info.volume_min, min(info.volume_max, lots))
        lots = max(self.s.min_lots,  min(self.s.max_lots, lots))
        return lots

    def _pick_filling_mode(self, broker: str):
        """Select a broker-compatible filling mode (matches the reference script)."""
        if not self.ensure_connection():
            return getattr(mt5, "ORDER_FILLING_IOC", 1)

        info = mt5.symbol_info(broker)
        if info is None or not hasattr(info, "filling_mode"):
            return getattr(mt5, "ORDER_FILLING_IOC", 1)

        mode = int(info.filling_mode)

        # Match the reference bot's behavior:
        # 1 -> FOK, 2 -> IOC, otherwise RETURN
        if mode == 1:
            return getattr(mt5, "ORDER_FILLING_FOK", mode)
        elif mode == 2:
            return getattr(mt5, "ORDER_FILLING_IOC", mode)
        return getattr(mt5, "ORDER_FILLING_RETURN", mode)

    def _is_symbol_tradeable(self, broker: str) -> bool:
        if not self.ensure_connection():
            return False
        info = mt5.symbol_info(broker)
        if info is None:
            return False
        try:
            selected = mt5.symbol_select(broker, True)
        except Exception:
            selected = False
        return bool(selected)

    def _request_timeout_exceeded(self, started: float) -> bool:
        return (time.monotonic() - started) >= float(self.s.order_timeout_seconds)

    # ── Position queries ──────────────────────────────────────────────────────
    def get_positions(self) -> dict:
        """
        Return bot-managed positions keyed by logical symbol name.
        Returns {symbol: {"side": 1/-1, "lots": float, "ticket": int}}
        """
        result = {}
        if not self.ensure_connection():
            return result
        positions = mt5.positions_get()
        if positions is None:
            return result
        for pos in positions:
            if pos.magic != self.s.magic_number:
                continue
            # Find logical name for this broker symbol
            for logical, broker in self._resolved.items():
                if pos.symbol == broker:
                    result[logical] = {
                        "side":   1 if pos.type == mt5.ORDER_TYPE_BUY else -1,
                        "lots":   pos.volume,
                        "ticket": pos.ticket,
                        "profit": pos.profit,
                    }
        return result

    def has_position(self, logical: str) -> bool:
        return logical in self.get_positions()

    # ── Order execution ───────────────────────────────────────────────────────
    def open_position(self, logical: str, side: int, lots: float) -> Optional[int]:
        """
        Open a new position. side=+1 for long, -1 for short.
        Returns MT5 ticket number or None on failure.
        """
        if not self.ensure_connection():
            logger.error(f"[{logical}] MT5 is disconnected")
            return None
        broker   = self.resolve(logical)
        lots     = self.normalise_lots(logical, lots)
        order_type = mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL
        price      = mt5.symbol_info_tick(broker)

        if price is None:
            logger.error(f"Cannot get tick for {broker}")
            return None
        if not self._is_symbol_tradeable(broker):
            logger.error(f"[{logical}] Symbol {broker} is not tradeable right now")
            return None

        ask = price.ask
        bid = price.bid
        exec_price = ask if side == 1 else bid

        request = {
            "action":     mt5.TRADE_ACTION_DEAL,
            "symbol":     broker,
            "volume":     lots,
            "type":       order_type,
            "price":      exec_price,
            "deviation":  self.s.slippage_points,
            "magic":      self.s.magic_number,
            "comment":    self.s.order_comment,
            "type_time":  mt5.ORDER_TIME_GTC,
            "type_filling": self._pick_filling_mode(broker),
        }

        return self._send_order(request, logical, side, lots)

    def close_position(self, logical: str, ticket: int, lots: float, side: int) -> bool:
        """Close an existing position by ticket."""
        if not self.ensure_connection():
            logger.error(f"[{logical}] MT5 is disconnected")
            return False
        broker     = self.resolve(logical)
        close_type = mt5.ORDER_TYPE_SELL if side == 1 else mt5.ORDER_TYPE_BUY
        price      = mt5.symbol_info_tick(broker)

        if price is None:
            logger.error(f"Cannot get tick for {broker}")
            return False

        exec_price = price.bid if side == 1 else price.ask
        request = {
            "action":     mt5.TRADE_ACTION_DEAL,
            "symbol":     broker,
            "volume":     lots,
            "type":       close_type,
            "position":   ticket,
            "price":      exec_price,
            "deviation":  self.s.slippage_points,
            "magic":      self.s.magic_number,
            "comment":    f"{self.s.order_comment}_close",
            "type_time":  mt5.ORDER_TIME_GTC,
            "type_filling": self._pick_filling_mode(broker),
        }
        result = self._send_order(request, logical, -side, lots)
        return result is not None

    def flip_position(self, logical: str, new_side: int, new_lots: float) -> Optional[int]:
        """
        Reverse an existing position: close old + open new in one sequence.
        Returns the new ticket or None on failure.
        """
        positions = self.get_positions()
        if logical in positions:
            pos = positions[logical]
            logger.info(
                f"[{logical}] Flipping  "
                f"{'LONG' if pos['side']==1 else 'SHORT'} → "
                f"{'LONG' if new_side==1 else 'SHORT'}  "
                f"lots={new_lots:.3f}"
            )
            closed = self.close_position(logical, pos["ticket"], pos["lots"], pos["side"])
            if not closed:
                logger.error(f"[{logical}] Close failed — aborting flip to protect position")
                return None
            time.sleep(0.5)   # brief pause between close and open
        else:
            logger.info(f"[{logical}] Opening initial {'LONG' if new_side==1 else 'SHORT'}")

        return self.open_position(logical, new_side, new_lots)

    # ── Stop loss management ──────────────────────────────────────────────────
    def calc_sl_price(
        self,
        symbol:     str,
        side:       int,
        ref_price:  float,
        atr:        float,
        atr_mult:   float,
    ) -> float:
        """
        Calculate a stop-loss price.

        Parameters
        ----------
        side      : +1 = long (SL below price), -1 = short (SL above price)
        ref_price : price to measure distance from (entry or current)
        atr       : current ATR in price units
        atr_mult  : number of ATRs to place the SL away

        Returns
        -------
        SL price rounded to the broker's tick size.
        """
        broker   = self.resolve(symbol)
        info     = mt5.symbol_info(broker)
        tick_sz  = info.trade_tick_size if info else 0.0001
        distance = atr * atr_mult

        if side == 1:    # long — SL below entry
            raw = ref_price - distance
        else:             # short — SL above entry
            raw = ref_price + distance

        # Round to tick size
        if tick_sz > 0:
            raw = round(round(raw / tick_sz) * tick_sz, 8)
        return raw

    def set_stop_loss(
        self,
        logical: str,
        ticket:  int,
        sl_price: float,
    ) -> bool:
        """
        Set or update the stop-loss on an existing position.
        Never sets a TP (always-in — no fixed profit target).
        """
        if not self.ensure_connection():
            return False
        broker = self.resolve(logical)
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol":   broker,
            "sl":       sl_price,
            "tp":       0.0,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            logger.warning(
                f"[{logical}] SL update failed  ticket={ticket}  "
                f"sl={sl_price:.5f}  retcode={code}"
            )
            return False
        logger.debug(
            f"[{logical}] SL set  ticket={ticket}  sl={sl_price:.5f}"
        )
        return True

    def trail_stop_loss(
        self,
        logical:    str,
        ticket:     int,
        side:       int,
        atr:        float,
        atr_mult:   float,
    ) -> bool:
        """
        Trail the stop-loss to follow the current price.
        SL only moves in the profitable direction — never backwards.

        For LONG  : SL = max(current_SL, current_price − ATR×mult)
        For SHORT : SL = min(current_SL, current_price + ATR×mult)
        """
        if not self.ensure_connection():
            return False
        broker = self.resolve(logical)

        # Get current position state
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        pos = positions[0]

        current_price = pos.price_current
        current_sl    = pos.sl

        new_sl = self.calc_sl_price(logical, side, current_price, atr, atr_mult)

        # Only move SL in profitable direction
        if side == 1:    # long — only move SL up
            if new_sl <= current_sl:
                return True   # no update needed
        else:             # short — only move SL down
            if new_sl >= current_sl:
                return True   # no update needed

        return self.set_stop_loss(logical, ticket, new_sl)

    def get_position_sl(self, logical: str, ticket: int) -> float:
        """Return the current SL price for a position (0.0 if not set)."""
        if not self.ensure_connection():
            return 0.0
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return 0.0
        return float(positions[0].sl)

    def position_exists(self, logical: str, ticket: int) -> bool:
        """Check if a specific ticket is still open."""
        if not self.ensure_connection():
            return False
        positions = mt5.positions_get(ticket=ticket)
        return bool(positions)


    def get_tick(self, logical: str):
        """Return the latest tick for a logical symbol or None if unavailable."""
        if not self.ensure_connection():
            return None
        broker = self.resolve(logical)
        return mt5.symbol_info_tick(broker)

    def _send_order(
        self, request: dict, logical: str, side: int, lots: float
    ) -> Optional[int]:
        if not self.ensure_connection():
            logger.error(f"[{logical}] MT5 is disconnected")
            return None
        started = time.monotonic()
        for attempt in range(1, self.s.max_retries + 1):
            if self._request_timeout_exceeded(started):
                logger.error(f"[{logical}] Order timeout exceeded after {self.s.order_timeout_seconds}s")
                break
            result = mt5.order_send(request)
            if result is None:
                logger.error(f"[{logical}] order_send returned None: {mt5.last_error()}")
            elif result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.success(
                    f"[{logical}]  {'BUY' if side==1 else 'SELL'}  "
                    f"{lots:.3f} lots  "
                    f"ticket={result.order}  "
                    f"price={result.price:.5f}"
                )
                return result.order
            else:
                logger.warning(
                    f"[{logical}] Order attempt {attempt}/{self.s.max_retries} failed  "
                    f"retcode={result.retcode}  {result.comment}"
                )

            if attempt < self.s.max_retries and not self._request_timeout_exceeded(started):
                time.sleep(self.s.retry_delay_seconds)
                # Refresh price for next attempt
                tick = mt5.symbol_info_tick(request["symbol"])
                if tick:
                    request["price"] = tick.ask if side == 1 else tick.bid

        logger.error(f"[{logical}] Order failed after {self.s.max_retries} attempts")
        return None
