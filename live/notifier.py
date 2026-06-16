"""
live/notifier.py  —  Telegram notifications for live trading events
──────────────────────────────────────────────────────────────────────────────
Sends alerts for: flips, circuit breakers, daily summary, errors.
All methods are fire-and-forget — failures are logged but never crash the bot.

Enable in .env:
    TELEGRAM_ENABLED=true
    TELEGRAM_BOT_TOKEN=your_token_from_BotFather
    TELEGRAM_CHAT_ID=your_chat_id_from_userinfobot
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from live.settings import LiveSettings


class Notifier:
    """
    Sends Telegram messages for key trading events.
    Safe to use even when Telegram is disabled — all methods become no-ops.
    """

    def __init__(self, settings: LiveSettings):
        self.s       = settings
        self._bot    = None
        self._chat   = settings.telegram_chat_id

        if settings.telegram_enabled:
            self._init_bot()

    def _init_bot(self):
        try:
            import requests
            self._requests = requests
            # Verify the token works
            url  = f"https://api.telegram.org/bot{self.s.telegram_bot_token}/getMe"
            resp = requests.get(url, timeout=5)
            if resp.ok:
                name = resp.json().get("result", {}).get("username", "?")
                logger.success(f"Telegram connected — @{name}")
            else:
                logger.warning(f"Telegram token invalid: {resp.text}")
                self.s.telegram_enabled = False
        except Exception as e:
            logger.warning(f"Telegram init failed: {e} — notifications disabled")
            self.s.telegram_enabled = False

    # ── Public notifications ───────────────────────────────────────────────────
    def flip(
        self,
        symbol:   str,
        old_side: int,
        new_side: int,
        lots:     float,
        price:    float,
        balance:  float,
    ):
        if not self.s.telegram_enabled or not self.s.telegram_notify_on_flip:
            return
        old = "LONG ▲" if old_side ==  1 else "SHORT ▼"
        new = "LONG ▲" if new_side ==  1 else "SHORT ▼"
        emoji = "🔄"
        msg = (
            f"{emoji} <b>FLIP  —  {symbol}</b>\n"
            f"  {old}  →  {new}\n"
            f"  Lots   : {lots:.3f}\n"
            f"  Price  : {price:.5f}\n"
            f"  Balance: ${balance:,.2f}\n"
            f"  <i>{self._ts()}</i>"
        )
        self._send(msg)

    def circuit_breaker(self, symbol: str, drawdown: float, frozen: bool):
        if not self.s.telegram_enabled or not self.s.telegram_notify_on_circuit:
            return
        emoji = "🔴" if frozen else "🟢"
        state = "OPEN (flips blocked)" if frozen else "CLOSED (trading resumed)"
        msg = (
            f"{emoji} <b>Circuit Breaker  —  {symbol}</b>\n"
            f"  State    : {state}\n"
            f"  Drawdown : {drawdown:.2%}\n"
            f"  <i>{self._ts()}</i>"
        )
        self._send(msg)

    def daily_summary(self, stats: dict):
        if not self.s.telegram_enabled or not self.s.telegram_notify_on_daily:
            return
        lines = [f"📊 <b>Daily Summary  —  {self._ts(date_only=True)}</b>\n"]
        for sym, m in stats.items():
            pnl   = m.get("daily_pnl", 0)
            flips = m.get("n_flips", 0)
            side  = "LONG ▲" if m.get("position", 1) == 1 else "SHORT ▼"
            sign  = "+" if pnl >= 0 else ""
            lines.append(
                f"  <b>{sym}</b>: {sign}${pnl:,.2f}  "
                f"|  {flips} flips  |  now {side}"
            )
        lines.append(f"\n  Balance: ${stats.get('_balance', 0):,.2f}")
        self._send("\n".join(lines))

    def error(self, message: str):
        if not self.s.telegram_enabled:
            return
        self._send(f"⚠️ <b>Error</b>\n{message}\n<i>{self._ts()}</i>")

    def startup(self, mode: str, symbols: list, balance: float):
        if not self.s.telegram_enabled:
            return
        emoji = "🔴 LIVE" if mode == "LIVE" else "🟡 DEMO"
        syms  = ", ".join(symbols)
        msg = (
            f"🤖 <b>Always-In Bot Started</b>\n"
            f"  Mode    : {emoji}\n"
            f"  Symbols : {syms}\n"
            f"  Balance : ${balance:,.2f}\n"
            f"  <i>{self._ts()}</i>"
        )
        self._send(msg)

    def shutdown(self, reason: str = ""):
        if not self.s.telegram_enabled:
            return
        self._send(
            f"🛑 <b>Bot Stopped</b>  {reason}\n<i>{self._ts()}</i>"
        )

    # ── Internal ───────────────────────────────────────────────────────────────
    def _send(self, text: str):
        if not self.s.telegram_enabled:
            return
        try:
            url  = f"https://api.telegram.org/bot{self.s.telegram_bot_token}/sendMessage"
            data = {
                "chat_id":    self._chat,
                "text":       text,
                "parse_mode": "HTML",
            }
            resp = self._requests.post(url, data=data, timeout=10)
            if not resp.ok:
                logger.warning(f"Telegram send failed: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Telegram send error: {e}")

    @staticmethod
    def _ts(date_only: bool = False) -> str:
        now = datetime.now(tz=timezone.utc)
        if date_only:
            return now.strftime("%Y-%m-%d")
        return now.strftime("%Y-%m-%d %H:%M:%S UTC")
