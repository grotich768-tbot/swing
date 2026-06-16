"""
live/settings.py  —  Load and validate live trading configuration from .env
"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv
from loguru import logger

# Load .env from project root
_ENV_PATH = Path(__file__).parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
    logger.info(f"Loaded .env from {_ENV_PATH}")
else:
    logger.warning(
        f".env not found at {_ENV_PATH}. "
        f"Copy .env.example to .env and fill in your values."
    )


def _get(key: str, default=None, required=False):
    val = os.environ.get(key, default)
    if required and not val:
        raise ValueError(f"Required .env variable '{key}' is not set.")
    return val


def _getbool(key: str, default: bool = False) -> bool:
    return _get(key, str(default)).lower() in ("true", "1", "yes")


def _getfloat(key: str, default: float = 0.0) -> float:
    return float(_get(key, str(default)))


def _getint(key: str, default: int = 0) -> int:
    return int(_get(key, str(default)))


def _getlist(key: str, default: str = "") -> List[str]:
    raw = _get(key, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass
class LiveSettings:
    """All live trading settings loaded from .env"""

    # ── MT5 connection ─────────────────────────────────────────────────────────
    mt5_login:    Optional[int] = None
    mt5_password: Optional[str] = None
    mt5_server:   Optional[str] = None
    mt5_path:     Optional[str] = None

    # ── Symbol mapping ─────────────────────────────────────────────────────────
    symbol_prefix: str = ""
    symbol_suffix: str = ""

    # Explicit broker names (override auto-detection)
    gold_name:   str = "XAUUSD"
    silver_name: str = "XAGUSD"
    eurusd_name: str = "EURUSD"
    gbpusd_name: str = "GBPUSD"

    # Candidate fallback lists
    gold_candidates:   List[str] = field(default_factory=lambda: ["GOLD","XAUUSD","XAUUSDm","XAUUSDpro","XAUUSD."])
    silver_candidates: List[str] = field(default_factory=lambda: ["SILVER","XAGUSD","XAGUSDm","XAGUSD."])
    eurusd_candidates: List[str] = field(default_factory=lambda: ["EURUSD","EURUSDm","EURUSD."])
    gbpusd_candidates: List[str] = field(default_factory=lambda: ["GBPUSD","GBPUSDm","GBPUSD."])

    # ── Active symbols ─────────────────────────────────────────────────────────
    active_symbols: List[str] = field(default_factory=lambda: ["GOLD","SILVER","EURUSD","GBPUSD"])

    # ── Trading mode ───────────────────────────────────────────────────────────
    trading_mode: str = "DEMO"   # DEMO or LIVE

    # ── Position sizing ────────────────────────────────────────────────────────
    initial_balance: float = 10000.0
    risk_pct:        float = 0.02
    max_lots:        float = 0.5
    min_lots:        float = 0.01
    atr_stop_mult:   float = 1.5

    # ── Order settings ─────────────────────────────────────────────────────────
    magic_number:          int   = 241010
    order_comment:         str   = "AlwaysInBot_v3"
    slippage_points:       int   = 10
    order_timeout_seconds: int   = 30
    max_retries:           int   = 3
    retry_delay_seconds:   int   = 5

    # ── Session filter ─────────────────────────────────────────────────────────
    session_filter_enabled:    bool  = True
    session_full_size_start:   int   = 7
    session_full_size_end:     int   = 17
    session_reduced_multiplier: float = 0.5

    # ── Rollover guard ─────────────────────────────────────────────────────────
    rollover_guard_enabled: bool = True
    rollover_start_hour:    int  = 21
    rollover_start_min:     int  = 30
    rollover_end_hour:      int  = 22
    rollover_end_min:       int  = 5

    # ── Risk limits ────────────────────────────────────────────────────────────
    max_drawdown_pct:              float = 0.08
    circuit_breaker_recovery_pct:  float = 0.04
    max_daily_loss_usd:            float = 500.0
    max_spread_pips_gold:          float = 80.0
    max_spread_pips_silver:        float = 8.0
    max_spread_pips_fx:            float = 3.0

    # ── Warmup bars ────────────────────────────────────────────────────────────
    warmup_m15: int = 1500
    warmup_h1:  int = 600
    warmup_h4:  int = 200
    warmup_d1:  int = 100

    # ── Timing ─────────────────────────────────────────────────────────────────
    bar_check_interval_sec: int = 5
    exec_delay_sec:         int = 3

    # ── Telegram ───────────────────────────────────────────────────────────────
    telegram_enabled:               bool = False
    telegram_bot_token:             str  = ""
    telegram_chat_id:               str  = ""
    telegram_notify_on_flip:        bool = True
    telegram_notify_on_circuit:     bool = True
    telegram_notify_on_daily:       bool = True
    telegram_daily_summary_hour:    int  = 22

    # ── Stop loss / trailing stop ──────────────────────────────────────────────
    trail_stop_enabled:  bool  = True
    trail_stop_atr_mult: float = 5.0

    # ── Shutdown behaviour ─────────────────────────────────────────────────────
    close_on_stop:        bool  = True
    emergency_stop_pips:  float = 0.0

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level:   str  = "INFO"
    log_to_file: bool = True

    # ── Derived: symbol → broker name map ─────────────────────────────────────
    @property
    def symbol_name_map(self) -> dict:
        return {
            "GOLD":   self.gold_name,
            "SILVER": self.silver_name,
            "EURUSD": self.eurusd_name,
            "GBPUSD": self.gbpusd_name,
        }

    @property
    def symbol_candidates_map(self) -> dict:
        return {
            "GOLD":   self.gold_candidates,
            "SILVER": self.silver_candidates,
            "EURUSD": self.eurusd_candidates,
            "GBPUSD": self.gbpusd_candidates,
        }

    @property
    def is_live(self) -> bool:
        return self.trading_mode.upper() == "LIVE"

    def max_spread_pips(self, symbol: str) -> float:
        if symbol == "GOLD":   return self.max_spread_pips_gold
        if symbol == "SILVER": return self.max_spread_pips_silver
        return self.max_spread_pips_fx

    def validate(self):
        """Raise ValueError for any dangerous misconfiguration."""
        if self.trading_mode.upper() not in ("DEMO", "LIVE"):
            raise ValueError(f"TRADING_MODE must be DEMO or LIVE, got: {self.trading_mode}")
        if self.risk_pct > 0.05:
            raise ValueError(f"RISK_PCT={self.risk_pct:.1%} is dangerously high. Keep ≤ 5%.")
        if self.max_lots > 50.0:
            raise ValueError(f"MAX_LOTS={self.max_lots} is very large. Double-check.")
        if self.is_live and not self.mt5_login and not self.mt5_server:
            logger.warning("TRADING_MODE=LIVE but no MT5 credentials set. Ensure terminal is logged in.")
        if self.is_live:
            logger.warning(
                "\n" + "!"*60 +
                "\n  TRADING_MODE=LIVE — REAL MONEY WILL BE USED" +
                "\n" + "!"*60
            )


def load_settings(terminal_id: int = 1) -> LiveSettings:
    """Load all settings from environment variables (populated from .env)."""
    s = LiveSettings(
        # MT5
        mt5_login    = int(_get("MT5_LOGIN")) if _get("MT5_LOGIN") else None,
        mt5_password = _get("MT5_PASSWORD") or None,
        mt5_server   = _get("MT5_SERVER")   or None,
        mt5_path     = _get("MT5_PATH")     or None,

        # Symbol mapping
        symbol_prefix = _get("SYMBOL_PREFIX", ""),
        symbol_suffix = _get("SYMBOL_SUFFIX", ""),
        gold_name     = _get("GOLD",   "XAUUSD"),
        silver_name   = _get("SILVER", "XAGUSD"),
        eurusd_name   = _get("EURUSD", "EURUSD"),
        gbpusd_name   = _get("GBPUSD", "GBPUSD"),
        gold_candidates   = _getlist("GOLD_CANDIDATES",   "GOLD,XAUUSD,XAUUSDm,XAUUSDpro,XAUUSD."),
        silver_candidates = _getlist("SILVER_CANDIDATES", "SILVER,XAGUSD,XAGUSDm,XAGUSD."),
        eurusd_candidates = _getlist("EURUSD_CANDIDATES", "EURUSD,EURUSDm,EURUSD."),
        gbpusd_candidates = _getlist("GBPUSD_CANDIDATES", "GBPUSD,GBPUSDm,GBPUSD."),

        # Active
        active_symbols = _getlist("ACTIVE_SYMBOLS", "GOLD,SILVER,EURUSD,GBPUSD"),
        trading_mode   = _get("TRADING_MODE", "DEMO").upper(),

        # Sizing
        initial_balance = _getfloat("INITIAL_BALANCE", 10000.0),
        risk_pct        = _getfloat("RISK_PCT",        0.02),
        max_lots        = _getfloat("MAX_LOTS",        0.5),
        min_lots        = _getfloat("MIN_LOTS",        0.01),
        atr_stop_mult   = _getfloat("ATR_STOP_MULT",   1.5),

        # Orders
        magic_number          = _getint("MAGIC_NUMBER",          241010),
        order_comment         = _get("ORDER_COMMENT",            "AlwaysInBot_v3"),
        slippage_points       = _getint("SLIPPAGE_POINTS",       10),
        order_timeout_seconds = _getint("ORDER_TIMEOUT_SECONDS", 30),
        max_retries           = _getint("MAX_RETRIES",           3),
        retry_delay_seconds   = _getint("RETRY_DELAY_SECONDS",   5),

        # Session
        session_filter_enabled     = _getbool("SESSION_FILTER_ENABLED", True),
        session_full_size_start    = _getint("SESSION_FULL_SIZE_START",  7),
        session_full_size_end      = _getint("SESSION_FULL_SIZE_END",    17),
        session_reduced_multiplier = _getfloat("SESSION_REDUCED_MULTIPLIER", 0.5),

        # Rollover
        rollover_guard_enabled = _getbool("ROLLOVER_GUARD_ENABLED", True),
        rollover_start_hour    = _getint("ROLLOVER_START_HOUR", 21),
        rollover_start_min     = _getint("ROLLOVER_START_MIN",  30),
        rollover_end_hour      = _getint("ROLLOVER_END_HOUR",   22),
        rollover_end_min       = _getint("ROLLOVER_END_MIN",    5),

        # Risk
        max_drawdown_pct             = _getfloat("MAX_DRAWDOWN_PCT",             0.08),
        circuit_breaker_recovery_pct = _getfloat("CIRCUIT_BREAKER_RECOVERY_PCT", 0.04),
        max_daily_loss_usd           = _getfloat("MAX_DAILY_LOSS_USD",           500.0),
        max_spread_pips_gold         = _getfloat("MAX_SPREAD_PIPS_GOLD",         80.0),
        max_spread_pips_silver       = _getfloat("MAX_SPREAD_PIPS_SILVER",       8.0),
        max_spread_pips_fx           = _getfloat("MAX_SPREAD_PIPS_FX",           3.0),

        # Warmup
        warmup_m15 = _getint("WARMUP_BARS_M15", 1500),
        warmup_h1  = _getint("WARMUP_BARS_H1",  600),
        warmup_h4  = _getint("WARMUP_BARS_H4",  200),
        warmup_d1  = _getint("WARMUP_BARS_D1",  100),

        # Timing
        bar_check_interval_sec = _getint("BAR_CHECK_INTERVAL_SEC", 5),
        exec_delay_sec         = _getint("EXEC_DELAY_SEC",         3),

        # Telegram
        telegram_enabled            = _getbool("TELEGRAM_ENABLED",            False),
        telegram_bot_token          = _get("TELEGRAM_BOT_TOKEN",              ""),
        telegram_chat_id            = _get("TELEGRAM_CHAT_ID",               ""),
        telegram_notify_on_flip     = _getbool("TELEGRAM_NOTIFY_ON_FLIP",     True),
        telegram_notify_on_circuit  = _getbool("TELEGRAM_NOTIFY_ON_CIRCUIT_BREAKER", True),
        telegram_notify_on_daily    = _getbool("TELEGRAM_NOTIFY_ON_DAILY_SUMMARY",   True),
        telegram_daily_summary_hour = _getint("TELEGRAM_DAILY_SUMMARY_HOUR",  22),

        # Stop loss / trailing stop
        trail_stop_enabled  = _getbool("TRAIL_STOP_ENABLED",   True),
        trail_stop_atr_mult = _getfloat("TRAIL_STOP_ATR_MULT", 5.0),

        # Shutdown
        close_on_stop       = _getbool("CLOSE_ON_STOP",       True),
        emergency_stop_pips = _getfloat("EMERGENCY_STOP_PIPS", 0.0),

        # Logging
        log_level   = _get("LOG_LEVEL",   "INFO"),
        log_to_file = _getbool("LOG_TO_FILE", True),
    )

    # Apply prefix/suffix to explicit names if set
    if s.symbol_prefix or s.symbol_suffix:
        p, su = s.symbol_prefix, s.symbol_suffix
        s.gold_name   = f"{p}{s.gold_name}{su}"
        s.silver_name = f"{p}{s.silver_name}{su}"
        s.eurusd_name = f"{p}{s.eurusd_name}{su}"
        s.gbpusd_name = f"{p}{s.gbpusd_name}{su}"
        # Also prefix/suffix the candidates
        for attr in ("gold_candidates","silver_candidates","eurusd_candidates","gbpusd_candidates"):
            setattr(s, attr, [f"{p}{c}{su}" for c in getattr(s, attr)])

    s.validate()
    return s


# Singleton — import from anywhere
settings = load_settings()



