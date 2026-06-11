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
    usdjpy_name: str = "USDJPY"
    ethusd_name: str = "ETHUSD"
    btcusd_name: str = "BTCUSD"
    us500_name:  str = "US500Cash"
    us100_name:  str = "US100Cash"
    us30_name:   str = "US30Cash"
    uk100_name:  str = "UK100Cash"
    ger40_name:  str = "GER40Cash"
    aus200_name: str = "AUS200Cash"
    jp225_name:  str = "JP225Cash"

    # Candidate fallback lists
    gold_candidates:   List[str] = field(default_factory=lambda: ["GOLD","XAUUSD","XAUUSDm","XAUUSDpro","XAUUSD."])
    silver_candidates: List[str] = field(default_factory=lambda: ["SILVER","XAGUSD","XAGUSDm","XAGUSD."])
    eurusd_candidates: List[str] = field(default_factory=lambda: ["EURUSD","EURUSDm","EURUSD."])
    gbpusd_candidates: List[str] = field(default_factory=lambda: ["GBPUSD","GBPUSDm","GBPUSD."])
    usdjpy_candidates: List[str] = field(default_factory=lambda: ["USDJPY","USDJPYm","USDJPY."])
    ethusd_candidates: List[str] = field(default_factory=lambda: ["ETHUSD","ETHUSDm","ETHUSD."])
    btcusd_candidates: List[str] = field(default_factory=lambda: ["BTCUSD","BTCUSDm","BTCUSD.","XBTUSD"])
    us500_candidates:  List[str] = field(default_factory=lambda: ["US500Cash","US500m","US500."])
    us100_candidates:  List[str] = field(default_factory=lambda: ["US100Cash","US100m","US100."])
    us30_candidates:   List[str] = field(default_factory=lambda: ["US30Cash","US30m","US30."])
    uk100_candidates:  List[str] = field(default_factory=lambda: ["UK100Cash","UK100m","UK100."])
    ger40_candidates:  List[str] = field(default_factory=lambda: ["GER40Cash","GER40m","GER40."])
    aus200_candidates: List[str] = field(default_factory=lambda: ["AUS200Cash","AUS200m","AUS200."])
    jp225_candidates:  List[str] = field(default_factory=lambda: ["JP225Cash","JN225m","JN225."])

    # ── Active symbols ─────────────────────────────────────────────────────────
    active_symbols: List[str] = field(default_factory=lambda: ["GOLD","SILVER","EURUSD","GBPUSD","USDJPY","ETHUSD","BTCUSD"])

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
    order_comment:         str   = "AlwaysInBot_v1"
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
    cb_cooldown_minutes:           int   = 60    # min wait after CB fires before re-entry
    max_daily_loss_usd:            float = 500.0
    # Max spread thresholds — 3x typical broker spread
    # If current spread exceeds this → block flip (news event / thin market)
    # Eightcap typical spreads: GOLD=6, SILVER=10, EURUSD=1, BTCUSD=297.6
    max_spread_pips_gold:          float = 18.0    # 3x typical 6.0
    max_spread_pips_silver:        float = 30.0    # 3x typical 10.0
    max_spread_pips_fx:            float = 3.0     # 3x typical 1.0
    max_spread_pips_usdjpy:        float = 3.3     # 3x typical 1.1
    max_spread_pips_eth:           float = 150.0   # 3x typical 49.8
    max_spread_pips_btc:           float = 900.0   # 3x typical 297.6
    max_spread_pips_ltc:           float = 150.0   # 3x typical 50.0
    max_spread_pips_us30:          float = 12.0    # 3x typical 3.9
    max_spread_pips_us100:         float = 6.0     # 3x typical 1.95
    max_spread_pips_us500:         float = 2.0     # 3x typical 0.55
    max_spread_pips_uk100:         float = 5.0     # 3x typical 1.60
    max_spread_pips_aus200:        float = 17.0    # 3x typical 5.54
    max_spread_pips_ger40:         float = 6.0     # 3x typical 1.95
    max_spread_pips_jp225:         float = 24.0    # 3x typical 8.0

    # ── Warmup bars ────────────────────────────────────────────────────────────
    warmup_m15: int = 200
    warmup_h1:  int = 300
    warmup_h4:  int = 60
    warmup_d1:  int = 50

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
    log_rotation: str  = "1 week"

    # ── Terminal UI ────────────────────────────────────────────────────────────
    terminal_ui_enabled:         bool  = True
    terminal_ui_refresh_seconds:  float = 1.0
    terminal_ui_max_events:      int   = 12

    # ── Derived: symbol → broker name map ─────────────────────────────────────
    @property
    def symbol_name_map(self) -> dict:
        return {
            "GOLD":   self.gold_name,
            "SILVER": self.silver_name,
            "EURUSD": self.eurusd_name,
            "GBPUSD": self.gbpusd_name,
            "USDJPY": self.usdjpy_name,
            "ETHUSD": self.ethusd_name,
            "BTCUSD": self.btcusd_name,
            "US500":  self.us500_name,
            "US100":  self.us100_name,
            "US30":   self.us30_name,
            "UK100":  self.uk100_name,
            "GER40":  self.ger40_name,
            "AUS200": self.aus200_name,
            "JP225":  self.jp225_name,
        }

    @property
    def symbol_candidates_map(self) -> dict:
        return {
            "GOLD":   self.gold_candidates,
            "SILVER": self.silver_candidates,
            "EURUSD": self.eurusd_candidates,
            "GBPUSD": self.gbpusd_candidates,
            "USDJPY": self.usdjpy_candidates,
            "ETHUSD": self.ethusd_candidates,
            "BTCUSD": self.btcusd_candidates,
            "US500":  self.us500_candidates,
            "US100":  self.us100_candidates,
            "US30":   self.us30_candidates,
            "UK100":  self.uk100_candidates,
            "GER40":  self.ger40_candidates,
            "AUS200": self.aus200_candidates,
            "JP225":  self.jp225_candidates,
        }

    @property
    def is_live(self) -> bool:
        return self.trading_mode.upper() == "LIVE"

    def max_spread_pips(self, symbol: str) -> float:
        """
        Return max allowed spread in pips before flips are blocked.
        Set to ~3x typical broker spread per symbol.
        Uses MT5 live spread if available via SymbolSpecs.
        """
        # Try to get live typical spread from MT5 (3x multiplier)
        try:
            from live.symbol_specs import get_specs
            specs = get_specs()
            if specs.is_loaded(symbol):
                typical = specs._specs[symbol].get("spread", 0)
                if typical > 0:
                    return typical * 3.0
        except Exception:
            pass

        # Fallback to hardcoded values per symbol
        _map = {
            "GOLD":   self.max_spread_pips_gold,
            "SILVER": self.max_spread_pips_silver,
            "USDJPY": self.max_spread_pips_usdjpy,
            "ETHUSD": self.max_spread_pips_eth,
            "BTCUSD": self.max_spread_pips_btc,
            "LTCUSD": self.max_spread_pips_ltc,
            "US30":   self.max_spread_pips_us30,
            "US100":  self.max_spread_pips_us100,
            "US500":  self.max_spread_pips_us500,
            "UK100":  self.max_spread_pips_uk100,
            "AUS200": self.max_spread_pips_aus200,
            "GER40":  self.max_spread_pips_ger40,
            "JP225":  self.max_spread_pips_jp225,
        }
        return _map.get(symbol, self.max_spread_pips_fx)

    def validate(self):
        """Raise ValueError for any dangerous misconfiguration."""
        if self.trading_mode.upper() not in ("DEMO", "LIVE"):
            raise ValueError(f"TRADING_MODE must be DEMO or LIVE, got: {self.trading_mode}")
        if self.risk_pct > 0.20:
            raise ValueError(f"RISK_PCT={self.risk_pct:.1%} is dangerously high. Keep ≤ 20%.")
        if self.risk_pct > 0.05:
            logger.warning(f"RISK_PCT={self.risk_pct:.1%} is above 5% — ensure this is intentional.")
        if self.max_lots > 50.0:
            raise ValueError(f"MAX_LOTS={self.max_lots} is very large. Double-check.")
        if self.is_live and not self.mt5_login and not self.mt5_server:
            logger.warning("TRADING_MODE=LIVE but no MT5 credentials set. Ensure terminal is logged in.")
        if self.order_timeout_seconds <= 0:
            raise ValueError("ORDER_TIMEOUT_SECONDS must be > 0")
        if self.max_retries < 1:
            raise ValueError("MAX_RETRIES must be at least 1")
        if self.is_live:
            logger.warning(
                "\n" + "!"*60 +
                "\n  TRADING_MODE=LIVE — REAL MONEY WILL BE USED" +
                "\n" + "!"*60
            )


def load_settings(terminal_id: int = 1) -> LiveSettings:
    """Load all settings from environment variables (populated from .env)."""
    
    # If running terminal 2, use the '2' suffix for MT5 credentials and symbols
    suffix = "2" if terminal_id == 2 else ""
    
    s = LiveSettings(
        # MT5
        mt5_login    = int(_get(f"MT5_LOGIN{suffix}")) if _get(f"MT5_LOGIN{suffix}") else None,
        mt5_password = _get(f"MT5_PASSWORD{suffix}") or None,
        mt5_server   = _get(f"MT5_SERVER{suffix}")   or None,
        mt5_path     = _get(f"MT5_PATH{suffix}")     or None,

        # Symbol mapping
        symbol_prefix = _get("SYMBOL_PREFIX", ""),
        symbol_suffix = _get("SYMBOL_SUFFIX", ""),
        gold_name     = _get("GOLD",   "XAUUSD"),
        silver_name   = _get("SILVER", "XAGUSD"),
        eurusd_name   = _get("EURUSD", "EURUSD"),
        gbpusd_name   = _get("GBPUSD", "GBPUSD"),
        usdjpy_name   = _get("USDJPY", "USDJPY"),
        ethusd_name   = _get("ETHUSD", "ETHUSD"),
        btcusd_name   = _get("BTCUSD", "BTCUSD"),
        us500_name    = _get("US500",  "US500Cash"),
        us100_name    = _get("US100",  "US100Cash"),
        us30_name     = _get("US30",   "US30Cash"),
        uk100_name    = _get("UK100",  "UK100Cash"),
        ger40_name    = _get("GER40",  "GER40Cash"),
        aus200_name   = _get("AUS200", "AUS200Cash"),
        jp225_name    = _get("JP225",  "JP225Cash"),
        gold_candidates   = _getlist("GOLD_CANDIDATES",   "GOLD,XAUUSD,XAUUSDm,XAUUSDpro,XAUUSD.,GOLD.vx,XAUUSD.vx"),
        silver_candidates = _getlist("SILVER_CANDIDATES", "SILVER,XAGUSD,XAGUSDm,XAGUSD.,SILVER.vx,XAGUSD.vx"),
        eurusd_candidates = _getlist("EURUSD_CANDIDATES", "EURUSD,EURUSDm,EURUSD.,EURUSD.vx"),
        gbpusd_candidates = _getlist("GBPUSD_CANDIDATES", "GBPUSD,GBPUSDm,GBPUSD.,GBPUSD.vx"),
        usdjpy_candidates = _getlist("USDJPY_CANDIDATES", "USDJPY,USDJPYm,USDJPY.,USDJPY.vx"),
        ethusd_candidates = _getlist("ETHUSD_CANDIDATES", "ETHUSD,ETHUSDm,ETHUSD.,ETHUSD.vx"),
        btcusd_candidates = _getlist("BTCUSD_CANDIDATES", "BTCUSD,BTCUSDm,BTCUSD.,XBTUSD,BTCUSD.vx"),
        us500_candidates  = _getlist("US500_CANDIDATES",  "US500Cash,US500m,US500.,US500.vx"),
        us100_candidates  = _getlist("US100_CANDIDATES",  "US100Cash,US100m,US100.,US100.vx"),
        us30_candidates   = _getlist("US30_CANDIDATES",   "US30Cash,US30m,US30.,US30.vx"),
        uk100_candidates  = _getlist("UK100_CANDIDATES",  "UK100Cash,UK100m,UK100.,UK100.vx"),
        ger40_candidates  = _getlist("GER40_CANDIDATES",  "GER40Cash,GER40m,GER40.,GER40.vx"),
        aus200_candidates = _getlist("AUS200_CANDIDATES", "AUS200Cash,AUS200m,AUS200.,AUS200.vx"),
        jp225_candidates  = _getlist("JP225_CANDIDATES",  "JP225Cash,JN225m,JN225.,JP225.vx"),

        # Active
        active_symbols = _getlist(f"ACTIVE_SYMBOLS{suffix}", "GOLD,SILVER,EURUSD,GBPUSD,USDJPY,ETHUSD,BTCUSD"),
        trading_mode   = _get("TRADING_MODE", "DEMO").upper(),

        # Sizing
        initial_balance = _getfloat("INITIAL_BALANCE", 10000.0),
        risk_pct        = _getfloat("RISK_PCT",        0.02),
        max_lots        = _getfloat("MAX_LOTS",        0.5),
        min_lots        = _getfloat("MIN_LOTS",        0.01),
        atr_stop_mult   = _getfloat("ATR_STOP_MULT",   1.5),

        # Orders
        magic_number          = _getint("MAGIC_NUMBER",          241010),
        order_comment         = _get("ORDER_COMMENT",            "AlwaysInBot_v1"),
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
        cb_cooldown_minutes          = _getint("CB_COOLDOWN_MINUTES",            60),
        max_daily_loss_usd           = _getfloat("MAX_DAILY_LOSS_USD",           500.0),
        max_spread_pips_gold         = _getfloat("MAX_SPREAD_PIPS_GOLD",         80.0),
        max_spread_pips_silver       = _getfloat("MAX_SPREAD_PIPS_SILVER",       8.0),
        max_spread_pips_fx           = _getfloat("MAX_SPREAD_PIPS_FX",           3.0),
        max_spread_pips_eth          = _getfloat("MAX_SPREAD_PIPS_ETH",          100.0),
        max_spread_pips_btc          = _getfloat("MAX_SPREAD_PIPS_BTC",          400.0),

        # Warmup
        warmup_m15 = _getint("WARMUP_BARS_M15", 200),
        warmup_h1  = _getint("WARMUP_BARS_H1",  300),
        warmup_h4  = _getint("WARMUP_BARS_H4",  60),
        warmup_d1  = _getint("WARMUP_BARS_D1",  50),

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
        log_rotation = _get("LOG_ROTATION", "1 week"),

        # Terminal UI
        terminal_ui_enabled        = _getbool("TERMINAL_UI_ENABLED", True),
        terminal_ui_refresh_seconds = _getfloat("TERMINAL_UI_REFRESH_SECONDS", 1.0),
        terminal_ui_max_events      = _getint("TERMINAL_UI_MAX_EVENTS", 12),
    )

    # Apply prefix/suffix to explicit names if set
    if s.symbol_prefix or s.symbol_suffix:
        p, su = s.symbol_prefix, s.symbol_suffix
        s.gold_name   = f"{p}{s.gold_name}{su}"
        s.silver_name = f"{p}{s.silver_name}{su}"
        s.eurusd_name = f"{p}{s.eurusd_name}{su}"
        s.gbpusd_name = f"{p}{s.gbpusd_name}{su}"
        s.usdjpy_name = f"{p}{s.usdjpy_name}{su}"
        s.ethusd_name = f"{p}{s.ethusd_name}{su}"
        s.btcusd_name = f"{p}{s.btcusd_name}{su}"
        s.us500_name  = f"{p}{s.us500_name}{su}"
        s.us100_name  = f"{p}{s.us100_name}{su}"
        s.us30_name   = f"{p}{s.us30_name}{su}"
        s.uk100_name  = f"{p}{s.uk100_name}{su}"
        s.ger40_name  = f"{p}{s.ger40_name}{su}"
        s.aus200_name = f"{p}{s.aus200_name}{su}"
        s.jp225_name  = f"{p}{s.jp225_name}{su}"
        # Also prefix/suffix the candidates
        for attr in (
            "gold_candidates",
            "silver_candidates",
            "eurusd_candidates",
            "gbpusd_candidates",
            "usdjpy_candidates",
            "ethusd_candidates",
            "btcusd_candidates",
            "us500_candidates",
            "us100_candidates",
            "us30_candidates",
            "uk100_candidates",
            "ger40_candidates",
            "aus200_candidates",
            "jp225_candidates",
        ):
            setattr(s, attr, [f"{p}{c}{su}" for c in getattr(s, attr)])

    s.validate()
    return s


# Singleton — import from anywhere
settings = load_settings()
