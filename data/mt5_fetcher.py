"""
data/mt5_fetcher.py  —  Fetch OHLCV data from an open MetaTrader 5 terminal
──────────────────────────────────────────────────────────────────────────────
Run with NO arguments and it will:
  1. Connect to whichever MT5 terminal is already open and logged in
  2. Auto-detect the correct broker symbol name for GOLD and SILVER
     (tries: GOLD / XAUUSD / XAUUSDm / XAUUSD. and SILVER / XAGUSD / XAGUSDm / XAGUSD.)
  3. Download all standard timeframes from TRAIN_START → TEST_END
  4. Save Parquet files to data/raw/

Usage
─────
  # Auto-detect everything from the open terminal (recommended)
  python -m data.mt5_fetcher

  # Custom date range
  python -m data.mt5_fetcher --from 2020-01-01 --to 2024-12-31

  # Single symbol / timeframe
  python -m data.mt5_fetcher --symbol GOLD --tf H1

  # Specific terminal exe (if you have multiple MT5 installations)
  python -m data.mt5_fetcher --path "C:/Program Files/MetaTrader 5/terminal64.exe"

  # Explicit login (only if the terminal is NOT already logged in)
  python -m data.mt5_fetcher --login 12345 --password yourpass --server BrokerName-Live
"""

import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

# All four timeframes used in training
DEFAULT_TIMEFRAMES = ["M15", "H1", "H4", "D1"]
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MT5_SYMBOLS, SYMBOLS, DATA_DIR


# ── Timeframe string → MT5 constant ──────────────────────────────────────────
def _tf_map() -> dict:
    if not MT5_AVAILABLE:
        return {"M1": 1, "M5": 5, "M15": 15, "H1": 16385, "H4": 16388, "D1": 16408}
    return {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }


class MT5Fetcher:
    """
    Connects to an already-open MT5 terminal and downloads OHLCV data.

    No credentials needed if the terminal is already logged in.
    Just run:  python -m data.mt5_fetcher
    """

    def __init__(
        self,
        path:     str = None,   # Optional path to terminal64.exe
        login:    int = None,   # Optional — only if not already logged in
        password: str = None,
        server:   str = None,
    ):
        if not MT5_AVAILABLE:
            raise EnvironmentError(
                "\n"
                "  MetaTrader5 package not found.\n"
                "  Install it with:  pip install MetaTrader5\n"
                "  ⚠  This only works on Windows with MT5 installed.\n"
                "  On Linux/Codespaces the training pipeline uses yfinance instead.\n"
            )
        self.path     = path
        self.login    = login
        self.password = password
        self.server   = server
        self._resolved: dict = {}   # logical name → actual broker symbol name

    # ── Connection ────────────────────────────────────────────────────────────
    def connect(self) -> bool:
        """
        Attach to the open MT5 terminal.
        Works without any arguments if MT5 is already open and logged in.
        """
        kwargs = {}
        if self.path:
            kwargs["path"] = self.path

        logger.info("Connecting to MetaTrader 5 terminal...")

        if not mt5.initialize(**kwargs):
            err = mt5.last_error()
            logger.error(
                f"mt5.initialize() failed: {err}\n"
                "  → Make sure MetaTrader 5 is open and you are logged in,\n"
                "    then run the fetcher again."
            )
            return False

        # Log in only if credentials were explicitly provided
        if self.login:
            ok = mt5.login(self.login, password=self.password, server=self.server)
            if not ok:
                logger.error(f"mt5.login() failed: {mt5.last_error()}")
                mt5.shutdown()
                return False

        info    = mt5.terminal_info()
        account = mt5.account_info()

        if info is None:
            logger.error("Could not read terminal info — is MT5 running?")
            return False

        acct_str = (
            f"  account : {account.login}\n"
            f"  broker  : {account.company}\n"
            f"  server  : {account.server}"
        ) if account else "  (no account info)"

        logger.success(
            f"Connected to MT5\n"
            f"  build   : {info.build}\n"
            f"  path    : {info.path}\n"
            + acct_str
        )
        return True

    def disconnect(self):
        mt5.shutdown()
        logger.info("MT5 disconnected.")

    # ── Symbol resolution ─────────────────────────────────────────────────────
    def resolve_symbol(self, logical_name: str) -> str:
        """
        Find which name this broker uses for a logical symbol like "GOLD".

        Tries every candidate in MT5_SYMBOLS[logical_name] against the
        terminal's live symbol list and returns the first match.
        """
        if logical_name in self._resolved:
            return self._resolved[logical_name]

        candidates = MT5_SYMBOLS.get(logical_name)
        if not candidates:
            raise ValueError(
                f"No candidates configured for '{logical_name}'. "
                f"Add them to MT5_SYMBOLS in config.py"
            )

        # Pull the full symbol list from the open terminal
        all_syms = mt5.symbols_get()
        if all_syms is None:
            raise RuntimeError("Could not retrieve symbol list from MT5.")
        all_names = {s.name for s in all_syms}

        for candidate in candidates:
            if candidate in all_names:
                # Make it visible in Market Watch so history is available
                mt5.symbol_select(candidate, True)
                self._resolved[logical_name] = candidate
                logger.info(f"  {logical_name:8s}  →  {candidate}")
                return candidate

        raise ValueError(
            f"None of the candidates for '{logical_name}' exist in this terminal.\n"
            f"  Tried   : {candidates}\n"
            f"  Fix     : Add the correct broker name to MT5_SYMBOLS in config.py\n"
            f"  Hint    : In MT5 open View → Symbols and search for 'gold' or 'xau'"
        )

    def resolve_all(self, symbols: list) -> dict:
        """Resolve a list of logical names. Returns {logical_name: broker_name}."""
        logger.info("Resolving broker symbol names ...")
        resolved, skipped = {}, []
        for sym in symbols:
            try:
                resolved[sym] = self.resolve_symbol(sym)
            except ValueError as e:
                logger.warning(str(e))
                skipped.append(sym)
        if skipped:
            logger.warning(f"Skipping unresolved symbols: {skipped}")
        return resolved

    # ── Single fetch ──────────────────────────────────────────────────────────
    def fetch(
        self,
        logical_name: str,
        timeframe:    str,
        date_from:    str = None,   # None → all available history from 2000
        date_to:      str = None,   # None → up to today
    ) -> pd.DataFrame:
        """
        Download OHLCV bars for one symbol / timeframe.

        Pass date_from=None (the default) to request every bar the broker
        has stored — MT5 returns as far back as its history goes.
        """
        broker_sym = self.resolve_symbol(logical_name)
        tf_const   = _tf_map()[timeframe]

        from_dt = (
            datetime(2000, 1, 1, tzinfo=timezone.utc)
            if date_from is None
            else datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        )
        to_dt = (
            datetime.now(tz=timezone.utc)
            if date_to is None
            else datetime.strptime(date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        )

        rates = mt5.copy_rates_range(broker_sym, tf_const, from_dt, to_dt)

        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            logger.warning(
                f"  ✗ {logical_name} ({broker_sym}) {timeframe}  —  no data returned\n"
                f"    MT5 error : {err}\n"
                f"    Tip       : In MT5 right-click the symbol → History Centre\n"
                f"                and download the missing history."
            )
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.index.name = "datetime"
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        keep = ["open", "high", "low", "close", "volume", "spread"]
        df = df[[c for c in keep if c in df.columns]]

        logger.info(
            f"  ✓ {logical_name:8s} ({broker_sym}) {timeframe:4s}  "
            f"{len(df):7,} bars  "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )
        return df

    # ── Save ──────────────────────────────────────────────────────────────────
    def save(self, df: pd.DataFrame, symbol: str, timeframe: str):
        path  = DATA_DIR / f"{symbol}_{timeframe}.parquet"
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path, compression="snappy")
        size  = path.stat().st_size / 1024
        logger.info(f"    Saved → {path.name}  ({size:.1f} KB)")

    # ── Batch fetch ───────────────────────────────────────────────────────────
    def fetch_all(
        self,
        symbols:    list = None,
        timeframes: list = None,   # None → H1 + D1 only (what training uses)
        date_from:  str  = None,   # None → all available broker history
        date_to:    str  = None,   # None → up to today
    ):
        """
        Fetch every symbol × timeframe and save Parquet files.

        Called with no arguments it:
          - fetches GOLD, SILVER, EURUSD, GBPUSD
          - downloads H1 and D1 only (the timeframes training actually uses)
          - requests ALL history the broker has stored (back to year 2000)
        """
        symbols    = symbols    or SYMBOLS
        timeframes = timeframes or DEFAULT_TIMEFRAMES

        if not self.connect():
            raise ConnectionError(
                "Could not connect to MetaTrader 5.\n"
                "Make sure MT5 is open and logged in, then run the fetcher again."
            )

        try:
            resolved = self.resolve_all(symbols)
            if not resolved:
                raise ValueError(
                    "No symbols could be resolved.\n"
                    "Check MT5_SYMBOLS in config.py and make sure the symbols\n"
                    "exist in your broker's terminal."
                )

            range_str = (
                f"{date_from}  →  {date_to if date_to else 'today'}"
                if date_from
                else "ALL available history (back to 2000)"
            )
            logger.info(
                f"\nFetching {len(resolved)} symbol(s)  ×  "
                f"{len(timeframes)} timeframe(s)\n"
                f"Date range : {range_str}\n"
            )

            saved, failed = 0, 0
            for sym in resolved:
                logger.info(f"\n── {sym} ──────────────────────────────────")
                for tf in timeframes:
                    try:
                        df = self.fetch(sym, tf, date_from, date_to)
                        if not df.empty:
                            self.save(df, sym, tf)
                            saved += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"  ✗ {sym} {tf}: {e}")
                        failed += 1

            logger.success(
                f"\n{'='*55}\n"
                f"  Fetch complete\n"
                f"  Saved   : {saved} file(s)  →  {DATA_DIR}\n"
                f"  Failed  : {failed}"
                + (" (see warnings above)" if failed else "")
                + f"\n{'='*55}\n"
                f"  Next step: python train.py"
            )

        finally:
            self.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch GOLD & SILVER OHLCV data from an open MetaTrader 5 terminal.\n"
            "Run with no arguments to auto-connect to the open terminal."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
  python -m data.mt5_fetcher
  python -m data.mt5_fetcher --from 2020-01-01 --to 2024-12-31
  python -m data.mt5_fetcher --symbol GOLD --tf H1
  python -m data.mt5_fetcher --path "C:/MT5/terminal64.exe"
  python -m data.mt5_fetcher --login 123456 --password pass --server Broker-Live
        """,
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help=f"Specific symbol to fetch. Default: all ({SYMBOLS})"
    )
    parser.add_argument(
        "--tf", type=str, default=None,
        help="Specific timeframe (M1/M5/M15/H1/H4/D1). Default: all"
    )
    parser.add_argument(
        "--from", dest="date_from", type=str, default=None,
        help="Start date YYYY-MM-DD (default: all available broker history)"
    )
    parser.add_argument(
        "--to", dest="date_to", type=str, default=None,
        help="End date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--path", type=str, default=None,
        help="Path to terminal64.exe (optional; only needed with multiple MT5 installs)"
    )
    parser.add_argument(
        "--login", type=int, default=None,
        help="MT5 account number (only needed if the terminal is NOT already logged in)"
    )
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument(
        "--server", type=str, default=None,
        help="Broker server name e.g. ICMarkets-Live01"
    )
    args = parser.parse_args()

    fetcher = MT5Fetcher(
        path     = args.path,
        login    = args.login,
        password = args.password,
        server   = args.server,
    )

    fetcher.fetch_all(
        symbols    = [args.symbol]    if args.symbol else None,
        timeframes = [args.tf]        if args.tf     else None,
        date_from  = args.date_from,
        date_to    = args.date_to,
    )


if __name__ == "__main__":
    main()
