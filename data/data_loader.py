"""
data/data_loader.py  —  Load OHLCV data from Parquet or yfinance fallback
──────────────────────────────────────────────────────────────────────────────
Priority:
  1. data/raw/{symbol}_{timeframe}.parquet   (from mt5_fetcher.py — preferred)
  2. yfinance  GC=F / SI=F                   (fallback for Codespaces / Linux)

Usage:
    from data.data_loader import DataLoader
    loader = DataLoader()
    df = loader.load("GOLD", "H1")
    data = loader.load_all()          # → {"GOLD": df, "SILVER": df}
"""

import sys
from pathlib import Path
from loguru import logger

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, YF_SYMBOLS, SYMBOLS, TRAIN_START, TEST_END


# ── yfinance interval map ──────────────────────────────────────────────────────
YF_INTERVAL_MAP = {
    "M1":  "1m",
    "M5":  "5m",
    "M15": "15m",
    "H1":  "1h",
    "H4":  "1h",   # fetched as 1h then resampled → 4h
    "D1":  "1d",
}

# Max calendar days yfinance allows per interval
YF_MAX_DAYS = {
    "M1":  7,
    "M5":  60,
    "M15": 60,
    "H1":  730,
    "H4":  730,
    "D1":  None,
}


class DataLoader:
    """
    Unified data loader for GOLD and SILVER.

    Parameters
    ----------
    prefer_parquet : bool
        If True (default), try Parquet first.
        Set to False to force yfinance (useful for quick tests).
    """

    def __init__(self, prefer_parquet: bool = True):
        self.prefer_parquet = prefer_parquet

    # ── Public API ─────────────────────────────────────────────────────────────
    def load(
        self,
        symbol:    str,
        timeframe: str = "H1",
        date_from: str = TRAIN_START,
        date_to:   str = TEST_END,
    ) -> pd.DataFrame:
        """
        Load OHLCV for one symbol.

        Returns
        -------
        pd.DataFrame  —  columns: open, high, low, close, volume
                         index  : DatetimeIndex (UTC-aware)
        """
        if symbol not in SYMBOLS:
            raise ValueError(f"Unknown symbol '{symbol}'. Expected one of {SYMBOLS}.")

        if self.prefer_parquet:
            df = self._load_parquet(symbol, timeframe)
            if df is not None:
                return self._slice(df, date_from, date_to)

        logger.warning(
            f"[{symbol}] Parquet not found for {timeframe} — "
            f"falling back to yfinance ({YF_SYMBOLS[symbol]})"
        )
        return self._load_yfinance(symbol, timeframe, date_from, date_to)

    def load_all(
        self,
        symbols:   list = None,
        timeframe: str  = "H1",
        date_from: str  = TRAIN_START,
        date_to:   str  = TEST_END,
    ) -> dict:
        """
        Load all symbols. Returns { symbol: DataFrame }.
        """
        symbols = symbols or SYMBOLS
        data    = {}
        for sym in symbols:
            try:
                df = self.load(sym, timeframe, date_from, date_to)
                if df is not None and not df.empty:
                    data[sym] = df
                    logger.info(
                        f"[{sym}]  {len(df):,} bars  "
                        f"{df.index[0].date()} → {df.index[-1].date()}"
                    )
                else:
                    logger.warning(f"[{sym}] No data returned — skipping.")
            except Exception as e:
                logger.error(f"[{sym}] Load failed: {e}")
        return data

    # ── Parquet loader ─────────────────────────────────────────────────────────
    def _load_parquet(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        path = DATA_DIR / f"{symbol}_{timeframe}.parquet"
        if not path.exists():
            return None
        try:
            df = pq.read_table(path).to_pandas()
            if "datetime" in df.columns:
                df.set_index("datetime", inplace=True)
            df = self._standardise(df)
            logger.debug(f"[{symbol}] Loaded from Parquet: {path.name}")
            return df
        except Exception as e:
            logger.error(f"[{symbol}] Parquet read error ({path}): {e}")
            return None

    # ── yfinance loader ────────────────────────────────────────────────────────
    def _load_yfinance(
        self,
        symbol:    str,
        timeframe: str,
        date_from: str,
        date_to:   str,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed.  Run: pip install yfinance")

        yf_sym   = YF_SYMBOLS[symbol]
        interval = YF_INTERVAL_MAP[timeframe]
        max_days = YF_MAX_DAYS[timeframe]

        # yfinance caps intraday history — clamp start date if needed
        if max_days is not None:
            from datetime import datetime, timedelta
            from_dt = datetime.strptime(date_from, "%Y-%m-%d")
            to_dt   = datetime.strptime(date_to,   "%Y-%m-%d")
            if (to_dt - from_dt).days > max_days:
                adj_from = (to_dt - timedelta(days=max_days)).strftime("%Y-%m-%d")
                logger.warning(
                    f"[{symbol}] yfinance caps {interval} history at {max_days} days.  "
                    f"Adjusting start: {date_from} → {adj_from}.  "
                    f"Use MT5 fetcher on Windows for full history."
                )
                date_from = adj_from

        ticker = yf.Ticker(yf_sym)

        if timeframe == "H4":
            # yfinance has no native 4h — download 1h and resample
            raw = ticker.history(start=date_from, end=date_to,
                                 interval="1h", auto_adjust=True)
            df  = self._resample_to_4h(raw)
        else:
            df = ticker.history(start=date_from, end=date_to,
                                interval=interval, auto_adjust=True)

        if df is None or df.empty:
            raise ValueError(
                f"[{symbol}] yfinance returned no data for {yf_sym} ({interval}).  "
                f"Check the symbol and date range."
            )

        df = self._standardise(df)
        logger.info(f"[{symbol}] yfinance ({yf_sym}) {timeframe}: {len(df):,} bars")
        return df

    # ── Helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _standardise(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names, timezone, and sort."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={"vol": "volume", "tick_volume": "volume"}, inplace=True)

        keep = ["open", "high", "low", "close", "volume"]
        df   = df[[c for c in keep if c in df.columns]]

        # Ensure UTC-aware index
        if hasattr(df.index, "tz"):
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

        df.sort_index(inplace=True)
        df.dropna(subset=["close"], inplace=True)

        # Remove any weekend / all-zero rows (common in metals futures data)
        df = df[df["close"] > 0]
        return df

    @staticmethod
    def _resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
        df_1h = df_1h.copy()
        df_1h.columns = [c.lower() for c in df_1h.columns]
        agg = {
            c: ("first" if c == "open" else
                "max"   if c == "high" else
                "min"   if c == "low"  else
                "last"  if c == "close" else "sum")
            for c in ["open", "high", "low", "close", "volume"]
            if c in df_1h.columns
        }
        return df_1h.resample("4h").agg(agg).dropna(subset=["close"])

    @staticmethod
    def _slice(df: pd.DataFrame, date_from: str, date_to: str) -> pd.DataFrame:
        return df.loc[date_from:date_to]

    # ── Diagnostics ────────────────────────────────────────────────────────────
    def status(self):
        """Print which Parquet files are available for each symbol/timeframe."""
        # Only H1 and D1 are used in training — others are informational
        primary_tfs = ["H1", "D1"]
        other_tfs   = ["M1", "M5", "M15", "H4"]
        all_tfs     = primary_tfs + other_tfs

        print(f"\n{'Symbol':<10} {'TF':<6} {'Used?':^7} {'Parquet':^8}  {'Bars':>8}  Date range")
        print("─" * 65)
        for sym in SYMBOLS:
            for tf in all_tfs:
                path = DATA_DIR / f"{sym}_{tf}.parquet"
                used = "✓" if tf in primary_tfs else "–"
                if path.exists():
                    df   = pq.read_table(path).to_pandas()
                    rows = len(df)
                    try:
                        if "datetime" in df.columns:
                            df.set_index("datetime", inplace=True)
                        start = str(df.index[0])[:10]
                        end   = str(df.index[-1])[:10]
                        rng   = f"{start} → {end}"
                    except Exception:
                        rng = "?"
                    print(f"{sym:<10} {tf:<6} {used:^7} {'✓':^8}  {rows:>8,}  {rng}")
                else:
                    print(f"{sym:<10} {tf:<6} {used:^7} {'✗':^8}  {'—':>8}")
        print(f"\n  ✓ = used in training   – = fetched but not used\n")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test / status check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = DataLoader()
    loader.status()
    print("Loading H1 data ...")
    data = loader.load_all(timeframe="H1")
    for sym, df in data.items():
        print(f"  {sym}: {len(df):,} bars  close_last={df['close'].iloc[-1]:.4f}")
