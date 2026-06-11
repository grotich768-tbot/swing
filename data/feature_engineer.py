"""
data/feature_engineer.py  —  Full multi-timeframe feature engineering
──────────────────────────────────────────────────────────────────────────────
IMPROVEMENTS APPLIED (vs original):

Tier 3 — New features added
  • VWAP deviation  (h1_vwap_dev, vwap_dist_atr)   — GOLD respects VWAP strongly
  • Tick volume imbalance proxy  (vol_imbalance)     — directional volume signal
  • Spread dynamics  (hl_spread_ratio)               — volatility shock detector
  • H4 regime gate  (h4_regime_signal)               — primary decision gate per
                                                       the Tier 1 improvement plan
  • Market Profile  (value_area_position)            — is price in or out of VA?

Original feature count:  53 base
New feature count:       59 base  (+6)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURE_LAGS, RSI_PERIOD, ATR_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, EMA_PERIODS, FEATURE_WINDOW,
    DIRECTION_HORIZON,
    USE_VWAP_FEATURES, USE_TICK_VOLUME_IMBALANCE, USE_SPREAD_DYNAMICS,
)


class FeatureEngineer:
    """
    Transforms raw OHLCV DataFrames into a normalised multi-timeframe
    feature matrix aligned to H1 bars.
    """

    def __init__(self, normalise: bool = True, window: int = FEATURE_WINDOW):
        self.normalise = normalise
        self.window    = window

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """H1-only feature build (fallback when other TFs are unavailable)."""
        df    = df.copy()
        feats = pd.DataFrame(index=df.index)

        feats = self._h1_features(feats, df)
        feats["_close"] = df["close"]
        feats["_atr"]   = self._true_range(df).rolling(ATR_PERIOD).mean()
        feats.dropna(inplace=True)
        feats["_target_direction"] = self._direction_label(df, DIRECTION_HORIZON)

        if self.normalise:
            feats = self._rolling_zscore(feats)
        return feats

    def transform_multi_tf(
        self,
        df_h1:  pd.DataFrame,
        df_d1:  pd.DataFrame = None,
        df_h4:  pd.DataFrame = None,
        df_m15: pd.DataFrame = None,
        symbol: str = "",
    ) -> pd.DataFrame:
        """
        Build the full multi-timeframe feature matrix aligned to H1 bars.

        Layer order:
          M15  →  entry refinement  (most recent M15 bar before each H1 close)
          H1   →  primary features
          H4   →  setup context + H4 regime gate  [IMPROVED]
          D1   →  daily bias
        """
        # ── H1 base features ──────────────────────────────────────────────────
        h1_feats = pd.DataFrame(index=df_h1.index)
        h1_feats = self._h1_features(h1_feats, df_h1)
        h1_feats["_close"] = df_h1["close"]
        h1_feats["_atr"]   = self._true_range(df_h1).rolling(ATR_PERIOD).mean()
        h1_feats.dropna(inplace=True)
        h1_feats["_target_direction"] = self._direction_label(df_h1, DIRECTION_HORIZON)

        parts = [h1_feats]
        loaded = ["H1"]

        # ── M15 entry-refinement features ─────────────────────────────────────
        if df_m15 is not None and len(df_m15) >= 50:
            m15_ctx = self._build_m15_context(df_m15)
            m15_aln = m15_ctx.reindex(h1_feats.index, method="ffill")
            parts.append(m15_aln)
            loaded.append("M15")

        # ── H4 setup context features + regime gate ────────────────────────────
        if df_h4 is not None and len(df_h4) >= 50:
            h4_ctx = self._build_h4_context(df_h4)
            h4_aln = h4_ctx.reindex(h1_feats.index, method="ffill")
            parts.append(h4_aln)
            loaded.append("H4")

        # ── D1 bias context features ──────────────────────────────────────────
        if df_d1 is not None and len(df_d1) >= 50:
            d1_ctx = self._build_d1_context(df_d1)
            d1_aln = d1_ctx.reindex(h1_feats.index, method="ffill")
            parts.append(d1_aln)
            loaded.append("D1")

        # ── Combine and clean ─────────────────────────────────────────────────
        combined = pd.concat(parts, axis=1)
        combined.dropna(inplace=True)

        n_feat = len([c for c in combined.columns
                      if not c.startswith("_") and not c.startswith("target")])
        logger.debug(
            f"[{symbol}] Multi-TF features: {loaded}  "
            f"→ {n_feat} features  {len(combined):,} bars"
        )

        if self.normalise:
            combined = self._rolling_zscore(combined)

        return combined

    # ─────────────────────────────────────────────────────────────────────────
    # H1 feature builders
    # ─────────────────────────────────────────────────────────────────────────
    def _h1_features(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """All H1 primary features (original 27 + new VWAP/volume/spread)."""
        feats = self._add_returns(feats, df)
        feats = self._add_rsi(feats, df)
        feats = self._add_atr(feats, df)
        feats = self._add_macd(feats, df)
        feats = self._add_bollinger(feats, df)
        feats = self._add_ema_ratios(feats, df)
        feats = self._add_volume(feats, df)
        feats = self._add_time_features(feats, df)

        # ── NEW: VWAP-based features (Tier 3) ─────────────────────────────────
        if USE_VWAP_FEATURES:
            feats = self._add_vwap_features(feats, df)

        # ── NEW: Tick volume imbalance proxy (Tier 3) ──────────────────────────
        if USE_TICK_VOLUME_IMBALANCE:
            feats = self._add_volume_imbalance(feats, df)

        # ── NEW: Spread dynamics (Tier 3) ─────────────────────────────────────
        if USE_SPREAD_DYNAMICS:
            feats = self._add_spread_dynamics(feats, df)

        return feats

    def _add_returns(self, feats, df):
        log_ret = np.log(df["close"] / df["close"].shift(1))
        for lag in FEATURE_LAGS:
            feats[f"ret_{lag}"] = np.log(df["close"] / df["close"].shift(lag))
        feats["ret_1_abs"] = log_ret.abs()
        return feats

    def _add_rsi(self, feats, df):
        delta = df["close"].diff()
        up    = delta.clip(lower=0)
        dn    = (-delta).clip(lower=0)
        rs    = (up.ewm(com=RSI_PERIOD-1, min_periods=RSI_PERIOD).mean() /
                 dn.ewm(com=RSI_PERIOD-1, min_periods=RSI_PERIOD).mean())
        feats["rsi"]      = 100 - (100 / (1 + rs))
        feats["rsi_norm"] = (feats["rsi"] - 50) / 50
        return feats

    def _add_atr(self, feats, df):
        tr  = self._true_range(df)
        atr = tr.rolling(ATR_PERIOD).mean()
        feats["atr_pct"]  = atr / (df["close"] + 1e-10)
        feats["hl_range"] = (df["high"] - df["low"]) / (atr + 1e-10)
        return feats

    def _add_macd(self, feats, df):
        ema_fast  = df["close"].ewm(span=MACD_FAST,   adjust=False).mean()
        ema_slow  = df["close"].ewm(span=MACD_SLOW,   adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal    = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        feats["macd_hist"] = (macd_line - signal) / (df["close"] + 1e-10)
        feats["macd_line"] = macd_line              / (df["close"] + 1e-10)
        return feats

    def _add_bollinger(self, feats, df):
        mid   = df["close"].rolling(BB_PERIOD).mean()
        std   = df["close"].rolling(BB_PERIOD).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        feats["bb_pct_b"] = (df["close"] - lower) / (upper - lower + 1e-10)
        feats["bb_width"]  = (upper - lower)       / (mid           + 1e-10)
        return feats

    def _add_ema_ratios(self, feats, df):
        emas  = {p: df["close"].ewm(span=p, adjust=False).mean() for p in EMA_PERIODS}
        close = df["close"]
        feats["ema_8_ratio"]   = close / (emas[8]   + 1e-10) - 1
        feats["ema_21_ratio"]  = close / (emas[21]  + 1e-10) - 1
        feats["ema_50_ratio"]  = close / (emas[50]  + 1e-10) - 1
        feats["ema_200_ratio"] = close / (emas[200] + 1e-10) - 1
        feats["ema_8_21"]   = np.sign(emas[8]  - emas[21])
        feats["ema_21_50"]  = np.sign(emas[21] - emas[50])
        feats["ema_50_200"] = np.sign(emas[50] - emas[200])
        return feats

    def _add_volume(self, feats, df):
        if "volume" in df.columns and df["volume"].sum() > 0:
            avg = df["volume"].rolling(20).mean()
            feats["volume_ratio"] = df["volume"] / (avg + 1e-10)
        else:
            feats["volume_ratio"] = 1.0
        return feats

    def _add_time_features(self, feats, df):
        hour = df.index.hour + df.index.minute / 60
        feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        dow = df.index.dayofweek.astype(float)
        feats["dow_sin"]  = np.sin(2 * np.pi * dow / 5)
        feats["dow_cos"]  = np.cos(2 * np.pi * dow / 5)
        return feats

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: VWAP features (Tier 3)
    # ─────────────────────────────────────────────────────────────────────────
    def _add_vwap_features(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        VWAP (Volume-Weighted Average Price) and Market Profile-like features.

        GOLD and SILVER respect VWAP very strongly on H1. The distance from
        VWAP tells the agent whether it's trading in value or against value.

        Features added:
          vwap_dev       — (close - VWAP) / ATR  [signed distance]
          vwap_dist_atr  — abs(close - VWAP) / ATR  [magnitude]
          value_area_pos — +1 if inside value area (VWAP ± 1 ATR), else sign of deviation
        """
        typical   = (df["high"] + df["low"] + df["close"]) / 3
        vol       = df["volume"] if ("volume" in df.columns and df["volume"].sum() > 0) else pd.Series(1.0, index=df.index)

        # Rolling daily VWAP — reset each calendar day
        # Group by date then do cumulative VWAP within each day
        date_groups = df.index.date
        cum_tpv = (typical * vol).groupby(date_groups, group_keys=False).cumsum()
        cum_vol  = vol.groupby(date_groups, group_keys=False).cumsum()
        vwap     = cum_tpv / (cum_vol + 1e-10)

        atr      = self._true_range(df).rolling(ATR_PERIOD).mean()
        atr_safe = atr.clip(lower=1e-8)

        dev = df["close"] - vwap
        feats["vwap_dev"]      = dev / atr_safe
        feats["vwap_dist_atr"] = dev.abs() / atr_safe

        # Value-area position: inside ±1 ATR of VWAP = in value
        in_value = dev.abs() <= atr_safe
        feats["value_area_pos"] = np.where(in_value, 0.0, np.sign(dev))

        return feats

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: Tick volume imbalance proxy (Tier 3)
    # ─────────────────────────────────────────────────────────────────────────
    def _add_volume_imbalance(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Directional volume proxy without actual tick data.

        When a bar closes near the high with high volume → buying pressure.
        When a bar closes near the low  with high volume → selling pressure.

        Formula:
          close_loc = (close - low) / (high - low + 1e-10)  → [0, 1]
          imbalance = (close_loc - 0.5) * volume_ratio      → signed [-0.5, 0.5]

        This approximates the CMF (Chaikin Money Flow) signal.
        """
        hl_range  = (df["high"] - df["low"]).clip(lower=1e-10)
        close_loc = (df["close"] - df["low"]) / hl_range   # 0=closed at low, 1=at high

        vol       = df["volume"] if ("volume" in df.columns and df["volume"].sum() > 0) else pd.Series(1.0, index=df.index)
        vol_norm  = vol / (vol.rolling(20).mean() + 1e-10)

        feats["vol_imbalance"] = (close_loc - 0.5) * vol_norm

        # Cumulative money flow (rolling 14 bars)
        money_flow = (close_loc - 0.5) * vol
        feats["cmf_14"] = money_flow.rolling(14).sum() / (vol.rolling(14).sum() + 1e-10)

        return feats

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: Spread dynamics (Tier 3)
    # ─────────────────────────────────────────────────────────────────────────
    def _add_spread_dynamics(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        HL spread dynamics — widening spread = institutional caution.

        Features:
          hl_spread_ratio — current H-L / rolling average H-L  (>1 = widening)
          spread_z        — z-score of H-L spread over 20 bars
        """
        hl = df["high"] - df["low"]
        avg_hl  = hl.rolling(20).mean()
        std_hl  = hl.rolling(20).std()

        feats["hl_spread_ratio"] = hl / (avg_hl + 1e-10)
        feats["spread_z"]        = (hl - avg_hl) / (std_hl + 1e-10)

        return feats

    # ─────────────────────────────────────────────────────────────────────────
    # M15 entry-refinement context  (6 features, prefix m15_)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_m15_context(self, df: pd.DataFrame) -> pd.DataFrame:
        ctx   = pd.DataFrame(index=df.index)
        close = df["close"]

        ctx["m15_ret_1"] = np.log(close / close.shift(1))

        delta = close.diff()
        up    = delta.clip(lower=0)
        dn    = (-delta).clip(lower=0)
        rs    = (up.ewm(com=13, min_periods=14).mean() /
                 dn.ewm(com=13, min_periods=14).mean())
        ctx["m15_rsi"] = (100 - 100 / (1 + rs) - 50) / 50

        ema8  = close.ewm(span=8,  adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ctx["m15_ema8_rat"]  = close / (ema8  + 1e-10) - 1
        ctx["m15_ema21_rat"] = close / (ema21 + 1e-10) - 1

        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        ctx["m15_macd_hist"] = (macd - signal) / (close + 1e-10)

        tr  = self._true_range(df)
        atr = tr.rolling(14).mean()
        ctx["m15_atr_pct"] = atr / (close + 1e-10)

        ctx.dropna(inplace=True)
        return ctx

    # ─────────────────────────────────────────────────────────────────────────
    # H4 setup context  (12 features, prefix h4_)
    # IMPROVED: Added H4 regime signal as primary decision gate (Tier 1 #4)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_h4_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        H4 features give the agent the 4-hour market structure.

        NEW features vs original:
          h4_regime_signal — composite: -1 ranging, 0 neutral, +1 trending
                             This acts as the primary H4 decision gate.
          h4_adx           — ADX strength (is the H4 trend strong enough?)
        """
        ctx   = pd.DataFrame(index=df.index)
        close = df["close"]

        ctx["h4_ret_1"] = np.log(close / close.shift(1))

        delta = close.diff()
        up    = delta.clip(lower=0)
        dn    = (-delta).clip(lower=0)
        rs    = (up.ewm(com=13, min_periods=14).mean() /
                 dn.ewm(com=13, min_periods=14).mean())
        ctx["h4_rsi"] = (100 - 100 / (1 + rs) - 50) / 50

        tr  = self._true_range(df)
        atr = tr.rolling(14).mean()
        ctx["h4_atr_pct"]  = atr  / (close + 1e-10)
        ctx["h4_hl_range"] = (df["high"] - df["low"]) / (atr + 1e-10)

        ema20  = close.ewm(span=20,  adjust=False).mean()
        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        ctx["h4_ema20_rat"]  = close / (ema20  + 1e-10) - 1
        ctx["h4_ema50_rat"]  = close / (ema50  + 1e-10) - 1
        ctx["h4_ema200_rat"] = close / (ema200 + 1e-10) - 1
        ctx["h4_trend_up"]   = (
            (ema20 > ema50) & (ema50 > ema200)
        ).astype(float) * 2 - 1

        mid  = close.rolling(20).mean()
        std  = close.rolling(20).std()
        ctx["h4_bb_pct_b"] = (close - (mid - 2*std)) / (4*std + 1e-10)

        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        ctx["h4_macd_hist"] = (macd - signal) / (close + 1e-10)

        # ── NEW: H4 ADX — regime gate (Tier 1 #4) ────────────────────────────
        n = 14
        up_move = df["high"] - df["high"].shift(1)
        dn_move = df["low"].shift(1) - df["low"]
        plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        plus_di  = 100 * pd.Series(plus_dm,  index=df.index).rolling(n).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).mean() / (atr + 1e-10)
        dx       = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
        adx      = dx.rolling(n).mean()
        ctx["h4_adx"] = adx / 100.0   # normalise 0–1

        # H4 regime signal: +1 = trending (flip-friendly), -1 = ranging (hold)
        # ADX > 25 = trend; direction from DI crossover
        trending     = adx > 25
        bullish_trend = trending & (plus_di > minus_di)
        bearish_trend = trending & (minus_di > plus_di)
        regime_signal = np.where(bullish_trend, 1.0,
                        np.where(bearish_trend, -1.0, 0.0))
        ctx["h4_regime_signal"] = pd.Series(regime_signal, index=df.index)

        ctx.dropna(inplace=True)
        return ctx

    # ─────────────────────────────────────────────────────────────────────────
    # D1 bias context  (10 features, prefix d1_)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_d1_context(self, df: pd.DataFrame) -> pd.DataFrame:
        ctx   = pd.DataFrame(index=df.index)
        close = df["close"]

        ctx["d1_ret_1"] = np.log(close / close.shift(1))

        delta = close.diff()
        up    = delta.clip(lower=0)
        dn    = (-delta).clip(lower=0)
        rs    = (up.ewm(com=13, min_periods=14).mean() /
                 dn.ewm(com=13, min_periods=14).mean())
        ctx["d1_rsi"] = (100 - 100 / (1 + rs) - 50) / 50

        tr  = self._true_range(df)
        atr = tr.rolling(14).mean()
        ctx["d1_atr_pct"]  = atr  / (close + 1e-10)
        ctx["d1_hl_range"] = (df["high"] - df["low"]) / (atr + 1e-10)

        ema20  = close.ewm(span=20,  adjust=False).mean()
        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        ctx["d1_ema20_rat"]  = close / (ema20  + 1e-10) - 1
        ctx["d1_ema50_rat"]  = close / (ema50  + 1e-10) - 1
        ctx["d1_ema200_rat"] = close / (ema200 + 1e-10) - 1
        ctx["d1_trend_up"]   = (
            (ema20 > ema50) & (ema50 > ema200)
        ).astype(float) * 2 - 1

        mid  = close.rolling(20).mean()
        std  = close.rolling(20).std()
        ctx["d1_bb_pct_b"] = (close - (mid - 2*std)) / (4*std + 1e-10)

        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        ctx["d1_macd_hist"] = (macd - signal) / (close + 1e-10)

        ctx.dropna(inplace=True)
        return ctx

    # ─────────────────────────────────────────────────────────────────────────
    # Regime & direction labels
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _direction_label(df: pd.DataFrame, horizon: int) -> pd.Series:
        future = df["close"].shift(-horizon)
        return (future > df["close"]).astype(float)

    @staticmethod
    def regime_label(df: pd.DataFrame, atr_window: int = 14) -> pd.Series:
        """
        Hindsight regime labels  0–3:
          0 = Range low-vol
          1 = Trend up
          2 = Trend down
          3 = High-vol breakout
        """
        close = df["close"]
        tr    = pd.DataFrame({
            "hl": df["high"] - df["low"],
            "hc": (df["high"] - close.shift()).abs(),
            "lc": (df["low"]  - close.shift()).abs(),
        }).max(axis=1)
        atr = tr.rolling(atr_window).mean()

        n        = atr_window
        up_move  = df["high"] - df["high"].shift()
        dn_move  = df["low"].shift() - df["low"]
        plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

        plus_di  = 100 * pd.Series(plus_dm,  index=df.index).rolling(n).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).mean() / (atr + 1e-10)
        dx       = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
        adx      = dx.rolling(n).mean()

        vol_high = atr / close > atr.rolling(50).mean() / close * 1.5

        labels           = pd.Series(0, index=df.index, dtype=int)
        labels[adx > 25] = np.where(plus_di[adx > 25] > minus_di[adx > 25], 1, 2)
        labels[vol_high & (adx <= 25)] = 3
        return labels

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        prev = df["close"].shift(1)
        return pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"]  - prev).abs(),
        ], axis=1).max(axis=1)

    def _rolling_zscore(self, feats: pd.DataFrame) -> pd.DataFrame:
        result  = feats.copy()
        exclude = [c for c in feats.columns
                   if c.startswith("_") or c.startswith("target")]
        for col in feats.columns:
            if col in exclude:
                continue
            rm = feats[col].rolling(self.window, min_periods=self.window // 2).mean()
            rs = feats[col].rolling(self.window, min_periods=self.window // 2).std()
            result[col] = (feats[col] - rm) / (rs + 1e-8)
        num_cols = [c for c in result.columns if c not in exclude]
        result[num_cols] = result[num_cols].clip(-5, 5)
        result.dropna(inplace=True)
        return result
