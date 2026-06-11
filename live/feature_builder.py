"""
live/feature_builder.py  —  Build live observation vector from MT5 data
──────────────────────────────────────────────────────────────────────────────
Fetches the last N bars across all timeframes, runs feature engineering,
loads ML model outputs, and returns the observation vector ready for
the PPO model's predict() call.
"""

import sys
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR, N_REGIMES, LOOKBACK_STEPS
from data.feature_engineer import FeatureEngineer
from live.settings import LiveSettings
from live.mt5_bridge import MT5Bridge


class FeatureBuilder:
    """
    Builds the RL agent observation vector from live MT5 data.

    Parameters
    ----------
    bridge   : MT5Bridge — for fetching live bars
    settings : LiveSettings
    """

    def __init__(self, bridge: MT5Bridge, settings: LiveSettings):
        self.bridge   = bridge
        self.s        = settings
        self.engineer = FeatureEngineer(normalise=True)

        # Cache ML models (loaded once at startup)
        self._regime_clf   = {}   # symbol → RegimeClassifier
        self._lstm_model   = {}   # symbol → LSTMDirectionModel
        self._feat_cols    = {}   # symbol → list of feature column names

        self._load_ml_models()

    # ── Startup: load saved models ────────────────────────────────────────────
    def _load_ml_models(self):
        from models.regime_classifier import RegimeClassifier
        from models.price_predictor   import LSTMDirectionModel

        for sym in self.s.active_symbols:
            # Regime classifier
            for tag in ("shared", sym):
                try:
                    self._regime_clf[sym] = RegimeClassifier.load(tag)
                    logger.info(f"[{sym}] Regime classifier loaded ({tag})")
                    break
                except FileNotFoundError:
                    continue

            if sym not in self._regime_clf:
                logger.warning(f"[{sym}] No regime classifier found — regime features will be zeros")

        logger.info("ML models loaded.")

    def _load_lstm(self, sym: str, n_features: int):
        """Load LSTM on demand once we know the feature count."""
        if sym in self._lstm_model:
            return
        from models.price_predictor import LSTMDirectionModel
        for tag in ("shared", sym):
            try:
                self._lstm_model[sym] = LSTMDirectionModel.load(n_features, tag)
                logger.info(f"[{sym}] LSTM loaded ({tag})")
                return
            except FileNotFoundError:
                continue
        logger.warning(f"[{sym}] No LSTM found — direction feature will be 0.5")

    # ── Main: build observation for one symbol ────────────────────────────────
    def build_observation(
        self,
        symbol:   str,
        position: int,    # current position: +1 or -1
        balance:  float,
        peak:     float,
        n_flips:  int,
        steps_held: int,
    ) -> Optional[tuple]:
        """
        Fetch live bars, engineer features, attach ML outputs,
        and assemble the full observation vector.

        Returns np.ndarray of shape matching the training obs space,
        or None if data is unavailable.
        """
        try:
            # ── Fetch bars from MT5 ───────────────────────────────────────────
            df_h1  = self.bridge.get_bars(symbol, "H1",  self.s.warmup_h1)
            df_h4  = self.bridge.get_bars(symbol, "H4",  self.s.warmup_h4)
            df_d1  = self.bridge.get_bars(symbol, "D1",  self.s.warmup_d1)
            df_m15 = self.bridge.get_bars(symbol, "M15", self.s.warmup_m15)

            if df_h1 is None or len(df_h1) < 50:
                logger.error(f"[{symbol}] Not enough H1 bars for feature engineering")
                return None

            # ── Build multi-TF features ───────────────────────────────────────
            feats = self.engineer.transform_multi_tf(
                df_h1  = df_h1,
                df_d1  = df_d1  if df_d1  is not None and len(df_d1)  >= 10 else None,
                df_h4  = df_h4  if df_h4  is not None and len(df_h4)  >= 10 else None,
                df_m15 = df_m15 if df_m15 is not None and len(df_m15) >= 20 else None,
                symbol = symbol,
            )

            if feats.empty:
                logger.error(f"[{symbol}] Feature engineering produced empty DataFrame")
                return None

            feat_cols = [c for c in feats.columns
                         if not c.startswith("_") and not c.startswith("target")]
            self._feat_cols[symbol] = feat_cols

            # Use the most recent LOOKBACK_STEPS rows
            n_feat = len(feat_cols)
            lookback_rows = feats[feat_cols].iloc[-LOOKBACK_STEPS:]
            if len(lookback_rows) < LOOKBACK_STEPS:
                pad = pd.DataFrame(
                    np.zeros((LOOKBACK_STEPS - len(lookback_rows), n_feat)),
                    columns=feat_cols,
                )
                lookback_rows = pd.concat([pad, lookback_rows], ignore_index=True)
            feat_flat = lookback_rows.values.flatten().astype(np.float32)

            # ── ML model outputs ──────────────────────────────────────────────
            # Regime probabilities (most recent bar)
            regime_proba = np.full(N_REGIMES, 1.0 / N_REGIMES, dtype=np.float32)
            if symbol in self._regime_clf:
                try:
                    X_row = pd.DataFrame(
                        feats[feat_cols].iloc[[-1]].values,
                        columns=feat_cols,
                    )
                    regime_proba = self._regime_clf[symbol].predict_proba(X_row)[0].astype(np.float32)
                except Exception as e:
                    logger.warning(f"[{symbol}] Regime inference failed: {e}")

            # Direction probability (most recent bar)
            dir_proba = np.array([0.5], dtype=np.float32)
            self._load_lstm(symbol, n_feat)
            if symbol in self._lstm_model:
                try:
                    from config import LSTM_SEQ_LEN
                    seq_len = LSTM_SEQ_LEN
                    if len(feats) >= seq_len:
                        X_window = feats[feat_cols].iloc[-seq_len:].values.astype(np.float32)
                        p = self._lstm_model[symbol].predict_proba_single(X_window)
                        dir_proba = np.array([p], dtype=np.float32)
                except Exception as e:
                    logger.warning(f"[{symbol}] LSTM inference failed: {e}")

            # ─── Position / portfolio context ───────────────────────────────────
            last_atr = float(feats["_atr"].iloc[-1]) if "_atr" in feats.columns else 1.0
            
            from env.trading_env import _session_multiplier
            hour = int(feats.index[-1].hour) if not feats.empty else 0
            
            pos_vec = np.array([
                float(position),
                float(steps_held) / 100.0,
                (balance - self.s.initial_balance) / self.s.initial_balance,
                (peak - balance) / max(last_atr, 1e-8),
                float(n_flips) / 100.0,
                _session_multiplier(hour), # Session multiplier (Tier 2)
                0.0,                       # consecutive_losses / 10.0 (Tier 4)
                0.0,                       # consecutive_wins / 10.0 (Tier 4)
            ], dtype=np.float32)

            # ── Assemble full observation ─────────────────────────────────────
            obs = np.concatenate([feat_flat, pos_vec, regime_proba, dir_proba])
            obs = np.clip(obs, -10.0, 10.0)
            feat_row = feats[feat_cols].iloc[-1].values
            return obs, feat_row

        except Exception as e:
            logger.error(f"[{symbol}] build_observation failed: {e}", exc_info=True)
            return None

    def get_last_atr(self, symbol: str) -> float:
        """Return the most recent ATR for position sizing (H1 default)."""
        return self.get_last_atr_tf(symbol, "H1")

    def get_last_atr_tf(self, symbol: str, timeframe: str = "H1") -> float:
        """Return the most recent ATR for a given timeframe."""
        try:
            df = self.bridge.get_bars(symbol, timeframe, 30)
            if df is None or df.empty:
                return 1.0
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"]  - df["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            return atr if atr > 0 else 1.0
        except Exception as e:
            logger.warning(f"[{symbol}] ATR({timeframe}) failed: {e}")
            return 1.0
