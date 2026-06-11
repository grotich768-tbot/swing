"""
training/train_supervised.py  —  Train regime classifier + LSTM direction model
──────────────────────────────────────────────────────────────────────────────
Loads M15 + H1 + H4 + D1 for each symbol, builds full multi-TF feature
matrices, then trains:
  1. XGBoost regime classifier  (shared across all symbols)
  2. LSTM direction predictor   (shared across all symbols)

Usage:
    python -m training.train_supervised
    python -m training.train_supervised --symbol GOLD
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SYMBOLS, TRAIN_START, TRAIN_END, LSTM_SEQ_LEN, LSTM_MAX_SAMPLES
from data.data_loader import DataLoader
from data.feature_engineer import FeatureEngineer
from models.regime_classifier import RegimeClassifier
from models.price_predictor import LSTMDirectionModel


def _load_symbol(loader, engineer, sym, date_from, date_to):
    """Load all TFs for one symbol and build multi-TF features."""
    raw_h1  = loader.load(sym, "H1",  date_from, date_to)
    raw_d1  = loader.load(sym, "D1",  date_from, date_to)
    raw_h4  = loader.load(sym, "H4",  date_from, date_to)
    raw_m15 = loader.load(sym, "M15", date_from, date_to)

    if raw_h1 is None or len(raw_h1) < 500:
        logger.warning(f"[{sym}] Not enough H1 data — skipping")
        return None, None

    feats = engineer.transform_multi_tf(
        df_h1  = raw_h1,
        df_d1  = raw_d1,
        df_h4  = raw_h4,
        df_m15 = raw_m15,
        symbol = sym,
    )

    regime_labels = FeatureEngineer.regime_label(raw_h1)
    regime_labels = regime_labels.reindex(feats.index)

    return feats, regime_labels


def _resolve_dates(symbol: str, date_from: str, date_to: str):
    """Return per-symbol date overrides if configured, else use provided defaults."""
    try:
        from config import SYMBOL_TRAIN_START, SYMBOL_TRAIN_END
        date_from = SYMBOL_TRAIN_START.get(symbol, date_from)
        date_to   = SYMBOL_TRAIN_END.get(symbol,   date_to)
    except ImportError:
        pass
    return date_from, date_to


def train_supervised(
    symbols:   list = None,
    date_from: str  = TRAIN_START,
    date_to:   str  = TRAIN_END,
    shared:    bool = False,
):
    symbols  = symbols or SYMBOLS
    loader   = DataLoader()
    engineer = FeatureEngineer(normalise=True)

    all_features   = []
    all_regimes    = []
    all_directions = []
    symbol_map     = {}

    logger.info("=== Building multi-timeframe feature matrices ===")
    for sym in symbols:
        sym_from, sym_to = _resolve_dates(sym, date_from, date_to)
        feats, regime_labels = _load_symbol(loader, engineer, sym, sym_from, sym_to)
        if feats is None:
            continue

        feat_cols     = [c for c in feats.columns
                         if not c.startswith("_") and not c.startswith("target")]
        direction_col = "_target_direction"

        start = sum(len(f) for f in all_features)
        all_features.append(feats[feat_cols])
        all_regimes.append(regime_labels)
        all_directions.append(
            feats[direction_col] if direction_col in feats.columns
            else pd.Series(np.nan, index=feats.index)
        )
        symbol_map[sym] = (start, start + len(feats))

        tfs_loaded = []
        if loader.load(sym, "M15", date_from, date_to) is not None: tfs_loaded.append("M15")
        tfs_loaded.append("H1")
        if loader.load(sym, "H4",  date_from, date_to) is not None: tfs_loaded.append("H4")
        if loader.load(sym, "D1",  date_from, date_to) is not None: tfs_loaded.append("D1")

        logger.info(
            f"  {sym}: {len(feats):,} bars  "
            f"features={len(feat_cols)}  "
            f"TFs={tfs_loaded}  "
            f"regime_dist={regime_labels.value_counts().to_dict()}"
        )

    if not all_features:
        raise RuntimeError("No data loaded. Run the MT5 fetcher first.")

    X_all = pd.concat(all_features,   axis=0).reset_index(drop=True)
    y_reg = pd.concat(all_regimes,    axis=0).reset_index(drop=True)
    y_dir = pd.concat(all_directions, axis=0).reset_index(drop=True)

    logger.info(f"Total samples: {len(X_all):,}  Features: {X_all.shape[1]}")

    # ── Regime Classifier ─────────────────────────────────────────────────────
    logger.info("\n=== Training Regime Classifier (XGBoost) ===")
    if shared:
        clf = RegimeClassifier(symbol="shared")
        clf.fit(X_all, y_reg)
        clf.save()
        imp = clf.feature_importance(list(X_all.columns))
        logger.info(f"Top-10 features:\n{imp.head(10).to_string()}")
    else:
        for sym in symbols:
            s, e = symbol_map.get(sym, (None, None))
            if s is None: continue
            RegimeClassifier(symbol=sym).fit(X_all.iloc[s:e], y_reg.iloc[s:e]).save()

    # ── LSTM Direction Predictor ──────────────────────────────────────────────
    logger.info("\n=== Training LSTM Direction Predictor ===")
    valid = ~y_dir.isna()
    X_lstm = X_all[valid].values.astype(np.float32)
    y_lstm = y_dir[valid].values.astype(np.float32)

    if len(X_lstm) < LSTM_SEQ_LEN + 100:
        logger.warning(f"Not enough data for LSTM ({len(X_lstm)} rows). Skipping.")
    else:
        if shared:
            LSTMDirectionModel(n_features=X_lstm.shape[1], symbol="shared")\
                .fit(X_lstm, y_lstm).save()
        else:
            for sym in symbols:
                s, e   = symbol_map.get(sym, (None, None))
                if s is None: continue
                mask   = valid.iloc[s:e]
                Xs     = X_all.iloc[s:e][mask].values.astype(np.float32)
                ys     = y_dir.iloc[s:e][mask].values.astype(np.float32)
                LSTMDirectionModel(n_features=Xs.shape[1], symbol=sym)\
                    .fit(Xs, ys).save()

    logger.success("=== Supervised training complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",      type=str, default=None)
    parser.add_argument("--from",        dest="date_from", default=TRAIN_START,
                        help="Training start (overridden per-symbol by SYMBOL_TRAIN_START in config)")
    parser.add_argument("--to",          dest="date_to",   default=TRAIN_END,
                        help="Training end (overridden per-symbol by SYMBOL_TRAIN_END in config)")
    parser.add_argument("--shared",      action="store_true")
    args = parser.parse_args()
    train_supervised(
        symbols   = [args.symbol] if args.symbol else None,
        date_from = args.date_from,
        date_to   = args.date_to,
        shared    = args.shared,
    )
