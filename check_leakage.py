"""
Checks three specific leakage vectors in the always-in bot.
Run from your project root: python check_leakage.py
"""
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np

# ── 1. Date overlap check ─────────────────────────────────────────────────────
from config import TRAIN_START, TRAIN_END, TEST_START, TEST_END
print("=== 1. Date overlap ===")
print(f"  TRAIN : {TRAIN_START} → {TRAIN_END}")
print(f"  TEST  : {TEST_START}  → {TEST_END}")
overlap = TRAIN_END >= TEST_START
print(f"  Overlap: {'⚠  YES — LEAKAGE' if overlap else '✓  None'}")

# ── 2. Check if regime/LSTM models trained on test data ───────────────────────
from config import MODEL_DIR
import pickle, pathlib
print("\n=== 2. Supervised model training period ===")
try:
    with open(MODEL_DIR / "regime_shared.pkl", "rb") as f:
        clf = pickle.load(f)
    print(f"  Regime model exists — check training date_from/date_to manually")
    print(f"  n_estimators: {clf.model.n_estimators if clf.model else 'N/A'}")
except Exception as e:
    print(f"  Could not load: {e}")

# ── 3. Check env reward vs actual price direction ─────────────────────────────
print("\n=== 3. Win rate sanity check ===")
print("  Win rate 60%+ on H1 bars: is this realistic?")
print("  Checking GOLD 2024 simple trend-follow benchmark...")

from data.data_loader import DataLoader
loader = DataLoader()
raw = loader.load("GOLD", "H1", "2024-01-01", "2024-12-31")
if raw is not None:
    ret = raw["close"].pct_change().dropna()
    # Simple always-long benchmark
    long_wins  = (ret > 0).mean()
    # 1-bar momentum (buy if last bar up)
    mom_signal = (ret.shift(1) > 0)
    mom_wins   = (ret[mom_signal] > 0).mean()
    # Naive always-long return
    always_long_ret = ret.sum()
    print(f"  GOLD 2024 total return (buy&hold): {always_long_ret:.2%}")
    print(f"  Random long win rate (H1)         : {long_wins:.2%}")
    print(f"  1-bar momentum win rate           : {mom_wins:.2%}")
    print(f"  → If bot win rate >> momentum, it has real edge")
    print(f"  → If bot win rate ≈ buy&hold win rate, it's just riding the trend")

# ── 4. Check _target_direction exposure ───────────────────────────────────────
print("\n=== 4. _target_direction leakage check ===")
from data.feature_engineer import FeatureEngineer
eng = FeatureEngineer(normalise=True)
raw_h1 = loader.load("GOLD", "H1", "2023-01-01", "2023-06-30")
if raw_h1 is not None:
    feats = eng.transform(raw_h1)
    feat_cols = [c for c in feats.columns
                 if not c.startswith("_") and not c.startswith("target")]
    has_target = "_target_direction" in feats.columns
    in_feats   = "target_direction" in feat_cols or "_target_direction" in feat_cols
    print(f"  _target_direction in DataFrame : {has_target}")
    print(f"  _target_direction in feat_cols : {in_feats}")
    print(f"  ✓ Clean" if not in_feats else "  ⚠  LEAKAGE — future label in features!")

print("\nDone. Share this output to diagnose Sharpe inflation.")
