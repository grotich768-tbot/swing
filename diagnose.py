"""
Run this in your project folder to diagnose the Sharpe inflation.
python diagnose.py
"""
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from data.data_loader import DataLoader
from data.feature_engineer import FeatureEngineer

loader   = DataLoader()
engineer = FeatureEngineer(normalise=True)

sym = "GOLD"
raw_h1  = loader.load(sym, "H1",  "2024-01-01", "2024-12-31")
raw_d1  = loader.load(sym, "D1",  "2024-01-01", "2024-12-31")
raw_h4  = loader.load(sym, "H4",  "2024-01-01", "2024-12-31")
raw_m15 = loader.load(sym, "M15", "2024-01-01", "2024-12-31")

feats = engineer.transform_multi_tf(raw_h1, raw_d1, raw_h4, raw_m15, sym)

print(f"\n=== GOLD 2024 raw data ===")
print(f"H1 bars       : {len(raw_h1):,}")
print(f"Price range   : {raw_h1['close'].min():.2f} → {raw_h1['close'].max():.2f}")
print(f"H1 ATR (mean) : {(raw_h1['high']-raw_h1['low']).mean():.4f}")
print(f"Feature bars  : {len(feats):,}")
print(f"_atr (mean)   : {feats['_atr'].mean():.6f}")
print(f"_atr (min)    : {feats['_atr'].min():.6f}")
print(f"_atr (max)    : {feats['_atr'].max():.6f}")

# Simulate lot sizing
from config import PIP_VALUE, INITIAL_BALANCE, MAX_POSITION_PCT
PIP_USD_PER_LOT = {"GOLD": 1.0, "SILVER": 5.0, "EURUSD": 10.0, "GBPUSD": 10.0}
ATR_STOP_MULT = 1.5
MAX_LOTS = 0.5

atr_vals = feats["_atr"].values
pip = PIP_VALUE[sym]
pip_usd = PIP_USD_PER_LOT[sym]
risk_usd = INITIAL_BALANCE * MAX_POSITION_PCT
atr_pips = atr_vals * ATR_STOP_MULT / pip
lots = np.clip(risk_usd / (atr_pips * pip_usd + 1e-10), 0.01, MAX_LOTS)
print(f"\n=== Position sizing ===")
print(f"risk_usd      : ${risk_usd:.2f}")
print(f"ATR pips mean : {atr_pips.mean():.1f}")
print(f"Lots (mean)   : {lots.mean():.4f}")
print(f"Lots (min)    : {lots.min():.4f}")
print(f"Lots (max)    : {lots.max():.4f}")
print(f"% hitting MAX : {(lots >= MAX_LOTS).mean():.1%}")

# Check if _target_direction leaked into training features
feat_cols = [c for c in feats.columns if not c.startswith("_") and not c.startswith("target")]
print(f"\n=== Feature check ===")
print(f"Total feat cols : {len(feat_cols)}")
print(f"Has _target_dir : {'_target_direction' in feats.columns}")
leaked = [c for c in feat_cols if "target" in c.lower() or "future" in c.lower()]
print(f"Leaked cols     : {leaked if leaked else 'None found'}")

# Check if direction_proba correlates strongly with next bar return
print(f"\n=== Next-bar predictability ===")
close = raw_h1["close"].values
ret_next = np.roll(close, -1) / close - 1
ret_next[-1] = 0
feats_aligned = feats.iloc[:len(ret_next)]
print(f"Corr(rsi, next_ret)    : {np.corrcoef(feats_aligned['rsi_norm'].fillna(0), ret_next[:len(feats_aligned)])[0,1]:.4f}")

