"""
training/train_rl.py  —  Train PPO agents for each symbol
──────────────────────────────────────────────────────────────────────────────
Loads all four timeframes (M15 / H1 / H4 / D1), builds full multi-TF feature
matrices, attaches regime + direction ML outputs, then trains one PPO agent
per symbol using Stable-Baselines3.

Usage:
    python -m training.train_rl
    python -m training.train_rl --symbol GOLD
    python -m training.train_rl --timesteps 500000
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SYMBOLS, TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    PPO_PARAMS, TOTAL_TIMESTEPS, MODEL_DIR, LOG_DIR,
)
from data.data_loader import DataLoader
from data.feature_engineer import FeatureEngineer
from models.regime_classifier import RegimeClassifier
from models.price_predictor import LSTMDirectionModel
from env.trading_env import AlwaysInEnv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_features(loader, engineer, symbol, date_from, date_to):
    """Load all TFs and return (feats_df, feat_cols, raw_h1)."""
    raw_h1  = loader.load(symbol, "H1",  date_from, date_to)
    raw_d1  = loader.load(symbol, "D1",  date_from, date_to)
    raw_h4  = loader.load(symbol, "H4",  date_from, date_to)
    raw_m15 = loader.load(symbol, "M15", date_from, date_to)

    if raw_h1 is None or len(raw_h1) < 500:
        return None, None, None

    feats = engineer.transform_multi_tf(
        df_h1  = raw_h1,
        df_d1  = raw_d1,
        df_h4  = raw_h4,
        df_m15 = raw_m15,
        symbol = symbol,
    )
    feat_cols = [c for c in feats.columns
                 if not c.startswith("_") and not c.startswith("target")]
    return feats, feat_cols, raw_h1


def _attach_ml_outputs(feats, feat_cols, symbol):
    """Attach regime probabilities and direction probabilities."""
    regime_proba = direction_proba = None

    # Regime classifier
    for tag in ("shared", symbol):
        try:
            clf = RegimeClassifier.load(tag)
            regime_proba = clf.predict_proba(
                pd.DataFrame(feats[feat_cols].values, columns=feat_cols)
            )
            logger.info(f"[{symbol}] Regime proba loaded ({tag})")
            break
        except FileNotFoundError:
            continue

    # LSTM direction
    n_feat = len(feat_cols)
    for tag in ("shared", symbol):
        try:
            lstm = LSTMDirectionModel.load(n_features=n_feat, symbol=tag)
            direction_proba = lstm.predict_proba(feats[feat_cols].values)
            logger.info(f"[{symbol}] Direction proba loaded ({tag})")
            break
        except FileNotFoundError:
            continue

    return regime_proba, direction_proba


# ─────────────────────────────────────────────────────────────────────────────
# Single-symbol training
# ─────────────────────────────────────────────────────────────────────────────
def train_symbol(
    symbol:     str,
    timesteps:  int = TOTAL_TIMESTEPS,
    date_from:  str = TRAIN_START,
    date_to:    str = TRAIN_END,
    test_from:  str = TEST_START,
    test_to:    str = TEST_END,
) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, CallbackList
    )

    logger.info(f"\n{'='*60}\n  Training: {symbol}\n{'='*60}")

    loader   = DataLoader()
    engineer = FeatureEngineer(normalise=True)

    # ── Build training features ───────────────────────────────────────────────
    feats_train, feat_cols, _ = _load_features(
        loader, engineer, symbol, date_from, date_to
    )
    if feats_train is None:
        logger.error(f"[{symbol}] Not enough training data.")
        return {}

    reg_tr, dir_tr = _attach_ml_outputs(feats_train, feat_cols, symbol)

    # ── Build test features ───────────────────────────────────────────────────
    feats_test, feat_cols_t, _ = _load_features(
        loader, engineer, symbol, test_from, test_to
    )
    reg_te = dir_te = None
    if feats_test is not None and len(feats_test) > 100:
        reg_te, dir_te = _attach_ml_outputs(feats_test, feat_cols_t, symbol)

    # ── Environments ─────────────────────────────────────────────────────────
    def make_train_env():
        return Monitor(AlwaysInEnv(
            features_df     = feats_train,
            symbol          = symbol,
            regime_proba    = reg_tr,
            direction_proba = dir_tr,
            mode            = "train",
        ))

    train_env = DummyVecEnv([make_train_env])

    eval_env = None
    if feats_test is not None and len(feats_test) > 100:
        eval_env = Monitor(AlwaysInEnv(
            features_df     = feats_test,
            symbol          = symbol,
            regime_proba    = reg_te,
            direction_proba = dir_te,
            mode            = "test",
        ))

    obs_size = train_env.observation_space.shape[0]
    logger.info(f"[{symbol}] Observation size: {obs_size}  Timesteps: {timesteps:,}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    model_path = MODEL_DIR / f"ppo_{symbol}"
    log_path   = LOG_DIR   / f"ppo_{symbol}"
    log_path.mkdir(parents=True, exist_ok=True)

    callbacks = []
    if eval_env:
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path = str(model_path),
            log_path             = str(log_path),
            eval_freq            = max(1000, timesteps // 20),
            n_eval_episodes      = 5,
            deterministic        = True,
            verbose              = 1,
        ))
    callbacks.append(CheckpointCallback(
        save_freq   = max(5000, timesteps // 10),
        save_path   = str(model_path / "checkpoints"),
        name_prefix = f"ppo_{symbol}",
    ))

    # ── PPO model ─────────────────────────────────────────────────────────────
    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        verbose         = 1,
        tensorboard_log = str(LOG_DIR / "tensorboard"),
        **PPO_PARAMS,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = datetime.now()
    model.learn(
        total_timesteps = timesteps,
        callback        = CallbackList(callbacks) if callbacks else None,
        progress_bar    = True,
    )
    elapsed = (datetime.now() - t0).total_seconds()
    logger.success(f"[{symbol}] Training done in {elapsed:.0f}s")

    # ── Save final ────────────────────────────────────────────────────────────
    final_path = MODEL_DIR / f"ppo_{symbol}_final"
    model.save(str(final_path))
    logger.info(f"[{symbol}] Saved → {final_path}.zip")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = {}
    if eval_env:
        metrics = _evaluate(model, eval_env, symbol, n_episodes=10)
        logger.info(
            f"[{symbol}] "
            + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
        with open(LOG_DIR / f"metrics_{symbol}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    train_env.close()
    if eval_env: eval_env.close()
    return metrics


def _evaluate(model, env, symbol, n_episodes=10):
    all_pnls, all_flips, all_sharpes = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(int(action))
            done = done or truncated
        s = env.unwrapped.episode_summary()
        all_pnls.append(s["total_pnl"])
        all_flips.append(s["n_flips"])
        all_sharpes.append(s["sharpe"])
    return {
        "mean_pnl":    float(np.mean(all_pnls)),
        "std_pnl":     float(np.std(all_pnls)),
        "mean_flips":  float(np.mean(all_flips)),
        "mean_sharpe": float(np.mean(all_sharpes)),
        "win_rate":    float(np.mean([p > 0 for p in all_pnls])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train all symbols
# ─────────────────────────────────────────────────────────────────────────────
def train_all(
    symbols:   list = None,
    timesteps: int  = TOTAL_TIMESTEPS,
    date_from: str  = TRAIN_START,
    date_to:   str  = TRAIN_END,
    test_from: str  = TEST_START,
    test_to:   str  = TEST_END,
):
    symbols = symbols or SYMBOLS
    results = {}
    for sym in symbols:
        try:
            results[sym] = train_symbol(
                symbol    = sym,
                timesteps = timesteps,
                date_from = date_from,
                date_to   = date_to,
                test_from = test_from,
                test_to   = test_to,
            )
        except Exception as e:
            logger.error(f"[{sym}] Failed: {e}", exc_info=True)
            results[sym] = {"error": str(e)}

    logger.info("\n=== Training Summary ===")
    for sym, m in results.items():
        if "error" in m:
            logger.error(f"  {sym:10s}  ERROR: {m['error']}")
        else:
            logger.info(
                f"  {sym:10s}  "
                f"pnl={m.get('mean_pnl',0):+.4f}  "
                f"sharpe={m.get('mean_sharpe',0):.3f}  "
                f"win={m.get('win_rate',0):.2%}  "
                f"flips={m.get('mean_flips',0):.0f}"
            )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--from",      dest="date_from", default=TRAIN_START)
    parser.add_argument("--to",        dest="date_to",   default=TRAIN_END)
    parser.add_argument("--test-from", dest="test_from", default=TEST_START)
    parser.add_argument("--test-to",   dest="test_to",   default=TEST_END)
    args = parser.parse_args()
    train_all(
        symbols   = [args.symbol] if args.symbol else None,
        timesteps = args.timesteps,
        date_from = args.date_from,
        date_to   = args.date_to,
        test_from = args.test_from,
        test_to   = args.test_to,
    )
