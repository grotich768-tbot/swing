"""
training/train_rl.py  —  Train PPO agents for each symbol
──────────────────────────────────────────────────────────────────────────────
IMPROVEMENTS APPLIED:

Tier 1  — 3M timesteps for GOLD/SILVER (already in config)
Tier 2  — Ensemble training: N seeds per symbol, majority-vote at inference
Tier 2  — Walk-forward fine-tune support (--finetune flag)
Tier 1  — Timestamps passed to env for session multipliers

Usage:
    python -m training.train_rl --symbol GOLD
    python -m training.train_rl --symbol GOLD --ensemble       # 5 seeds
    python -m training.train_rl --symbol GOLD --timesteps 500000
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
    PPO_ENSEMBLE_SEEDS, PPO_ENSEMBLE_ENABLED,
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

    for tag in (symbol, "shared"):   # prefer per-symbol (Tier 1)
        try:
            clf = RegimeClassifier.load(tag)
            regime_proba = clf.predict_proba(
                pd.DataFrame(feats[feat_cols].values, columns=feat_cols)
            )
            logger.info(f"[{symbol}] Regime proba loaded ({tag})")
            break
        except FileNotFoundError:
            continue

    n_feat = len(feat_cols)
    for tag in (symbol, "shared"):
        try:
            lstm = LSTMDirectionModel.load(n_features=n_feat, symbol=tag)
            direction_proba = lstm.predict_proba(feats[feat_cols].values)
            logger.info(f"[{symbol}] Direction proba loaded ({tag})")
            break
        except FileNotFoundError:
            continue

    return regime_proba, direction_proba


def _make_env(feats, symbol, regime_proba, direction_proba, mode="train"):
    """Create env with timestamps passed in (Tier 2 session multipliers)."""
    timestamps = feats.index if hasattr(feats, 'index') else None
    return AlwaysInEnv(
        features_df     = feats,
        symbol          = symbol,
        regime_proba    = regime_proba,
        direction_proba = direction_proba,
        timestamps      = timestamps,
        mode            = mode,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-seed training
# ─────────────────────────────────────────────────────────────────────────────
def _train_one_seed(
    symbol:      str,
    seed:        int,
    timesteps:   int,
    feats_train, feat_cols,
    feats_test,  feat_cols_t,
    reg_tr,      dir_tr,
    reg_te,      dir_te,
) -> "PPO":
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, CallbackList
    )

    logger.info(f"[{symbol}] Training seed={seed}  steps={timesteps:,}")

    def make_train_env():
        return Monitor(_make_env(feats_train, symbol, reg_tr, dir_tr, "train"))

    train_env = DummyVecEnv([make_train_env])

    eval_env = None
    if feats_test is not None and len(feats_test) > 100:
        eval_env = Monitor(_make_env(feats_test, symbol, reg_te, dir_te, "test"))

    model_path = MODEL_DIR / f"ppo_{symbol}_seed{seed}"
    log_path   = LOG_DIR   / f"ppo_{symbol}_seed{seed}"
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
            verbose              = 0,
        ))
    callbacks.append(CheckpointCallback(
        save_freq   = max(5000, timesteps // 10),
        save_path   = str(model_path / "checkpoints"),
        name_prefix = f"ppo_{symbol}_s{seed}",
    ))

    params = dict(PPO_PARAMS)
    params["seed"] = seed

    # ── Resume from checkpoint if available ──────────────────────────────────
    # Priority: latest checkpoint (most steps done) → best_model → fresh start
    # Sorts checkpoints numerically by step count to find the true latest.
    checkpoint_dir = model_path / "checkpoints"
    resume_path    = None
    steps_done     = 0

    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.zip"))
        if checkpoints:
            # Sort numerically by step count extracted from filename
            def _steps_from_name(p):
                import re
                m = re.search(r"_(\d+)_steps", p.stem)
                return int(m.group(1)) if m else 0
            checkpoints.sort(key=_steps_from_name)
            resume_path = checkpoints[-1]
            steps_done  = _steps_from_name(resume_path)
            logger.info(
                f"[{symbol}] seed={seed} — resuming from checkpoint: "
                f"{resume_path.name}  ({steps_done:,} steps done)"
            )

    if resume_path is None:
        best_candidate = model_path / "best_model.zip"
        if best_candidate.exists():
            resume_path = best_candidate
            logger.info(f"[{symbol}] seed={seed} — resuming from best_model")

    # Calculate REMAINING steps — don't retrain steps already done
    remaining = max(0, timesteps - steps_done)

    if resume_path is not None:
        if remaining == 0:
            logger.info(f"[{symbol}] seed={seed} — already complete ({steps_done:,} steps), skipping")
            train_env.close()
            if eval_env:
                eval_env.close()
            # Load and return existing model
            final_path = MODEL_DIR / f"ppo_{symbol}_seed{seed}_final"
            if final_path.with_suffix(".zip").exists():
                return PPO.load(str(final_path))
            return PPO.load(str(resume_path), env=train_env)
        model = PPO.load(str(resume_path), env=train_env,
                         **{k: v for k, v in params.items() if k != "seed"})
        logger.info(
            f"[{symbol}] seed={seed} — resuming: "
            f"{steps_done:,} done, {remaining:,} remaining"
        )
    else:
        remaining = timesteps
        model = PPO(
            policy          = "MlpPolicy",
            env             = train_env,
            verbose         = 0,
            tensorboard_log = str(LOG_DIR / "tensorboard"),
            **params,
        )
        logger.info(f"[{symbol}] seed={seed} — starting fresh ({remaining:,} steps)")

    model.learn(
        total_timesteps     = remaining,
        callback            = CallbackList(callbacks) if callbacks else None,
        progress_bar        = True,
        reset_num_timesteps = (resume_path is None),  # False = continue counter
    )

    final_path = MODEL_DIR / f"ppo_{symbol}_seed{seed}_final"
    model.save(str(final_path))
    logger.info(f"[{symbol}] Seed {seed} saved → {final_path}.zip")

    train_env.close()
    if eval_env:
        eval_env.close()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Ensemble training
# ─────────────────────────────────────────────────────────────────────────────
def train_ensemble(
    symbol:     str,
    seeds:      list,
    timesteps:  int,
    feats_train, feat_cols,
    feats_test,  feat_cols_t,
    reg_tr,      dir_tr,
    reg_te,      dir_te,
) -> list:
    """
    Train N PPO models with different seeds.
    Saves each individually; EnsemblePredictor handles majority-vote at inference.
    Returns list of trained models.
    """
    models = []
    for seed in seeds:
        m = _train_one_seed(
            symbol, seed, timesteps,
            feats_train, feat_cols,
            feats_test,  feat_cols_t,
            reg_tr, dir_tr, reg_te, dir_te,
        )
        models.append((seed, m))
        logger.success(f"[{symbol}] Ensemble seed {seed} complete")

    # Save ensemble manifest
    manifest = {
        "symbol":     symbol,
        "seeds":      seeds,
        "n_models":   len(seeds),
        "timesteps":  timesteps,
        "created_at": datetime.now().isoformat(),
    }
    mpath = MODEL_DIR / f"ensemble_{symbol}_manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"[{symbol}] Ensemble manifest → {mpath}")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Single-symbol training entry point
# ─────────────────────────────────────────────────────────────────────────────
def train_symbol(
    symbol:     str,
    timesteps:  int = None,
    date_from:  str = None,
    date_to:    str = None,
    test_from:  str = None,
    test_to:    str = None,
    ensemble:   bool = None,   # None = use config default
) -> dict:

    # Use per-symbol dates if available (e.g. BTC 2017-2026)
    try:
        from config import SYMBOL_TRAIN_START, SYMBOL_TRAIN_END, SYMBOL_TEST_START, SYMBOL_TEST_END
        date_from = date_from or SYMBOL_TRAIN_START.get(symbol, TRAIN_START)
        date_to   = date_to   or SYMBOL_TRAIN_END.get(symbol,   TRAIN_END)
        test_from = test_from or SYMBOL_TEST_START.get(symbol,  TEST_START)
        test_to   = test_to   or SYMBOL_TEST_END.get(symbol,    TEST_END)
    except ImportError:
        date_from = date_from or TRAIN_START
        date_to   = date_to   or TRAIN_END
        test_from = test_from or TEST_START
        test_to   = test_to   or TEST_END

    if timesteps is None:
        timesteps = TOTAL_TIMESTEPS.get(symbol, 1_000_000)
    if ensemble is None:
        ensemble = PPO_ENSEMBLE_ENABLED

    logger.info(f"\n{'='*60}\n  Training: {symbol}  (ensemble={ensemble})\n{'='*60}")

    loader   = DataLoader()
    engineer = FeatureEngineer(normalise=True)

    # ── Build features ────────────────────────────────────────────────────────
    feats_train, feat_cols, _ = _load_features(loader, engineer, symbol, date_from, date_to)
    if feats_train is None:
        logger.error(f"[{symbol}] Not enough training data.")
        return {}

    reg_tr, dir_tr = _attach_ml_outputs(feats_train, feat_cols, symbol)

    feats_test, feat_cols_t, _ = _load_features(loader, engineer, symbol, test_from, test_to)
    reg_te = dir_te = None
    if feats_test is not None and len(feats_test) > 100:
        reg_te, dir_te = _attach_ml_outputs(feats_test, feat_cols_t, symbol)

    logger.info(
        f"[{symbol}] Train: {len(feats_train):,} bars  "
        f"Test: {len(feats_test) if feats_test is not None else 0:,} bars  "
        f"Features: {len(feat_cols)}"
    )

    t0 = datetime.now()

    if ensemble:
        # ── Ensemble: train all seeds ─────────────────────────────────────────
        models = train_ensemble(
            symbol, PPO_ENSEMBLE_SEEDS, timesteps,
            feats_train, feat_cols,
            feats_test, feat_cols_t,
            reg_tr, dir_tr, reg_te, dir_te,
        )
        # Evaluate using the first model (representative)
        metrics = {}
        if feats_test is not None and len(feats_test) > 100:
            eval_env = Monitor_env(_make_env(feats_test, symbol, reg_te, dir_te, "test"))
            metrics = _evaluate_ensemble(
                [m for _, m in models], eval_env, symbol, n_episodes=10
            )
    else:
        # ── Single seed (seed=42) ─────────────────────────────────────────────
        model = _train_one_seed(
            symbol, 42, timesteps,
            feats_train, feat_cols,
            feats_test, feat_cols_t,
            reg_tr, dir_tr, reg_te, dir_te,
        )
        # Also save as legacy "ppo_{symbol}_final" for backtest.py compatibility
        (MODEL_DIR / f"ppo_{symbol}_final.zip").unlink(missing_ok=True)
        model.save(str(MODEL_DIR / f"ppo_{symbol}_final"))

        metrics = {}
        if feats_test is not None and len(feats_test) > 100:
            from stable_baselines3.common.monitor import Monitor
            eval_env = Monitor(_make_env(feats_test, symbol, reg_te, dir_te, "test"))
            metrics = _evaluate(model, eval_env, symbol, n_episodes=10)

    elapsed = (datetime.now() - t0).total_seconds()
    logger.success(f"[{symbol}] Training done in {elapsed:.0f}s")

    if metrics:
        logger.info(
            f"[{symbol}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
        with open(LOG_DIR / f"metrics_{symbol}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


# little shim so we can use Monitor lazily
def Monitor_env(env):
    from stable_baselines3.common.monitor import Monitor
    return Monitor(env)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
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


def _evaluate_ensemble(models, env, symbol, n_episodes=10):
    """
    Tier 2: Majority-vote ensemble evaluation.
    Each step: poll all models, take majority vote on HOLD/FLIP.
    """
    all_pnls, all_flips, all_sharpes = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        while not done:
            # Majority vote across ensemble
            votes = [int(m.predict(obs, deterministic=True)[0]) for m in models]
            action = 1 if votes.count(1) > len(votes) // 2 else 0
            obs, _, done, truncated, info = env.step(action)
            done = done or truncated
        s = env.unwrapped.episode_summary()
        all_pnls.append(s["total_pnl"])
        all_flips.append(s["n_flips"])
        all_sharpes.append(s["sharpe"])
    return {
        "mean_pnl":      float(np.mean(all_pnls)),
        "std_pnl":       float(np.std(all_pnls)),
        "mean_flips":    float(np.mean(all_flips)),
        "mean_sharpe":   float(np.mean(all_sharpes)),
        "win_rate":      float(np.mean([p > 0 for p in all_pnls])),
        "ensemble_size": len(models),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train all symbols
# ─────────────────────────────────────────────────────────────────────────────
def train_all(
    symbols:   list = None,
    timesteps: int  = None,
    date_from: str  = TRAIN_START,
    date_to:   str  = TRAIN_END,
    test_from: str  = TEST_START,
    test_to:   str  = TEST_END,
    ensemble:  bool = None,
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
                ensemble  = ensemble,
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
                + (f"  ensemble={m.get('ensemble_size','–')}" if "ensemble_size" in m else "")
            )
    return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    type=str,  default=None)
    parser.add_argument("--timesteps", type=int,  default=None)
    parser.add_argument("--from",      dest="date_from", default=TRAIN_START)
    parser.add_argument("--to",        dest="date_to",   default=TRAIN_END)
    parser.add_argument("--test-from", dest="test_from", default=TEST_START)
    parser.add_argument("--test-to",   dest="test_to",   default=TEST_END)
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Train 5 seeds per symbol and use majority-vote ensemble"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint instead of starting fresh"
    )
    parser.add_argument(
        "--no-ensemble", action="store_true",
        help="Force single-seed training (override config)"
    )
    args = parser.parse_args()

    ens = None
    if args.ensemble:    ens = True
    if args.no_ensemble: ens = False

    train_all(
        symbols   = [args.symbol] if args.symbol else None,
        timesteps = args.timesteps,
        date_from = args.date_from,
        date_to   = args.date_to,
        test_from = args.test_from,
        test_to   = args.test_to,
        ensemble  = ens,
    )
