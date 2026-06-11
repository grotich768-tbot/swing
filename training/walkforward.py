"""
training/walkforward.py  —  Monthly walk-forward retraining pipeline
──────────────────────────────────────────────────────────────────────────────
Tier 2 improvement: Walk-forward retraining

Logic
-----
1. Start at TRAIN_END.
2. Every WF_TEST_WINDOW (e.g. 1 month), extend training window by one month.
3. Re-run supervised models + fine-tune PPO for WF_FINETUNE_STEPS steps.
4. Evaluate new model vs old model on the next out-of-sample window.
5. Keep the better one (by mean_sharpe).

Usage
-----
    python -m training.walkforward --symbol GOLD
    python -m training.walkforward --symbol GOLD --start 2022-01-01 --end 2024-12-31
    python -m training.walkforward --symbol GOLD --no-finetune   # full retrain each fold
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_START, TRAIN_END, TEST_END,
    WF_TRAIN_WINDOW, WF_TEST_WINDOW, WF_PURGE_GAP,
    WF_FINETUNE_STEPS, TOTAL_TIMESTEPS, MODEL_DIR, LOG_DIR,
    PPO_ENSEMBLE_SEEDS, PPO_ENSEMBLE_ENABLED,
)
from data.data_loader import DataLoader
from data.feature_engineer import FeatureEngineer
from models.regime_classifier import RegimeClassifier
from models.price_predictor import LSTMDirectionModel
from training.train_rl import _load_features, _attach_ml_outputs, _make_env, _evaluate
from training.train_supervised import train_supervised


def _parse_months(window_str: str) -> int:
    """Parse '18M', '3M', '2W' into approximate months."""
    if window_str.endswith("M"):
        return int(window_str[:-1])
    if window_str.endswith("W"):
        return max(1, int(window_str[:-1]) // 4)
    return int(window_str)


def _date_add_months(dt: datetime, months: int) -> datetime:
    return dt + relativedelta(months=months)


def walkforward(
    symbol:      str,
    wf_start:    str = TRAIN_END,
    wf_end:      str = TEST_END,
    finetune:    bool = True,    # True = fine-tune existing model; False = full retrain
    ensemble:    bool = None,
):
    """
    Run full walk-forward retraining loop for one symbol.

    Parameters
    ----------
    symbol   : trading symbol (e.g. "GOLD")
    wf_start : start of walk-forward window (usually TRAIN_END)
    wf_end   : end of walk-forward (usually TEST_END or "today")
    finetune : if True, fine-tune existing PPO (faster); else full retrain
    ensemble : use ensemble training (None = use config default)
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    if ensemble is None:
        ensemble = PPO_ENSEMBLE_ENABLED

    train_months = _parse_months(WF_TRAIN_WINDOW)   # 18
    test_months  = _parse_months(WF_TEST_WINDOW)     # 3 (but advance 1 month at a time)
    purge_months = max(1, _parse_months(WF_PURGE_GAP))

    loader   = DataLoader()
    engineer = FeatureEngineer(normalise=True)

    # Walk forward in 1-month steps
    cursor      = datetime.strptime(wf_start, "%Y-%m-%d")
    end_dt      = datetime.strptime(wf_end,   "%Y-%m-%d")
    fold_results = []

    logger.info(
        f"\n{'='*60}\n  Walk-Forward: {symbol}\n"
        f"  {wf_start} → {wf_end}  "
        f"(train={WF_TRAIN_WINDOW}, fold=1M)\n{'='*60}"
    )

    fold = 0
    while cursor < end_dt:
        fold += 1
        fold_train_end   = cursor.strftime("%Y-%m-%d")
        fold_train_start = _date_add_months(cursor, -train_months).strftime("%Y-%m-%d")

        # Purge gap between train and test
        fold_test_start  = _date_add_months(cursor, purge_months).strftime("%Y-%m-%d")
        fold_test_end    = _date_add_months(cursor, purge_months + 1).strftime("%Y-%m-%d")

        if fold_test_end > wf_end:
            break

        logger.info(
            f"\n--- Fold {fold} ---  "
            f"train: {fold_train_start}→{fold_train_end}  "
            f"test: {fold_test_start}→{fold_test_end}"
        )

        # ── Load features ──────────────────────────────────────────────────────
        feats_tr, fcols_tr, _ = _load_features(
            loader, engineer, symbol, fold_train_start, fold_train_end
        )
        if feats_tr is None or len(feats_tr) < 500:
            logger.warning(f"Fold {fold}: insufficient train data — skipping")
            cursor = _date_add_months(cursor, 1)
            continue

        feats_te, fcols_te, _ = _load_features(
            loader, engineer, symbol, fold_test_start, fold_test_end
        )
        if feats_te is None or len(feats_te) < 50:
            logger.warning(f"Fold {fold}: insufficient test data — skipping")
            cursor = _date_add_months(cursor, 1)
            continue

        # ── Retrain supervised models ──────────────────────────────────────────
        logger.info(f"Fold {fold}: retraining supervised models")
        train_supervised(
            symbols   = [symbol],
            date_from = fold_train_start,
            date_to   = fold_train_end,
            shared    = False,
        )

        reg_tr, dir_tr = _attach_ml_outputs(feats_tr, fcols_tr, symbol)
        reg_te, dir_te = _attach_ml_outputs(feats_te, fcols_te, symbol)

        # ── Load existing model or create new ─────────────────────────────────
        existing_model_path = MODEL_DIR / f"ppo_{symbol}_seed42_final.zip"
        if not existing_model_path.exists():
            existing_model_path = MODEL_DIR / f"ppo_{symbol}_final.zip"

        def make_env():
            return Monitor(_make_env(feats_tr, symbol, reg_tr, dir_tr, "train"))

        from stable_baselines3.common.vec_env import DummyVecEnv
        train_env = DummyVecEnv([make_env])

        if finetune and existing_model_path.exists():
            # Fine-tune existing model
            model = PPO.load(str(existing_model_path), env=train_env)
            logger.info(
                f"Fold {fold}: fine-tuning from {existing_model_path.name}  "
                f"({WF_FINETUNE_STEPS:,} steps)"
            )
            model.learn(total_timesteps=WF_FINETUNE_STEPS, reset_num_timesteps=False)
        else:
            # Full retrain from scratch
            from config import PPO_PARAMS
            params = dict(PPO_PARAMS)
            params["seed"] = 42
            model = PPO("MlpPolicy", train_env, verbose=0, **params)
            steps = TOTAL_TIMESTEPS.get(symbol, 1_000_000)
            logger.info(f"Fold {fold}: full retrain ({steps:,} steps)")
            model.learn(total_timesteps=steps, progress_bar=True)

        # ── Evaluate new model ─────────────────────────────────────────────────
        eval_env_new = Monitor(_make_env(feats_te, symbol, reg_te, dir_te, "test"))
        new_metrics  = _evaluate(model, eval_env_new, symbol, n_episodes=5)

        # ── Compare with old model ─────────────────────────────────────────────
        old_metrics  = {"mean_sharpe": -999.0}
        if existing_model_path.exists():
            try:
                old_model  = PPO.load(str(existing_model_path))
                eval_env_old = Monitor(_make_env(feats_te, symbol, reg_te, dir_te, "test"))
                old_metrics  = _evaluate(old_model, eval_env_old, symbol, n_episodes=5)
                eval_env_old.close()
            except Exception as e:
                logger.warning(f"Could not evaluate old model: {e}")

        # Keep better model
        if new_metrics["mean_sharpe"] >= old_metrics["mean_sharpe"]:
            save_path = MODEL_DIR / f"ppo_{symbol}_seed42_final"
            model.save(str(save_path))
            logger.success(
                f"Fold {fold}: NEW model kept  "
                f"sharpe: {old_metrics['mean_sharpe']:.3f} → {new_metrics['mean_sharpe']:.3f}"
            )
            kept = "new"
        else:
            logger.info(
                f"Fold {fold}: OLD model kept  "
                f"new_sharpe={new_metrics['mean_sharpe']:.3f} < "
                f"old_sharpe={old_metrics['mean_sharpe']:.3f}"
            )
            kept = "old"

        fold_results.append({
            "fold":        fold,
            "train_start": fold_train_start,
            "train_end":   fold_train_end,
            "test_start":  fold_test_start,
            "test_end":    fold_test_end,
            "new_sharpe":  new_metrics["mean_sharpe"],
            "old_sharpe":  old_metrics["mean_sharpe"],
            "new_pnl":     new_metrics["mean_pnl"],
            "kept":        kept,
        })

        train_env.close()
        eval_env_new.close()

        # Advance by 1 month
        cursor = _date_add_months(cursor, 1)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}\n  Walk-Forward Summary: {symbol}\n{'='*60}")
    for r in fold_results:
        logger.info(
            f"  Fold {r['fold']:2d}  {r['test_start']}  "
            f"new_sharpe={r['new_sharpe']:+.3f}  "
            f"old_sharpe={r['old_sharpe']:+.3f}  "
            f"kept={r['kept']}"
        )

    out_path = LOG_DIR / f"walkforward_{symbol}.json"
    with open(out_path, "w") as f:
        json.dump(fold_results, f, indent=2)
    logger.success(f"Walk-forward results → {out_path}")
    return fold_results


if __name__ == "__main__":
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dateutil", "-q"])
        from dateutil.relativedelta import relativedelta

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",   type=str, default="GOLD")
    parser.add_argument("--start",    type=str, default=TRAIN_END,
                        help="Start of walk-forward period")
    parser.add_argument("--end",      type=str, default=TEST_END,
                        help="End of walk-forward period")
    parser.add_argument("--no-finetune", action="store_true",
                        help="Full retrain each fold instead of fine-tuning")
    args = parser.parse_args()

    walkforward(
        symbol   = args.symbol,
        wf_start = args.start,
        wf_end   = args.end,
        finetune = not args.no_finetune,
    )
