"""
training/train_regime_agents.py  —  Regime-Specific Agent Training (Item #10)
──────────────────────────────────────────────────────────────────────────────
Trains 4 specialist PPO agents per symbol — one per regime:
  0 = Range-LowVol      (~40% of bars)
  1 = Trend-Up          (~25% of bars)
  2 = Trend-Down        (~25% of bars)
  3 = Breakout-HighVol  (~10% of bars)

At inference the XGBoost classifier picks the right specialist.
The 5-seed ensemble acts as safety net when regime is ambiguous.

Usage:
    python -m training.train_regime_agents --symbol GOLD
    python -m training.train_regime_agents --symbol GOLD --regime 0
    python -m training.train_regime_agents --symbol GOLD --all-parallel
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
    PPO_PARAMS, MODEL_DIR, LOG_DIR, N_REGIMES,
)
from data.data_loader import DataLoader
from data.feature_engineer import FeatureEngineer
from models.regime_classifier import RegimeClassifier, REGIME_NAMES
from models.price_predictor import LSTMDirectionModel
from env.trading_env import AlwaysInEnv
from training.train_rl import _load_features, _attach_ml_outputs, _make_env, _evaluate


# Default timesteps per regime — overridden per symbol via config.REGIME_TIMESTEPS_OVERRIDE
REGIME_TIMESTEPS = {
    0: 1_500_000,   # Range     — most common
    1: 1_000_000,   # Trend-Up
    2: 1_000_000,   # Trend-Down
    # Regime 3 (Breakout): skipped — too rare in most symbols
}

def _get_regime_timesteps(symbol: str, regime_id: int) -> int:
    """Get timesteps for a regime, using per-symbol override if available."""
    try:
        from config import REGIME_TIMESTEPS_OVERRIDE
        if symbol in REGIME_TIMESTEPS_OVERRIDE:
            return REGIME_TIMESTEPS_OVERRIDE[symbol].get(regime_id, REGIME_TIMESTEPS.get(regime_id, 1_000_000))
    except ImportError:
        pass
    return REGIME_TIMESTEPS.get(regime_id, 1_000_000)


# ─────────────────────────────────────────────────────────────────────────────
# Regime-filtered environment
# ─────────────────────────────────────────────────────────────────────────────
class RegimeFilteredEnv(AlwaysInEnv):
    """
    AlwaysInEnv subclass that only starts episodes on bars of a specific regime.
    The agent is thus trained exclusively on its specialist regime conditions.

    Parameters
    ----------
    regime_id      : int — which regime to specialise on (0-3)
    regime_labels  : np.ndarray — per-bar regime labels aligned to features_df
    All other params passed to AlwaysInEnv.
    """

    def __init__(self, regime_id: int, regime_labels: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self._regime_id     = regime_id
        self._regime_labels = regime_labels

        # Pre-compute valid start indices — bars where regime == regime_id
        self._regime_starts = np.where(
            regime_labels == regime_id
        )[0].tolist()

        # Filter out starts too close to boundaries
        from config import LOOKBACK_STEPS, MAX_EPISODE_STEPS
        self._regime_starts = [
            i for i in self._regime_starts
            if LOOKBACK_STEPS <= i <= len(regime_labels) - MAX_EPISODE_STEPS - 1
        ]

        if not self._regime_starts:
            raise ValueError(
                f"No valid start bars for regime {regime_id} "
                f"({REGIME_NAMES[regime_id]})"
            )

        logger.debug(
            f"[{self.symbol}] Regime {regime_id} ({REGIME_NAMES[regime_id]}): "
            f"{len(self._regime_starts):,} valid start bars"
        )

    def reset(self, seed=None, options=None):
        # Override: always start on a regime-specific bar
        super().reset(seed=seed, options=options)
        if self.mode == "train" and self._regime_starts:
            idx = int(
                self.np_random.integers(0, len(self._regime_starts))
            )
            self._episode_start = self._regime_starts[idx]
            self._step_idx      = self._episode_start
            self._entry_price   = self._close[self._step_idx]
        return self._get_obs(), {}


# ─────────────────────────────────────────────────────────────────────────────
# Train one regime agent
# ─────────────────────────────────────────────────────────────────────────────
def train_regime_agent(
    symbol:        str,
    regime_id:     int,
    feats_train:   pd.DataFrame,
    regime_labels: np.ndarray,
    reg_tr,        dir_tr,
    feats_test:    pd.DataFrame = None,
    reg_te=None,   dir_te=None,
    timesteps:     int = None,
) -> "PPO":
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

    regime_name = REGIME_NAMES[regime_id]
    if timesteps is None:
        timesteps = _get_regime_timesteps(symbol, regime_id)

    logger.info(
        f"\n[{symbol}] Regime {regime_id} ({regime_name})  "
        f"steps={timesteps:,}"
    )

    # Count bars in this regime
    n_regime_bars = int((regime_labels == regime_id).sum())
    if n_regime_bars < 500:
        logger.warning(
            f"[{symbol}] Regime {regime_id} has only {n_regime_bars} bars "
            f"— skipping (need ≥500)"
        )
        return None

    logger.info(
        f"[{symbol}] Regime {regime_id}: {n_regime_bars:,} bars "
        f"({n_regime_bars/len(regime_labels):.1%} of training data)"
    )

    timestamps = feats_train.index if hasattr(feats_train, 'index') else None

    def make_env():
        return Monitor(RegimeFilteredEnv(
            regime_id      = regime_id,
            regime_labels  = regime_labels,
            features_df    = feats_train,
            symbol         = symbol,
            regime_proba   = reg_tr,
            direction_proba= dir_tr,
            timestamps     = timestamps,
            mode           = "train",
        ))

    train_env  = DummyVecEnv([make_env])
    model_path = MODEL_DIR / f"ppo_{symbol}_regime{regime_id}"
    log_path   = LOG_DIR   / f"ppo_{symbol}_regime{regime_id}"
    log_path.mkdir(parents=True, exist_ok=True)

    callbacks = []

    # Eval on general test env (not regime-filtered) — tests real OOS performance
    if feats_test is not None and len(feats_test) > 100:
        eval_env = Monitor(_make_env(feats_test, symbol, reg_te, dir_te, "test"))
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path = str(model_path),
            log_path             = str(log_path),
            eval_freq            = max(1000, timesteps // 15),
            n_eval_episodes      = 5,
            deterministic        = True,
            verbose              = 0,
        ))

    callbacks.append(CheckpointCallback(
        save_freq   = max(5000, timesteps // 8),
        save_path   = str(model_path / "checkpoints"),
        name_prefix = f"ppo_{symbol}_r{regime_id}",
    ))

    params = dict(PPO_PARAMS)
    params["seed"] = 42 + regime_id   # deterministic but different per regime

    # ── Resume from checkpoint if available ──────────────────────────────────
    import re as _re
    checkpoint_dir = model_path / "checkpoints"
    resume_path    = None
    steps_done     = 0

    def _steps_from_name(p):
        m = _re.search(r"_(\d+)_steps", p.stem)
        return int(m.group(1)) if m else 0

    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.zip"))
        if checkpoints:
            checkpoints.sort(key=_steps_from_name)
            resume_path = checkpoints[-1]
            steps_done  = _steps_from_name(resume_path)
            logger.info(
                f"[{symbol}] Regime {regime_id} — resuming from checkpoint: "
                f"{resume_path.name} ({steps_done:,} steps done)"
            )

    if resume_path is None:
        best_candidate = model_path / "best_model.zip"
        if best_candidate.exists():
            resume_path = best_candidate
            logger.info(
                f"[{symbol}] Regime {regime_id} — resuming from best_model"
            )

    # Calculate remaining steps
    remaining = max(0, timesteps - steps_done)

    if resume_path is not None:
        if remaining == 0:
            logger.info(
                f"[{symbol}] Regime {regime_id} ({regime_name}) — "
                f"already complete ({steps_done:,} steps), skipping"
            )
            train_env.close()
            final_path = MODEL_DIR / f"ppo_{symbol}_regime{regime_id}_final"
            if final_path.with_suffix(".zip").exists():
                return PPO.load(str(final_path))
            return PPO.load(str(resume_path), env=train_env)
        model = PPO.load(str(resume_path), env=train_env,
                         **{k: v for k, v in params.items() if k != "seed"})
        logger.info(
            f"[{symbol}] Regime {regime_id} — "
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
        logger.info(
            f"[{symbol}] Regime {regime_id} ({regime_name}) — "
            f"starting fresh ({remaining:,} steps)"
        )

    model.learn(
        total_timesteps     = remaining,
        callback            = CallbackList(callbacks) if callbacks else None,
        progress_bar        = True,
        reset_num_timesteps = (resume_path is None),
    )

    # Save final
    final_path = MODEL_DIR / f"ppo_{symbol}_regime{regime_id}_final"
    model.save(str(final_path))
    logger.success(
        f"[{symbol}] Regime {regime_id} ({regime_name}) saved → {final_path}.zip"
    )

    train_env.close()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Train all 4 regime agents for one symbol
# ─────────────────────────────────────────────────────────────────────────────
def train_all_regime_agents(
    symbol:    str,
    date_from: str = None,
    date_to:   str = None,
    test_from: str = None,
    test_to:   str = None,
    regimes:   list = None,   # None = all 4
) -> dict:

    regimes = regimes if regimes is not None else [0, 1, 2]  # Regime 3 (Breakout) skipped — only 354 bars

    # Resolve per-symbol date overrides (e.g. BTCUSD starts 2017 not 2002)
    try:
        from config import (SYMBOL_TRAIN_START, SYMBOL_TRAIN_END,
                            SYMBOL_TEST_START,  SYMBOL_TEST_END)
        date_from = date_from or SYMBOL_TRAIN_START.get(symbol, TRAIN_START)
        date_to   = date_to   or SYMBOL_TRAIN_END.get(symbol,   TRAIN_END)
        test_from = test_from or SYMBOL_TEST_START.get(symbol,  TEST_START)
        test_to   = test_to   or SYMBOL_TEST_END.get(symbol,    TEST_END)
    except ImportError:
        date_from = date_from or TRAIN_START
        date_to   = date_to   or TRAIN_END
        test_from = test_from or TEST_START
        test_to   = test_to   or TEST_END

    logger.info(
        f"\n{'='*60}\n"
        f"  Regime-Specific Agents: {symbol}  regimes={regimes}\n"
        f"{'='*60}"
    )

    loader   = DataLoader()
    engineer = FeatureEngineer(normalise=True)

    # ── Load features ─────────────────────────────────────────────────────────
    feats_train, fcols, raw_h1 = _load_features(
        loader, engineer, symbol, date_from, date_to
    )
    if feats_train is None:
        logger.error(f"[{symbol}] No training data")
        return {}

    # Build regime labels aligned to feature index
    raw_h1_train = loader.load(symbol, "H1", date_from, date_to)
    regime_series = FeatureEngineer.regime_label(raw_h1_train)
    regime_series = regime_series.reindex(feats_train.index).fillna(0).astype(int)
    regime_labels = regime_series.values

    # Log regime distribution
    unique, counts = np.unique(regime_labels, return_counts=True)
    dist = {REGIME_NAMES[int(r)]: int(c) for r, c in zip(unique, counts)}
    logger.info(f"[{symbol}] Regime distribution: {dist}")

    reg_tr, dir_tr = _attach_ml_outputs(feats_train, fcols, symbol)

    feats_test, fcols_t, _ = _load_features(
        loader, engineer, symbol, test_from, test_to
    )
    reg_te = dir_te = None
    if feats_test is not None and len(feats_test) > 100:
        reg_te, dir_te = _attach_ml_outputs(feats_test, fcols_t, symbol)

    # ── Train each regime agent ───────────────────────────────────────────────
    results  = {}
    t0       = datetime.now()

    for regime_id in regimes:
        try:
            model = train_regime_agent(
                symbol        = symbol,
                regime_id     = regime_id,
                feats_train   = feats_train,
                regime_labels = regime_labels,
                reg_tr        = reg_tr,
                dir_tr        = dir_tr,
                feats_test    = feats_test,
                reg_te        = reg_te,
                dir_te        = dir_te,
            )
            if model is not None and feats_test is not None:
                from stable_baselines3.common.monitor import Monitor
                eval_env = Monitor(
                    _make_env(feats_test, symbol, reg_te, dir_te, "test")
                )
                metrics = _evaluate(model, eval_env, symbol, n_episodes=5)
                results[regime_id] = metrics
                eval_env.close()
                logger.info(
                    f"[{symbol}] Regime {regime_id} ({REGIME_NAMES[regime_id]})  "
                    + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                )
        except Exception as e:
            logger.error(
                f"[{symbol}] Regime {regime_id} failed: {e}", exc_info=True
            )
            results[regime_id] = {"error": str(e)}

    # Save manifest
    elapsed  = (datetime.now() - t0).total_seconds()
    manifest = {
        "symbol":           symbol,
        "regimes_trained":  [r for r in regimes if r in results and "error" not in results[r]],
        "regime_dist":      {str(k): v for k, v in dist.items()},
        "results":          {str(k): v for k, v in results.items()},
        "train_period":     f"{date_from} → {date_to}",
        "elapsed_seconds":  elapsed,
        "created_at":       datetime.now().isoformat(),
    }
    mpath = MODEL_DIR / f"regime_agents_{symbol}_manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.success(
        f"[{symbol}] All regime agents done in {elapsed/60:.1f}min  "
        f"→ {mpath}"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train regime-specific PPO agents"
    )
    parser.add_argument("--symbol",   type=str, default="GOLD")
    parser.add_argument("--regime",   type=int, default=None,
                        help="Single regime to train (0-3). Default: all 4.")
    parser.add_argument("--from",     dest="date_from", default=None,
                        help="Training start — defaults to SYMBOL_TRAIN_START[symbol] or TRAIN_START")
    parser.add_argument("--to",       dest="date_to",   default=None,
                        help="Training end — defaults to SYMBOL_TRAIN_END[symbol] or TRAIN_END")
    parser.add_argument("--test-from",dest="test_from", default=None,
                        help="Test start — defaults to SYMBOL_TEST_START[symbol] or TEST_START")
    parser.add_argument("--test-to",  dest="test_to",   default=None,
                        help="Test end — defaults to SYMBOL_TEST_END[symbol] or TEST_END")
    args = parser.parse_args()

    train_all_regime_agents(
        symbol    = args.symbol,
        date_from = args.date_from,
        date_to   = args.date_to,
        test_from = args.test_from,
        test_to   = args.test_to,
        regimes   = [args.regime] if args.regime is not None else [0, 1, 2],
    )
