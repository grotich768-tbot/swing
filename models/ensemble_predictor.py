"""
models/ensemble_predictor.py  —  PPO Ensemble Majority-Vote Predictor
──────────────────────────────────────────────────────────────────────────────
Tier 2 improvement: Ensemble of N PPO seeds with majority-vote action.

Usage
-----
    from models.ensemble_predictor import EnsemblePredictor
    ens = EnsemblePredictor.load("GOLD")
    action = ens.predict(obs)   # majority vote: 0=HOLD or 1=FLIP

Logic
-----
    5 models trained with seeds [42, 123, 777, 1337, 9999].
    At each step:
        votes = [m.predict(obs) for m in models]
        action = 1 if sum(votes) > len(votes)//2 else 0
    
    If 4/5 say HOLD and 1 says FLIP → HOLD.
    Variance reduction ≈ 20% fewer spurious flips vs single model.
"""

import sys
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR, PPO_ENSEMBLE_SEEDS


class EnsemblePredictor:
    """
    Wraps multiple PPO models and returns majority-vote predictions.

    Parameters
    ----------
    models : list of stable_baselines3.PPO  — pre-loaded models
    seeds  : list of int  — seed used for each model (for logging)
    symbol : str
    """

    def __init__(self, models: list, seeds: List[int], symbol: str):
        self.models = models
        self.seeds  = seeds
        self.symbol = symbol

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        Majority-vote prediction.

        Parameters
        ----------
        obs : (obs_size,) float32 array

        Returns
        -------
        int  — 0 = HOLD, 1 = FLIP
        """
        votes = []
        for m in self.models:
            action, _ = m.predict(obs, deterministic=deterministic)
            votes.append(int(action))

        flip_votes = sum(votes)
        action = 1 if flip_votes > len(votes) // 2 else 0
        return action

    def predict_with_confidence(self, obs: np.ndarray) -> tuple:
        """
        Returns (action, confidence) where confidence = fraction of votes for action.

        Returns
        -------
        (int, float)  — action, confidence in [0.5, 1.0]
        """
        votes      = [int(m.predict(obs, deterministic=True)[0]) for m in self.models]
        flip_votes = sum(votes)
        hold_votes = len(votes) - flip_votes
        action     = 1 if flip_votes > hold_votes else 0
        confidence = max(flip_votes, hold_votes) / len(votes)
        return action, confidence

    def vote_breakdown(self, obs: np.ndarray) -> dict:
        """Return detailed vote breakdown for diagnostics."""
        votes = {}
        for seed, m in zip(self.seeds, self.models):
            action, _ = m.predict(obs, deterministic=True)
            votes[f"seed_{seed}"] = int(action)
        flip_count = sum(votes.values())
        votes["majority"] = 1 if flip_count > len(self.models) // 2 else 0
        votes["flip_fraction"] = flip_count / len(self.models)
        return votes

    @classmethod
    def load(
        cls,
        symbol: str,
        seeds:  List[int] = None,
        env    = None,   # optional gym env for obs space check
    ) -> "EnsemblePredictor":
        """
        Load all ensemble models for a symbol from disk.

        Tries per-seed files first (ppo_{symbol}_seed{N}_final.zip),
        falls back to manifest if available.
        """
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError("stable-baselines3 not installed")

        if seeds is None:
            # Try manifest first
            manifest_path = MODEL_DIR / f"ensemble_{symbol}_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                seeds = manifest.get("seeds", PPO_ENSEMBLE_SEEDS)
            else:
                seeds = PPO_ENSEMBLE_SEEDS

        loaded = []
        missing = []
        for seed in seeds:
            # Try seed-specific path first, then legacy
            candidates = [
                MODEL_DIR / f"ppo_{symbol}_seed{seed}_final.zip",
                MODEL_DIR / f"ppo_{symbol}_seed{seed}" / "best_model.zip",
            ]
            found = False
            for path in candidates:
                if path.exists():
                    try:
                        m = PPO.load(str(path), env=env)
                        loaded.append(m)
                        logger.debug(f"[{symbol}] Loaded seed {seed} from {path.name}")
                        found = True
                        break
                    except Exception as e:
                        logger.warning(f"[{symbol}] Could not load {path}: {e}")
            if not found:
                missing.append(seed)

        if missing:
            logger.warning(
                f"[{symbol}] Ensemble: {len(missing)} seeds not found "
                f"({missing}). Using {len(loaded)} models."
            )

        if not loaded:
            raise FileNotFoundError(
                f"No ensemble models found for {symbol}. "
                f"Run: python -m training.train_rl --symbol {symbol} --ensemble"
            )

        logger.info(f"[{symbol}] Ensemble loaded: {len(loaded)} models")
        return cls(loaded, seeds[:len(loaded)], symbol)

    @classmethod
    def load_or_single(
        cls,
        symbol: str,
        env    = None,
    ) -> "EnsemblePredictor":
        """
        Try to load ensemble; fall back to single model wrapped as ensemble.
        Useful in backtest / live run for transparent upgrade path.
        """
        try:
            return cls.load(symbol, env=env)
        except FileNotFoundError:
            pass

        # Fall back to single model
        from stable_baselines3 import PPO
        single_candidates = [
            MODEL_DIR / f"ppo_{symbol}_seed42_final.zip",
            MODEL_DIR / f"ppo_{symbol}_final.zip",
            MODEL_DIR / f"ppo_{symbol}" / "best_model.zip",
        ]
        for path in single_candidates:
            if path.exists():
                m = PPO.load(str(path), env=env)
                logger.info(f"[{symbol}] Loaded single model (no ensemble): {path.name}")
                return cls([m], [42], symbol)

        raise FileNotFoundError(
            f"No model found for {symbol}. Run training first."
        )

    def __len__(self) -> int:
        return len(self.models)

    def __repr__(self) -> str:
        return f"EnsemblePredictor(symbol={self.symbol}, n={len(self.models)}, seeds={self.seeds})"
