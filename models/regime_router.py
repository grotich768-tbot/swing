"""
models/regime_router.py  —  Regime-Specific Agent Router
──────────────────────────────────────────────────────────────────────────────
At inference, routes each observation to the correct specialist agent:

  XGBoost classifies regime → picks specialist agent
  Ensemble of 5 generalist seeds → safety net / override

Decision logic per bar:
  1. XGBoost predicts regime (0-3) and confidence
  2. If confidence >= REGIME_CONFIDENCE_THRESHOLD:
       use specialist agent
  3. If ensemble and generalist STRONGLY disagree with specialist (4/5 vs 1):
       override with ensemble
  4. Otherwise: trust specialist

Usage:
    router = RegimeRouter.load("GOLD")
    action = router.predict(obs)
    info   = router.predict_verbose(obs)   # includes routing decision
"""

import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR, N_REGIMES, PPO_ENSEMBLE_SEEDS
from models.regime_classifier import REGIME_NAMES

# Minimum XGBoost confidence to trust a specialist over the ensemble
REGIME_CONFIDENCE_THRESHOLD = 0.55   # 55% probability for predicted regime

# Ensemble override: if specialist says FLIP but >=4/5 ensemble say HOLD → hold
ENSEMBLE_OVERRIDE_VOTES = 4   # out of 5


class RegimeRouter:
    """
    Routes observations to regime-specific specialist agents.
    Falls back to ensemble generalist when regime is ambiguous.

    Parameters
    ----------
    specialists  : dict {regime_id: PPO model}
    ensemble     : EnsemblePredictor | None
    classifier   : RegimeClassifier — XGBoost regime predictor
    symbol       : str
    """

    def __init__(
        self,
        specialists: dict,
        ensemble,
        classifier,
        symbol: str,
    ):
        self.specialists  = specialists   # {0: model, 1: model, ...}
        self.ensemble     = ensemble
        self.classifier   = classifier
        self.symbol       = symbol

        available = [REGIME_NAMES[r] for r in sorted(specialists.keys())]
        has_ens   = ensemble is not None
        logger.info(
            f"[{symbol}] RegimeRouter ready  "
            f"specialists={available}  ensemble={has_ens}"
        )

    # ── Main prediction ───────────────────────────────────────────────────────
    def predict(self, obs: np.ndarray, feat_row: np.ndarray = None) -> int:
        """
        Predict action for one observation.

        Parameters
        ----------
        obs      : RL observation vector (for PPO models)
        feat_row : raw feature vector for XGBoost regime prediction.
                   If None, uses ensemble only.

        Returns
        -------
        int — 0=HOLD, 1=FLIP
        """
        info = self.predict_verbose(obs, feat_row)
        return info["action"]

    def predict_verbose(
        self,
        obs:      np.ndarray,
        feat_row: np.ndarray = None,
    ) -> dict:
        """
        Full routing decision with diagnostics.

        Returns dict with:
          action          : 0 or 1
          regime_id       : predicted regime (0-3)
          regime_name     : e.g. "Trend-Up"
          regime_conf     : XGBoost confidence (0-1)
          source          : "specialist" | "ensemble" | "ensemble_override"
          specialist_vote : action from specialist (if used)
          ensemble_votes  : list of actions from each ensemble model
        """
        result = {
            "action":          0,
            "regime_id":       -1,
            "regime_name":     "unknown",
            "regime_conf":     0.0,
            "source":          "ensemble",
            "specialist_vote": None,
            "ensemble_votes":  [],
        }

        # ── Step 1: Regime classification ─────────────────────────────────────
        regime_id   = -1
        regime_conf = 0.0
        if feat_row is not None and self.classifier is not None:
            try:
                proba       = self.classifier.predict_proba_row(feat_row)
                regime_id   = int(np.argmax(proba))
                regime_conf = float(proba[regime_id])
                result["regime_id"]   = regime_id
                result["regime_name"] = REGIME_NAMES.get(regime_id, str(regime_id))
                result["regime_conf"] = regime_conf
            except Exception as e:
                logger.debug(f"Regime classification failed: {e}")

        # ── Step 2: Specialist prediction ─────────────────────────────────────
        specialist_action = None
        has_specialist    = (
            regime_id >= 0
            and regime_id in self.specialists
            and regime_conf >= REGIME_CONFIDENCE_THRESHOLD
        )
        if has_specialist:
            try:
                spec_model        = self.specialists[regime_id]
                spec_action, _    = spec_model.predict(obs, deterministic=True)
                specialist_action = int(spec_action)
                result["specialist_vote"] = specialist_action
            except Exception as e:
                logger.debug(f"Specialist prediction failed: {e}")
                has_specialist = False

        # ── Step 3: Ensemble prediction ───────────────────────────────────────
        ensemble_votes  = []
        ensemble_action = None
        if self.ensemble is not None:
            try:
                for m in self.ensemble.models:
                    v, _ = m.predict(obs, deterministic=True)
                    ensemble_votes.append(int(v))
                flip_count      = sum(ensemble_votes)
                ensemble_action = 1 if flip_count > len(ensemble_votes) // 2 else 0
                result["ensemble_votes"] = ensemble_votes
            except Exception as e:
                logger.debug(f"Ensemble prediction failed: {e}")

        # ── Step 4: Routing decision ──────────────────────────────────────────
        if not has_specialist:
            # No specialist available — use ensemble
            action = ensemble_action if ensemble_action is not None else 0
            result["source"] = "ensemble"

        elif ensemble_action is None:
            # No ensemble — use specialist directly
            action = specialist_action
            result["source"] = "specialist"

        else:
            # Both available — check for strong ensemble override
            flip_votes = sum(ensemble_votes)
            hold_votes = len(ensemble_votes) - flip_votes

            strong_hold = hold_votes >= ENSEMBLE_OVERRIDE_VOTES
            strong_flip = flip_votes >= ENSEMBLE_OVERRIDE_VOTES

            if specialist_action == 1 and strong_hold:
                # Specialist says FLIP but ensemble strongly disagrees → HOLD
                action = 0
                result["source"] = "ensemble_override"
                logger.debug(
                    f"[{self.symbol}] Override: specialist FLIP blocked by "
                    f"ensemble ({hold_votes}/5 hold)"
                )
            elif specialist_action == 0 and strong_flip:
                # Specialist says HOLD but ensemble strongly says FLIP → FLIP
                action = 1
                result["source"] = "ensemble_override"
                logger.debug(
                    f"[{self.symbol}] Override: specialist HOLD overridden by "
                    f"ensemble ({flip_votes}/5 flip)"
                )
            else:
                # Normal case: trust specialist
                action = specialist_action
                result["source"] = "specialist"

        result["action"] = int(action)
        return result

    # ── Load ──────────────────────────────────────────────────────────────────
    @classmethod
    def load(
        cls,
        symbol:    str,
        regimes:   list = None,   # None = load all available
        load_ensemble: bool = True,
    ) -> "RegimeRouter":
        from stable_baselines3 import PPO
        from models.ensemble_predictor import EnsemblePredictor
        from models.regime_classifier import RegimeClassifier

        regimes = regimes or list(range(N_REGIMES))

        # Load specialist agents
        specialists = {}
        for r in regimes:
            candidates = [
                MODEL_DIR / f"ppo_{symbol}_regime{r}_final.zip",
                MODEL_DIR / f"ppo_{symbol}_regime{r}" / "best_model.zip",
            ]
            for path in candidates:
                if path.exists():
                    try:
                        specialists[r] = PPO.load(str(path))
                        logger.info(
                            f"[{symbol}] Loaded regime {r} "
                            f"({REGIME_NAMES[r]}) from {path.name}"
                        )
                        break
                    except Exception as e:
                        logger.warning(f"Could not load regime {r}: {e}")

        if not specialists:
            raise FileNotFoundError(
                f"No regime agents found for {symbol}. "
                f"Run: python -m training.train_regime_agents --symbol {symbol}"
            )

        # Load ensemble (generalist safety net)
        ensemble = None
        if load_ensemble:
            try:
                ensemble = EnsemblePredictor.load_or_single(symbol)
            except FileNotFoundError:
                logger.warning(
                    f"[{symbol}] No ensemble found — "
                    f"router will use specialists only"
                )

        # Load regime classifier
        classifier = None
        for tag in (symbol, "shared"):
            try:
                classifier = RegimeClassifier.load(tag)
                break
            except FileNotFoundError:
                continue
        if classifier is None:
            logger.warning(
                f"[{symbol}] No regime classifier found — "
                f"routing will use ensemble only"
            )

        return cls(specialists, ensemble, classifier, symbol)

    @classmethod
    def load_or_ensemble(cls, symbol: str) -> "RegimeRouter":
        """
        Try to load full router; fall back to ensemble-only router.
        Safe to call at any stage of training.
        """
        try:
            return cls.load(symbol)
        except FileNotFoundError:
            pass

        # Fall back to ensemble only (no specialists)
        from models.ensemble_predictor import EnsemblePredictor
        from models.regime_classifier import RegimeClassifier

        try:
            ensemble = EnsemblePredictor.load_or_single(symbol)
        except FileNotFoundError:
            ensemble = None
        classifier = None
        for tag in (symbol, "shared"):
            try:
                classifier = RegimeClassifier.load(tag)
                break
            except FileNotFoundError:
                continue

        logger.info(
            f"[{symbol}] RegimeRouter: no specialists found, "
            f"using ensemble only"
        )
        return cls({}, ensemble, classifier, symbol)

    # ── Stats ──────────────────────────────────────────────────────────────────
    def routing_stats(self, n_obs: int = 100, obs_dim: int = 132) -> dict:
        """
        Sample N random observations to estimate routing distribution.
        Useful for sanity checking before deployment.
        """
        np.random.seed(42)
        sources = {"specialist": 0, "ensemble": 0, "ensemble_override": 0}
        regimes = {r: 0 for r in range(N_REGIMES)}

        for _ in range(n_obs):
            obs      = np.random.randn(obs_dim).astype(np.float32)
            feat_row = np.random.randn(62).astype(np.float32)   # approx feature size
            info     = self.predict_verbose(obs, feat_row)
            sources[info.get("source", "ensemble")] += 1
            r = info.get("regime_id", -1)
            if r >= 0:
                regimes[r] = regimes.get(r, 0) + 1

        return {
            "routing":  {k: v/n_obs for k, v in sources.items()},
            "regimes":  {REGIME_NAMES.get(r, str(r)): v/n_obs
                         for r, v in regimes.items()},
        }

    def __repr__(self) -> str:
        specs = [REGIME_NAMES[r] for r in sorted(self.specialists.keys())]
        return (
            f"RegimeRouter(symbol={self.symbol}, "
            f"specialists={specs}, "
            f"ensemble={self.ensemble is not None})"
        )
