"""
models/regime_classifier.py  —  XGBoost market regime classifier
──────────────────────────────────────────────────────────────────
Detects one of four market regimes per bar:
  0 = Low-volatility range
  1 = Trending up
  2 = Trending down
  3 = High-volatility breakout

The predicted regime probabilities are added to the RL agent's
observation vector so it can adapt behaviour per regime.
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR, REGIME_PARAMS, N_REGIMES


REGIME_NAMES = {
    0: "Range-LowVol",
    1: "Trend-Up",
    2: "Trend-Down",
    3: "Breakout-HighVol",
}


class RegimeClassifier:
    """
    Trains one XGBoost classifier per symbol (or a shared one).

    Parameters
    ----------
    symbol : str
        Used for model file naming. Pass "shared" to train on all symbols.
    """

    def __init__(self, symbol: str = "shared"):
        self.symbol  = symbol
        self.model   = None
        self.encoder = LabelEncoder()
        self._model_path = MODEL_DIR / f"regime_{symbol}.pkl"

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_fraction: float = 0.15,
    ) -> "RegimeClassifier":
        """
        Train XGBoost on feature matrix X with regime labels y.

        Parameters
        ----------
        X : feature DataFrame (output of FeatureEngineer.transform, private cols removed)
        y : integer regime labels (0-3), aligned with X
        eval_fraction : fraction of data used as early-stopping eval set
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost not installed: pip install xgboost")

        # Align and drop NaNs
        combined = pd.concat([X, y.rename("label")], axis=1).dropna()
        X_clean  = combined.drop(columns=["label"])
        y_clean  = combined["label"].astype(int)

        # Temporal split (no shuffle — time series data)
        split_idx = int(len(X_clean) * (1 - eval_fraction))
        X_tr, X_ev = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
        y_tr, y_ev = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]

        logger.info(
            f"[{self.symbol}] Training regime classifier  "
            f"train={len(X_tr):,}  eval={len(X_ev):,}"
        )

        self.model = XGBClassifier(
            **REGIME_PARAMS,
            n_jobs=-1,
            verbosity=0,
        )
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_ev, y_ev)],
            verbose=False,
        )

        # Report
        preds = self.model.predict(X_ev)
        logger.info(
            f"\n[{self.symbol}] Regime classifier eval:\n"
            + classification_report(y_ev, preds,
                                    target_names=list(REGIME_NAMES.values()),
                                    zero_division=0)
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime integer labels."""
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict soft regime probabilities.
        Returns array of shape (n_samples, N_REGIMES).
        """
        self._check_fitted()
        proba = self.model.predict_proba(X)
        # Ensure we always return N_REGIMES columns (some regimes may be missing)
        if proba.shape[1] < N_REGIMES:
            full = np.zeros((len(proba), N_REGIMES))
            for i, cls in enumerate(self.model.classes_):
                full[:, cls] = proba[:, i]
            return full
        return proba

    def predict_proba_row(self, x: np.ndarray) -> np.ndarray:
        """Single-row inference for the live RL environment step."""
        self._check_fitted()
        row_2d = x.reshape(1, -1)
        df = pd.DataFrame(row_2d, columns=self.model.feature_names_in_)
        return self.predict_proba(df)[0]

    # ── Feature importance ────────────────────────────────────────────────────
    def feature_importance(self, feature_names: list) -> pd.DataFrame:
        self._check_fitted()
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": feature_names[:len(imp)], "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self):
        with open(self._model_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Regime classifier saved → {self._model_path}")

    @classmethod
    def load(cls, symbol: str = "shared") -> "RegimeClassifier":
        path = MODEL_DIR / f"regime_{symbol}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}. Train first.")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Regime classifier loaded from {path}")
        return obj

    # ── Internal ──────────────────────────────────────────────────────────────
    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("Model not trained. Call .fit() first.")
