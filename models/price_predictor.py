"""
models/price_predictor.py  —  LSTM-based price direction predictor
──────────────────────────────────────────────────────────────────
Predicts probability of an up move over the next N bars.
The output probability is used as a feature in the RL state vector.

Architecture:
    Input  → LSTM layers → Dropout → Linear → Sigmoid
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODEL_DIR, LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS,
    LSTM_DROPOUT, LSTM_LR, LSTM_EPOCHS, LSTM_BATCH,
    LSTM_EVAL_BATCH, LSTM_MAX_SAMPLES, LSTM_EARLY_STOP_PAT,
    DIRECTION_HORIZON,
)


class LSTMDirectionModel:
    """
    PyTorch LSTM that predicts directional probability.

    Parameters
    ----------
    n_features : int   — number of input features per timestep
    symbol     : str   — used for model file naming
    """

    def __init__(self, n_features: int, symbol: str = "shared"):
        self.n_features = n_features
        self.symbol     = symbol
        self.net        = None
        self._model_path = MODEL_DIR / f"lstm_{symbol}.pt"
        self._scaler_path = MODEL_DIR / f"lstm_{symbol}_scaler.pkl"
        self._scaler    = None
        self._device    = None

    # ── Build network ─────────────────────────────────────────────────────────
    def _build(self):
        import torch
        import torch.nn as nn

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[{self.symbol}] LSTM device: {self._device}")

        class _Net(nn.Module):
            def __init__(self, n_feat, hidden, n_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_feat,
                    hidden_size=hidden,
                    num_layers=n_layers,
                    dropout=dropout if n_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.dropout = nn.Dropout(dropout)
                self.head    = nn.Sequential(
                    nn.Linear(hidden, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                last   = out[:, -1, :]          # last timestep
                return self.head(self.dropout(last)).squeeze(-1)

        self.net = _Net(
            self.n_features, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT
        ).to(self._device)

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_fraction: float = 0.15,
    ) -> "LSTMDirectionModel":
        """
        Train the LSTM.

        Parameters
        ----------
        X : (n_samples, n_features) normalised feature array
        y : (n_samples,) binary direction label (1=up, 0=down)
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        self._build()

        # ── Scale ─────────────────────────────────────────────────────────────
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # ── Build sequences ───────────────────────────────────────────────────
        X_seq, y_seq = self._make_sequences(X_scaled, y)
        n_total = len(X_seq)
        logger.info(
            f"[{self.symbol}] LSTM sequences: {X_seq.shape}  "
            f"pos_rate={y_seq.mean():.2%}"
        )

        # ── Subsample if too large (prevents RAM OOM on CPU) ──────────────────
        if n_total > LSTM_MAX_SAMPLES:
            rng  = np.random.default_rng(42)
            idx  = rng.choice(n_total, size=LSTM_MAX_SAMPLES, replace=False)
            idx  = np.sort(idx)           # keep temporal order
            X_seq, y_seq = X_seq[idx], y_seq[idx]
            logger.info(
                f"[{self.symbol}] Subsampled {n_total:,} → {LSTM_MAX_SAMPLES:,} sequences"
            )

        # ── Train / eval split ────────────────────────────────────────────────
        split = int(len(X_seq) * (1 - eval_fraction))
        X_tr, X_ev = X_seq[:split], X_seq[split:]
        y_tr, y_ev = y_seq[:split], y_seq[split:]

        # ── DataLoaders — keep eval on CPU, batch it to avoid OOM ────────────
        train_ds = TensorDataset(
            torch.FloatTensor(X_tr),
            torch.FloatTensor(y_tr),
        )
        eval_ds = TensorDataset(
            torch.FloatTensor(X_ev),
            torch.FloatTensor(y_ev),
        )
        train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH,      shuffle=False)
        eval_loader  = DataLoader(eval_ds,  batch_size=LSTM_EVAL_BATCH, shuffle=False)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=LSTM_LR, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )

        best_val_loss    = float("inf")
        best_weights     = None
        no_improve_count = 0

        logger.info(
            f"[{self.symbol}] Training: {len(X_tr):,} samples  "
            f"Eval: {len(X_ev):,} samples  "
            f"Device: {self._device}"
        )

        for epoch in range(LSTM_EPOCHS):
            # ── Train pass ────────────────────────────────────────────────────
            self.net.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                preds = self.net(xb)
                loss  = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # ── Eval pass — batched to avoid OOM ─────────────────────────────
            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in eval_loader:
                    xb, yb = xb.to(self._device), yb.to(self._device)
                    val_loss += criterion(self.net(xb), yb).item()
            val_loss /= len(eval_loader)

            scheduler.step(val_loss)

            # ── Track best ────────────────────────────────────────────────────
            if val_loss < best_val_loss - 1e-5:
                best_val_loss    = val_loss
                best_weights     = {k: v.clone() for k, v in self.net.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"[{self.symbol}] Epoch {epoch+1:3d}/{LSTM_EPOCHS}  "
                    f"train={train_loss:.4f}  val={val_loss:.4f}  "
                    f"best={best_val_loss:.4f}  no_improve={no_improve_count}"
                )

            # ── Early stopping ────────────────────────────────────────────────
            if no_improve_count >= LSTM_EARLY_STOP_PAT:
                logger.info(
                    f"[{self.symbol}] Early stopping at epoch {epoch+1}  "
                    f"(no improvement for {LSTM_EARLY_STOP_PAT} epochs)"
                )
                break

            # ── Divergence guard — stop if badly overfitting ──────────────────
            if epoch > 5 and val_loss > train_loss * 1.5:
                logger.warning(
                    f"[{self.symbol}] Val loss diverging "
                    f"(train={train_loss:.4f} val={val_loss:.4f}) — stopping early"
                )
                break

        # ── Restore best weights ──────────────────────────────────────────────
        if best_weights:
            self.net.load_state_dict(best_weights)
        logger.success(
            f"[{self.symbol}] LSTM done. Best val loss: {best_val_loss:.4f}"
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict up-move probability for every bar in X.
        Uses batched inference — safe for large arrays.

        Returns np.ndarray of shape (n_samples,) with values in [0, 1].
        First LSTM_SEQ_LEN values are padded with 0.5 (neutral).
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        X_seq, _ = self._make_sequences(X_scaled, np.zeros(len(X_scaled)))

        ds     = TensorDataset(torch.FloatTensor(X_seq))
        loader = DataLoader(ds, batch_size=LSTM_EVAL_BATCH, shuffle=False)

        self.net.eval()
        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self._device)
                preds.append(self.net(xb).cpu().numpy())

        p   = np.concatenate(preds)
        pad = np.full(len(X) - len(p), 0.5)
        return np.concatenate([pad, p])

    def predict_proba_single(self, x_window: np.ndarray) -> float:
        """
        Single-window inference for live env step.

        Parameters
        ----------
        x_window : (LSTM_SEQ_LEN, n_features) array
        """
        import torch
        self._check_fitted()
        scaled = self._scaler.transform(x_window)
        t = torch.FloatTensor(scaled[np.newaxis, :, :]).to(self._device)
        self.net.eval()
        with torch.no_grad():
            return float(self.net(t).item())

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self):
        import torch
        torch.save(self.net.state_dict(), self._model_path)
        with open(self._scaler_path, "wb") as f:
            pickle.dump(self._scaler, f)
        logger.info(f"LSTM saved → {self._model_path}")

    @classmethod
    def load(cls, n_features: int, symbol: str = "shared") -> "LSTMDirectionModel":
        import torch
        obj = cls(n_features=n_features, symbol=symbol)
        obj._build()
        obj.net.load_state_dict(
            torch.load(obj._model_path, map_location="cpu")
        )
        with open(obj._scaler_path, "rb") as f:
            obj._scaler = pickle.load(f)
        logger.info(f"LSTM loaded from {obj._model_path}")
        return obj

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _make_sequences(X: np.ndarray, y: np.ndarray):
        """Slide a window of length LSTM_SEQ_LEN over X."""
        n   = len(X)
        seq = LSTM_SEQ_LEN
        if n < seq:
            raise ValueError(f"Not enough data ({n}) for sequence length {seq}")
        xs = np.stack([X[i: i + seq] for i in range(n - seq)])
        ys = y[seq:]
        return xs.astype(np.float32), ys.astype(np.float32)

    def _check_fitted(self):
        if self.net is None or self._scaler is None:
            raise RuntimeError("Model not fitted. Call .fit() or .load() first.")
