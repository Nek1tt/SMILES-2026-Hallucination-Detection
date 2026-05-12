from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

PCA_COMPONENTS = 256
SEED = 42

def _fix_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class HallucinationProbe(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None
        self._scaler = StandardScaler()
        self._pca    = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> None:
        _fix_seeds()
        self._net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Call fit() first.")
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        _fix_seeds()
        X_s = self._scaler.fit_transform(X)
        n_comp = min(PCA_COMPONENTS, X_s.shape[0] - 1, X_s.shape[1])
        self._pca = PCA(n_components=n_comp, random_state=SEED)
        X_p = self._pca.fit_transform(X_s)
        self._build_network(X_p.shape[1])
        X_t = torch.from_numpy(X_p).float()
        y_t = torch.from_numpy(y.astype(np.float32))
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.train()
        for _ in range(200):
            optimizer.zero_grad()
            loss = criterion(self(X_t), y_t)
            loss.backward()
            optimizer.step()
        self.eval()
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 201)]))
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            s = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            if s > best_f1:
                best_f1, best_t = s, float(t)
        self._threshold = best_t
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_s = self._scaler.transform(X)
        X_p = self._pca.transform(X_s)
        X_t = torch.from_numpy(X_p).float()
        with torch.no_grad():
            prob_pos = torch.sigmoid(self(X_t)).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
