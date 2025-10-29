# -----------------------------
# src/nn_model.py
# -----------------------------
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Dict, List


# ======= CONFIGS =======

@dataclass
class RFConfig:
    activation: str = "tanh"
    dtype: torch.dtype = torch.float32
    device: str = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    weight_scale: float = 3.0
    bias_scale: float = 1.0
    pinv_rcond: float = 1e-12


# ======= HELPERS =======

def _act(name: str):
    if name.lower() == "tanh": return torch.tanh
    if name.lower() == "relu": return torch.relu
    raise ValueError(f"Unknown activation '{name}'")

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def chi2(y_true: np.ndarray, y_pred: np.ndarray, sigma: float) -> float:
    r = (y_true - y_pred) / sigma
    return float(np.sum(r * r))


# ======= HYBRID MODEL =======

class HybridFeaturesModel:
    """
    Hybrid Linear + Random-Features model:
        Φ(x) = [1, x, x², ..., x^m, tanh(Wx+b)]
    Train linear head by exact ridgeless least squares.
    """
    def __init__(self, m_degree: int, rf_width: int, cfg: RFConfig, seed: int = 0):
        self.m_degree = m_degree
        self.rf_width = rf_width
        self.cfg = cfg
        self.act = _act(cfg.activation)
        set_seed(seed)

        # random tanh features
        W = cfg.weight_scale * torch.randn(1, rf_width, dtype=cfg.dtype, device=cfg.device)
        b = cfg.bias_scale   * torch.randn(rf_width, dtype=cfg.dtype, device=cfg.device)
        self.W = W
        self.b = b

        self.theta = None
        self.c = None

    def _poly_features(self, x: torch.Tensor) -> torch.Tensor:
        powers = [x**k for k in range(1, self.m_degree + 1)]
        return torch.cat(powers, dim=1) if powers else torch.zeros((x.shape[0], 0), device=x.device)

    def _rf_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x @ self.W + self.b)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        poly = self._poly_features(x)
        rf = self._rf_features(x)
        return torch.cat([poly, rf], dim=1)

    def fit_ridgeless(self, x_np: np.ndarray, y_np: np.ndarray):
        x = torch.tensor(x_np[:, None], dtype=self.cfg.dtype, device=self.cfg.device)
        y = torch.tensor(y_np[:, None], dtype=self.cfg.dtype, device=self.cfg.device)

        Phi = self._features(x)
        ones = torch.ones((Phi.shape[0], 1), dtype=self.cfg.dtype, device=self.cfg.device)
        A = torch.cat([Phi, ones], dim=1)

        pinv = torch.linalg.pinv(A, rcond=self.cfg.pinv_rcond)
        coeff = pinv @ y
        self.theta = coeff[:-1, 0].detach()
        self.c = coeff[-1, 0].detach()
        return self

    @torch.no_grad()
    def predict(self, x_np: np.ndarray) -> np.ndarray:
        x = torch.tensor(x_np[:, None], dtype=self.cfg.dtype, device=self.cfg.device)
        Phi = self._features(x)
        y_hat = Phi @ self.theta + self.c
        return y_hat.cpu().numpy().ravel()


# ======= WRAPPERS =======

def param_count_hybrid(m_degree: int, rf_width: int) -> int:
    return m_degree + rf_width + 1  # bias


def train_on_A_hybrid(
    x_A: np.ndarray,
    y_A: np.ndarray,
    m_degree: int,
    rf_width: int,
    cfg: RFConfig,
    seed: int,
) -> Tuple[HybridFeaturesModel, float]:
    model = HybridFeaturesModel(m_degree, rf_width, cfg, seed=seed)
    model.fit_ridgeless(x_A, y_A)
    y_hat = model.predict(x_A)
    mse = float(np.mean((y_hat - y_A) ** 2))
    return model, mse


def kfold_cv_on_B_hybrid(
    model: HybridFeaturesModel,
    x_B: np.ndarray,
    y_B: np.ndarray,
    sigma: float,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 0,
) -> Dict[str, float]:
    n = len(x_B)
    idx = np.arange(n)
    rng = np.random.RandomState(random_state)
    if shuffle:
        rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)

    chi2_vals = []
    for fold in folds:
        x_val = x_B[fold]
        y_val = y_B[fold]
        y_pred = model.predict(x_val)
        chi2_vals.append(chi2(y_val, y_pred, sigma))
    return {"chi2_mean": float(np.mean(chi2_vals)), "chi2_per_fold": chi2_vals}