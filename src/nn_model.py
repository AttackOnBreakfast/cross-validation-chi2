# -----------------------------
# src/nn_model.py
# -----------------------------
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Dict, List


@dataclass
class RFConfig:
    activation: str = "tanh"  # or "relu"
    dtype: torch.dtype = torch.float32
    device: str = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    weight_scale: float = 3.0   # scale for random first-layer weights
    bias_scale: float = 1.0     # scale for random first-layer bias
    pinv_rcond: float = 1e-12   # numerical jitter for pseudo-inverse / lstsq


def _act(name: str):
    name = name.lower()
    if name == "tanh": return torch.tanh
    if name == "relu": return torch.relu
    raise ValueError(f"Unknown activation '{name}'")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


class RandomFeaturesNN:
    """
    One-hidden-layer random-features network:
       phi(x) = act( x * W + b ),  W,b are FROZEN random,
       y_hat  = phi(x) @ theta + c, where (theta,c) are trained by least squares.
    Trainable parameter count (our complexity p) ≈ width (+1 for bias).
    """
    def __init__(self, width: int, cfg: RFConfig, seed: int = 0):
        set_seed(seed)
        self.width = width
        self.cfg = cfg
        self.act = _act(cfg.activation)

        # Random frozen first layer
        W = cfg.weight_scale * torch.randn(1, width, dtype=cfg.dtype, device=cfg.device)
        b = cfg.bias_scale   * torch.randn(width, dtype=cfg.dtype, device=cfg.device)
        self.W = nn.Parameter(W, requires_grad=False)   # shape (in_dim=1, width)
        self.b = nn.Parameter(b, requires_grad=False)   # shape (width,)

        # Trainable linear head (closed-form solution); store after fit
        self.theta = None   # shape (width,)
        self.c = None       # scalar bias

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,1)
        return self.act(x @ self.W + self.b)  # (N, width)

    def fit_ridgeless(self, x_np: np.ndarray, y_np: np.ndarray):
        """
        Solve (Φ,1) @ [theta; c] = y  in least-squares (λ=0) via pseudo-inverse.
        """
        x = torch.tensor(x_np[:, None], dtype=self.cfg.dtype, device=self.cfg.device)
        y = torch.tensor(y_np[:, None], dtype=self.cfg.dtype, device=self.cfg.device)

        Phi = self._features(x)                         # (N, width)
        ones = torch.ones((Phi.shape[0], 1), dtype=self.cfg.dtype, device=self.cfg.device)
        A = torch.cat([Phi, ones], dim=1)               # (N, width+1)

        # Ridgeless solution with pseudo-inverse (stable on GPU with rcond)
        pinv = torch.linalg.pinv(A, rcond=self.cfg.pinv_rcond)
        coeff = pinv @ y                                 # (width+1, 1)

        self.theta = coeff[:-1, 0].detach()             # (width,)
        self.c = coeff[-1, 0].detach()                  # ()
        return self

    @torch.no_grad()
    def predict(self, x_np: np.ndarray) -> np.ndarray:
        x = torch.tensor(x_np[:, None], dtype=self.cfg.dtype, device=self.cfg.device)
        Phi = self._features(x)                          # (N, width)
        y_hat = Phi @ self.theta + self.c               # (N,)
        return y_hat.cpu().numpy().ravel()


def param_count_rf(width: int) -> int:
    """
    Trainable parameters are the linear head (theta: width, bias: 1).
    We use p = width + 1 as the complexity axis.
    """
    return width + 1


def chi2(y_true: np.ndarray, y_pred: np.ndarray, sigma: float) -> float:
    r = (y_true - y_pred) / sigma
    return float(np.sum(r * r))


def train_on_A_ridgeless(
    x_A: np.ndarray,
    y_A: np.ndarray,
    width: int,
    cfg: RFConfig,
    seed: int,
    master_W: torch.Tensor = None,
    master_b: torch.Tensor = None,
) -> Tuple[RandomFeaturesNN, float]:
    """
    Fit the ridgeless head on dataset A using either fresh or sliced frozen features.
    If master_W/b provided, slice the first 'width' columns to preserve nesting.
    """
    if master_W is not None and master_b is not None:
        Ww = master_W[:, :width].contiguous()
        bw = master_b[:width].contiguous()
        model = RandomFeaturesNN(width, cfg, seed=seed)
        model.W.data = Ww
        model.b.data = bw
    else:
        model = RandomFeaturesNN(width, cfg, seed=seed)

    model.fit_ridgeless(x_A, y_A)
    y_hat = model.predict(x_A)
    mse = float(np.mean((y_hat - y_A) ** 2))
    return model, mse


def kfold_cv_on_B_with_fixed_model(
    model: RandomFeaturesNN,
    x_B: np.ndarray,
    y_B: np.ndarray,
    sigma: float,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 0,
) -> Dict[str, float]:
    """
    Evaluate ONLY (no training on B). We keep this to mirror your earlier API.
    """
    n = len(x_B)
    idx = np.arange(n)
    rng = np.random.RandomState(random_state)
    if shuffle:
        rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)

    chi2_vals: List[float] = []
    for fold in folds:
        x_val = x_B[fold]
        y_val = y_B[fold]
        y_pred = model.predict(x_val)
        chi2_vals.append(chi2(y_val, y_pred, sigma))
    return {"chi2_mean": float(np.mean(chi2_vals)), "chi2_per_fold": chi2_vals}