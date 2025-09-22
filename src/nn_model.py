# -----------------------------
# src/nn_model.py (with KFold CV and chi² tracking)
# -----------------------------

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from sklearn.model_selection import KFold
from typing import List, Dict
import numpy as np


class MLP(nn.Module):
    layer_sizes: List[int]

    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes[:-1]:
            x = nn.tanh(nn.Dense(size)(x))
        x = nn.Dense(self.layer_sizes[-1])(x)
        return x


def mse_loss(params, model, x, y):
    preds = model.apply(params, x)
    return jnp.mean((preds - y) ** 2)


def chi2(y_true, y_pred, sigma):
    return jnp.sum((y_true - y_pred) ** 2) / sigma**2


def train_mlp(
    x_train, y_train,
    x_val, y_val,
    hidden_layers: List[int],
    num_steps: int,
    sigma: float,
    seed: int
):
    x_mean = x_train.mean()
    x_std = x_train.std()
    x_train_norm = (x_train - x_mean) / x_std
    x_val_norm = (x_val - x_mean) / x_std

    model = MLP(layer_sizes=hidden_layers + [1])
    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, jnp.ones((1, 1)))

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    chi2_train_history = []
    chi2_val_history = []

    @jax.jit
    def step(params, opt_state, x_batch, y_batch):
        loss, grads = jax.value_and_grad(mse_loss)(params, model, x_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    for step_idx in range(num_steps):
        params, opt_state = step(params, opt_state, x_train_norm, y_train)
        if step_idx % 10 == 0:
            y_pred_train = model.apply(params, x_train_norm)
            y_pred_val = model.apply(params, x_val_norm)
            chi2_train_history.append(float(chi2(y_train, y_pred_train, sigma)))
            chi2_val_history.append(float(chi2(y_val, y_pred_val, sigma)))

    return params, model, x_mean, x_std, chi2_train_history, chi2_val_history


def kfold_mlp_chi2(
    x: jnp.ndarray,
    y: jnp.ndarray,
    hidden_layers: List[int],
    num_steps: int = 3000,
    sigma: float = 0.1,
    n_splits: int = 5,
    base_seed: int = 0
) -> Dict[str, float]:
    """
    Runs k-fold CV for a JAX MLP, returns average and std of chi^2.
    """
    x = jnp.array(x).reshape(-1, 1)
    y = jnp.array(y).reshape(-1, 1)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=base_seed)

    chi2_train_list = []
    chi2_val_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x)):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        params, model, x_mean, x_std, chi2_train_hist, chi2_val_hist = train_mlp(
            x_train, y_train,
            x_val, y_val,
            hidden_layers=hidden_layers,
            num_steps=num_steps,
            sigma=sigma,
            seed=base_seed + fold_idx
        )

        chi2_train_list.append(chi2_train_hist[-1])
        chi2_val_list.append(chi2_val_hist[-1])

    return {
        "chi2_train_mean": np.mean(chi2_train_list),
        "chi2_val_mean": np.mean(chi2_val_list),
        "chi2_train_std": np.std(chi2_train_list),
        "chi2_val_std": np.std(chi2_val_list),
    }


def predict_on_grid(params, model, x_mean, x_std):
    x_grid = jnp.linspace(0, 1, 500).reshape(-1, 1)
    x_norm = (x_grid - x_mean) / x_std
    y_pred = model.apply(params, x_norm)
    return np.array(x_grid).flatten(), np.array(y_pred).flatten()