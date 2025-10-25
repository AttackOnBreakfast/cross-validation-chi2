# -----------------------------
# src/nn_model.py
# JAX/Flax MLP with χ² tracking for double-descent experiments
# -----------------------------
from typing import List, Dict
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from sklearn.model_selection import KFold


# -----------------------------
# Model
# -----------------------------
class MLP(nn.Module):
    layer_sizes: List[int]  # e.g., [width, width, 1]

    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes[:-1]:
            x = nn.Dense(
                features=size,
                kernel_init=nn.initializers.kaiming_normal(),
                bias_init=nn.initializers.zeros,
            )(x)
            x = nn.tanh(x)
        # output layer
        x = nn.Dense(
            features=self.layer_sizes[-1],
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=nn.initializers.zeros,
        )(x)
        return x


# -----------------------------
# Loss / Metrics
# -----------------------------
def mse_loss(params, model, x, y):
    preds = model.apply(params, x)
    return jnp.mean((preds - y) ** 2)


def chi2(y_true, y_pred, sigma):
    """Raw χ² (no normalization by N)"""
    return jnp.sum((y_true - y_pred) ** 2) / (sigma ** 2)


# -----------------------------
# Training (single split)
# -----------------------------
def train_mlp(
    x_train,
    y_train,
    x_val,
    y_val,
    hidden_layers: List[int],
    num_steps: int,
    sigma: float,
    seed: int,
):
    """
    Train a small MLP; track χ² on train/val every 10 steps.
    Architecture and optimizer tuned to expose double-descent behavior.
    """
    # Convert to float32 arrays
    x_train = jnp.asarray(x_train, dtype=jnp.float32)
    y_train = jnp.asarray(y_train, dtype=jnp.float32)
    x_val   = jnp.asarray(x_val,   dtype=jnp.float32)
    y_val   = jnp.asarray(y_val,   dtype=jnp.float32)

    # Normalize inputs (not outputs)
    x_mean = x_train.mean()
    x_std  = x_train.std()
    x_std  = jnp.where(x_std == 0, 1.0, x_std)
    x_train_norm = (x_train - x_mean) / x_std
    x_val_norm   = (x_val   - x_mean) / x_std

    # -------------------------------------------------------
    # Model setup: 2 hidden layers of same width (captures overparam regime)
    # -------------------------------------------------------
    width = int(hidden_layers[0]) if len(hidden_layers) > 0 else 1
    model = MLP(layer_sizes=[width, width, 1])

    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, jnp.ones((1, 1), dtype=jnp.float32))

    # -------------------------------------------------------
    # Optimizer: cosine LR, small L2 decay (disabled at large widths)
    # -------------------------------------------------------
    base_lr = 5e-3 if width < 512 else 2e-3
    schedule = optax.cosine_decay_schedule(init_value=base_lr, decay_steps=num_steps)
    weight_decay = 1e-5 if width < 512 else 0.0

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.add_decayed_weights(weight_decay),
        optax.adam(schedule),
    )
    opt_state = optimizer.init(params)

    # -------------------------------------------------------
    # Training loop
    # -------------------------------------------------------
    chi2_train_history, chi2_val_history = [], []

    @jax.jit
    def step(params, opt_state, x_batch, y_batch):
        loss, grads = jax.value_and_grad(mse_loss)(params, model, x_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for step_idx in range(num_steps):
        params, opt_state, loss_val = step(params, opt_state, x_train_norm, y_train)

        # Log χ² every 10 steps
        if step_idx % 10 == 0 or step_idx == num_steps - 1:
            y_pred_train = model.apply(params, x_train_norm)
            y_pred_val   = model.apply(params, x_val_norm)
            chi2_train_history.append(float(chi2(y_train, y_pred_train, sigma)))
            chi2_val_history.append(float(chi2(y_val, y_pred_val, sigma)))

        # Divergence guard
        if not np.isfinite(loss_val) or float(loss_val) > 1e6:
            print(f"  ⚠️ Diverged at step {step_idx} (loss={float(loss_val):.2e}), stopping early.")
            break

    return params, model, x_mean, x_std, chi2_train_history, chi2_val_history


# -----------------------------
# K-Fold CV wrapper
# -----------------------------
def kfold_mlp_chi2(
    x: jnp.ndarray,
    y: jnp.ndarray,
    hidden_layers: List[int],
    num_steps: int = 8000,
    sigma: float = 0.3,
    n_splits: int = 1,
    base_seed: int = 0,
) -> Dict[str, float]:
    """
    Run (optionally single) cross-validation and return χ² stats.
    If n_splits=1, we manually create a single 80/20 split.
    """
    x = jnp.array(x).reshape(-1, 1).astype(jnp.float32)
    y = jnp.array(y).reshape(-1, 1).astype(jnp.float32)
    N = len(x)

    chi2_train_last, chi2_val_last = [], []
    chi2_train_hist_all, chi2_val_hist_all = [], []

    if n_splits == 1:
        # Manual single split (80% train / 20% val)
        rng = np.random.default_rng(base_seed)
        indices = np.arange(N)
        rng.shuffle(indices)
        split = int(0.8 * N)
        train_idx, val_idx = indices[:split], indices[split:]
        splits = [(train_idx, val_idx)]
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=base_seed)
        splits = list(kf.split(x))

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"    Fold {fold_idx + 1}/{len(splits)} … ", end="", flush=True)
        params, model, x_mean, x_std, chi2_train_hist, chi2_val_hist = train_mlp(
            x_train, y_train,
            x_val, y_val,
            hidden_layers=hidden_layers,
            num_steps=num_steps,
            sigma=sigma,
            seed=base_seed + fold_idx,
        )
        chi2_train_last.append(chi2_train_hist[-1])
        chi2_val_last.append(chi2_val_hist[-1])
        chi2_train_hist_all.append(chi2_train_hist)
        chi2_val_hist_all.append(chi2_val_hist)
        print("done.")

    return {
        "chi2_train_mean":   float(np.mean(chi2_train_last)),
        "chi2_val_mean":     float(np.mean(chi2_val_last)),
        "chi2_train_median": float(np.median(chi2_train_last)),
        "chi2_val_median":   float(np.median(chi2_val_last)),
        "chi2_train_std":    float(np.std(chi2_train_last)),
        "chi2_val_std":      float(np.std(chi2_val_last)),
        "train_histories":   chi2_train_hist_all,
        "val_histories":     chi2_val_hist_all,
    }


# -----------------------------
# Prediction grid (for plotting)
# -----------------------------
def predict_on_grid(params, model, x_mean, x_std):
    x_grid = jnp.linspace(0, 1, 500, dtype=jnp.float32).reshape(-1, 1)
    x_norm = (x_grid - jnp.asarray(x_mean, jnp.float32)) / jnp.asarray(x_std, jnp.float32)
    y_pred = model.apply(params, x_norm)
    return np.array(x_grid).flatten(), np.array(y_pred).flatten()