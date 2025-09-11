# -----------------------------
# src/nn_model.py
# -----------------------------
import jax.numpy as jnp
import jax.random as random
from jax import grad, jit, vmap
import optax
from src.truth_function import f_truth

# === MLP Initialization ===
def init_mlp_params(layer_sizes, key):
    """Xavier initialization of weights and zero biases for MLP."""
    keys = random.split(key, len(layer_sizes) - 1)
    params = []
    for k, (m, n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        W = random.normal(k, (n, m)) * jnp.sqrt(1.0 / m)
        b = jnp.zeros(n)
        params.append((W, b))
    return params

# === Forward Pass ===
def mlp_forward(params, x):
    """Applies MLP with tanh activations and a linear output layer."""
    for W, b in params[:-1]:
        x = jnp.tanh(jnp.dot(W, x) + b)
    W, b = params[-1]
    return jnp.dot(W, x) + b

@jit
def predict_batch(params, x_batch):
    return vmap(lambda x: mlp_forward(params, x))(x_batch)

# === Loss Function ===
@jit
def loss_fn(params, x, y, sigma):
    preds = predict_batch(params, x)
    return jnp.mean(((preds - y) / sigma) ** 2)  # Reduced χ²

# === JIT-compiled training step factory ===
def make_train_step(optimizer_update):
    @jit
    def train_step(params, opt_state, x, y, sigma):
        grads = grad(loss_fn)(params, x, y, sigma)
        updates, opt_state = optimizer_update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    return train_step

# === Training Loop with Cross-Validation Logging ===
def train_mlp_crossval(
    x_A, y_A, x_B, y_B, sigma,
    seed=0,
    hidden_sizes=[128, 128],
    lr=1e-3,
    steps=5000,
    use_clean_targets=False  # ← New flag to optionally use noise-free targets
):
    key = random.PRNGKey(seed)
    layer_sizes = [1] + hidden_sizes + [1]
    params = init_mlp_params(layer_sizes, key)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer.update)

    # Reshape inputs
    x_A = x_A.reshape(-1, 1)
    y_A = y_A.reshape(-1)
    x_B = x_B.reshape(-1, 1)
    y_B = y_B.reshape(-1)

    # === Replace targets with clean values if requested ===
    if use_clean_targets:
        y_A = f_truth(x_A)
        y_B = f_truth(x_B)

    # === Normalize ===
    x_mean = x_A.mean()
    x_std = x_A.std()
    y_mean = y_A.mean()
    y_std = y_A.std()

    x_A_norm = (x_A - x_mean) / x_std
    x_B_norm = (x_B - x_mean) / x_std
    y_A_norm = (y_A - y_mean) / y_std
    y_B_norm = (y_B - y_mean) / y_std

    chi2_A_hist = []
    chi2_B_hist = []

    for step in range(steps):
        params, opt_state = train_step(params, opt_state, x_A_norm, y_A_norm, sigma / y_std)
        chi2_A_hist.append(float(loss_fn(params, x_A_norm, y_A_norm, sigma / y_std)))
        chi2_B_hist.append(float(loss_fn(params, x_B_norm, y_B_norm, sigma / y_std)))

    # === Predict on dense grid (normalized input, unnormalized output) ===
    x_dense = jnp.linspace(0, 1, 500, dtype=jnp.float32).reshape(-1, 1)
    x_dense_norm = (x_dense - x_mean) / x_std
    y_pred_dense = predict_batch(params, x_dense_norm)
    y_pred_dense = y_pred_dense * y_std + y_mean

    return params, jnp.array(chi2_A_hist), jnp.array(chi2_B_hist), x_dense, y_pred_dense