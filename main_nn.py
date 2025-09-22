# -----------------------------
# main_nn.py (rewritten)
# -----------------------------

import numpy as np
from src.data import generate_fit_sample
from src.nn_model import kfold_mlp_chi2, train_mlp, predict_on_grid
from src.plot import plot_nn_chi2_double_descent

# --- Configurations ---
n_points = 300
sigma = 0.3
seed = 42
num_steps = 3000
n_splits = 5

# Widths to sweep for model complexity
width_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Generate a representative sample
x_sample, y_sample = generate_fit_sample(n_points=n_points, sigma=sigma, seed=seed)

# Store results
chi2_train_list = []
chi2_val_list = []

# --- Run width sweep ---
for width in width_list:
    print(f"Training MLP with width={width}...")

    # Run k-fold CV for this width
    result = kfold_mlp_chi2(
        x=x_sample,
        y=y_sample,
        hidden_layers=[width],
        num_steps=num_steps,
        sigma=sigma,
        n_splits=n_splits,
        base_seed=seed
    )

    chi2_train_list.append(result["chi2_train_mean"])
    chi2_val_list.append(result["chi2_val_mean"])

# --- Final model for plotting ---
# Train on full dataset with best width (e.g. last one)
final_width = width_list[-1]
params, model, x_mean, x_std, _, _ = train_mlp(
    x_train=x_sample.reshape(-1, 1),
    y_train=y_sample.reshape(-1, 1),
    x_val=x_sample.reshape(-1, 1),   # using train = val just for plotting
    y_val=y_sample.reshape(-1, 1),
    hidden_layers=[final_width],
    num_steps=num_steps,
    sigma=sigma,
    seed=seed
)

x_dense, y_pred_dense = predict_on_grid(params, model, x_mean, x_std)

# --- Plot ---
plot_nn_chi2_double_descent(
    x_sample=x_sample,
    y_sample=y_sample,
    y_pred_dense=y_pred_dense,
    chi2_train_list=chi2_train_list,
    chi2_val_list=chi2_val_list,
    complexities=width_list,
    complexity_label="Width"
)
