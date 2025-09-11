# -----------------------------
# main_nn.py
# -----------------------------

import os
import numpy as np
from src.data import generate_dataset_pair
from src.nn_model import train_mlp_crossval
from src.plot import plot_nn_fit_and_chi2

# === Settings ===
n_points = 300
sigma = 0.3
seed = 42
hidden_sizes = [512, 512, 512]
lr = 1e-4                   
steps = 10000              

# === Generate datasets A and B ===
(x_A, y_A), (x_B, y_B) = generate_dataset_pair(n_points, sigma, seed)

# === Train NN on A and track χ² on A and B ===
params, chi2_A, chi2_B, x_dense, y_dense = train_mlp_crossval(
    x_A, y_A, x_B, y_B, sigma,
    seed=seed,
    hidden_sizes=hidden_sizes,
    lr=lr,
    steps=steps,
    use_clean_targets=True  # ← Train on clean targets (no noise)
)

# === Plot ===
plot_nn_fit_and_chi2(
    x_sample=x_A,
    y_sample=y_A,
    sigma=sigma,
    x_dense=x_dense,
    y_dense=y_dense,
    chi2_A=chi2_A,
    chi2_B=chi2_B,
    n_points=n_points,
    steps=steps,
    max_m=20  # ← Or tune to taste
)

# === Print final χ² values ===
print(f"Final χ² on D_A (train): {chi2_A[-1]:.2f}")
print(f"Final χ² on D_B (valid): {chi2_B[-1]:.2f}")
