# -----------------------------
# src/nn_main.py
# -----------------------------
import os
import numpy as np
from tqdm import tqdm
import torch

from src.data import generate_dataset_pair
from src.truth_function import f_truth
from src.nn_model import (
    RFConfig,
    train_on_A_ridgeless,
    kfold_cv_on_B_with_fixed_model,
    chi2,
    param_count_rf,
)
from src.plot import plot_fit_and_double_descent


def main():
    seed_base = 42
    n_points = 300
    sigma = 0.3
    n_trials = 10

    cfg = RFConfig(activation="tanh")
    print(f"Device: {cfg.device.upper()}  (random-features NN, ridgeless head)")

    # Sweep the number of random features (this IS the parameter count, up to +1 for bias).
    width_list = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
    p_list = [param_count_rf(w) for w in width_list]

    chi2_train_trials = []
    chi2_val_trials = []

    max_width = max(width_list)
    for t in range(n_trials):
        (x_A, y_A), (x_B, y_B) = generate_dataset_pair(n_points, sigma, seed_base + 101 * t)

        # One fixed feature map per trial
        torch.manual_seed(seed_base + t)
        W_master = cfg.weight_scale * torch.randn(1, max_width, dtype=cfg.dtype, device=cfg.device)
        b_master = cfg.bias_scale   * torch.randn(max_width, dtype=cfg.dtype, device=cfg.device)

        chi2_train_list, chi2_val_list = [], []
        for w in tqdm(width_list, desc=f"Trial {t+1}/{n_trials}"):
            model, _ = train_on_A_ridgeless(x_A, y_A, w, cfg, seed=seed_base + t,
                                        master_W=W_master, master_b=b_master)
            yhat_A = model.predict(x_A)
            chi2_A = chi2(y_A, yhat_A, sigma)
            chi2_train_list.append(chi2_A)

            cv_stats = kfold_cv_on_B_with_fixed_model(model, x_B, y_B, sigma,
                                                    n_splits=5, shuffle=True, random_state=seed_base)
            chi2_val_list.append(cv_stats["chi2_mean"])

        chi2_train_trials.append(chi2_train_list)
        chi2_val_trials.append(chi2_val_list)

    chi2_train_mean = np.mean(chi2_train_trials, axis=0)
    chi2_val_mean   = np.mean(chi2_val_trials, axis=0)

    # Pick best width by CV mean for left-panel fit
    best_idx = int(np.argmin(chi2_val_mean))
    best_w = width_list[best_idx]

    # Train once more on A for the left panel
    (x_A, y_A), _ = generate_dataset_pair(n_points, sigma, seed_base)
    model_best, _ = train_on_A_ridgeless(x_A, y_A, width=best_w, cfg=cfg, seed=seed_base)
    x_dense = np.linspace(0, 1, 1000)
    y_hat_dense = model_best.predict(x_dense)

    os.makedirs("figures", exist_ok=True)
    plot_fit_and_double_descent(
        x_A=x_A,
        y_A=y_A,
        x_dense=x_dense,
        y_hat_dense=y_hat_dense,
        p_list=p_list,
        chi2_train_list=chi2_train_mean,
        chi2_cv_list=chi2_val_mean,
        sigma=sigma,
        savepath="figures/nn_double_descent.png",
    )

    print("\n[Done] Saved figures/nn_double_descent_random_features.png")
    print(f"Best width (min ⟨χ²_val⟩): {best_w} (p={p_list[best_idx]})")


if __name__ == "__main__":
    main()