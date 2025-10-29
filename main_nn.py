# -----------------------------
# src/nn_main.py
# -----------------------------
import os
import numpy as np
from tqdm import tqdm
from src.data import generate_dataset_pair
from src.truth_function import f_truth
from src.nn_model import (
    RFConfig,
    train_on_A_hybrid,
    kfold_cv_on_B_hybrid,
    chi2,
    param_count_hybrid,
)
from src.plot import plot_fit_and_double_descent


def main():
    seed_base = 42
    n_points = 300
    sigma = 0.3
    n_trials = 10

    cfg = RFConfig(activation="tanh")
    print(f"Device: {cfg.device.upper()}  (Hybrid linear + random-features ridgeless model)")

    # Sweep polynomial degrees and random widths
    m_list = [0, 1, 2, 4, 8, 16, 32]
    w_list = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    chi2_train_trials = []
    chi2_val_trials = []

    combo_list = [(m, w) for m in m_list for w in w_list]
    p_list = [param_count_hybrid(m, w) for (m, w) in combo_list]

    for t in range(n_trials):
        (x_A, y_A), (x_B, y_B) = generate_dataset_pair(n_points, sigma, seed_base + 101 * t)

        chi2_train_list, chi2_val_list = [], []
        for (m, w) in tqdm(combo_list, desc=f"Trial {t+1}/{n_trials}"):
            model, _ = train_on_A_hybrid(x_A, y_A, m, w, cfg, seed=seed_base + t)
            yhat_A = model.predict(x_A)
            chi2_A = chi2(y_A, yhat_A, sigma)
            chi2_train_list.append(chi2_A)

            cv_stats = kfold_cv_on_B_hybrid(model, x_B, y_B, sigma,
                                            n_splits=5, shuffle=True, random_state=seed_base)
            chi2_val_list.append(cv_stats["chi2_mean"])

        chi2_train_trials.append(chi2_train_list)
        chi2_val_trials.append(chi2_val_list)

    chi2_train_mean = np.mean(chi2_train_trials, axis=0)
    chi2_val_mean   = np.mean(chi2_val_trials, axis=0)

    # Find global best
    best_idx = int(np.argmin(chi2_val_mean))
    best_m, best_w = combo_list[best_idx]
    print(f"Best model: m={best_m}, w={best_w}, params={p_list[best_idx]}")

    (x_A, y_A), _ = generate_dataset_pair(n_points, sigma, seed_base)
    best_model, _ = train_on_A_hybrid(x_A, y_A, best_m, best_w, cfg, seed=seed_base)
    x_dense = np.linspace(0, 1, 1000)
    y_hat_dense = best_model.predict(x_dense)

    os.makedirs("figures", exist_ok=True)
    plot_fit_and_double_descent(
        x_A, y_A, x_dense, y_hat_dense,
        p_list, chi2_train_mean, chi2_val_mean,
        sigma, savepath="figures/nn_double_descent.png"
    )
    print("\n[Done] Saved figures/nn_double_descent.png")


if __name__ == "__main__":
    main()