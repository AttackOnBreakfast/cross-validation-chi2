# -----------------------------
# main_nn.py
# Neural Network χ² Cross-Validation Sweep (Double Descent Demo)
# -----------------------------
import numpy as np
from tqdm import tqdm
from src.data import generate_fit_sample
from src.nn_model import kfold_mlp_chi2, train_mlp, predict_on_grid
from src.plot import plot_nn_chi2_double_descent


# ===============================================
# Configuration
# ===============================================
n_points = 300          # number of data points
sigma = 0.3             # noise level
seed = 42               # base random seed
num_steps = 8000        # training iterations per model
n_splits = 1            # keep raw interpolation peak (disable smoothing)
n_trials = 2            # few trials for averaging, keeps runtime reasonable

# Sweep over model capacity (hidden layer width)
width_list = [
    2, 4, 8, 16, 32, 64, 128, 256,
    512, 1024, 2048, 4096
]

print(f"\n=== Neural Network Double Descent Experiment ===")
print(f"Dataset: N={n_points}, σ={sigma:.2f}, steps={num_steps}, trials={n_trials}, CV={n_splits}-fold")
print(f"Sweeping {len(width_list)} widths: {width_list}\n")


# ===============================================
# Generate one dataset (shared across all widths)
# ===============================================
x_sample, y_sample = generate_fit_sample(
    n_points=n_points, sigma=sigma, seed=seed
)


# ===============================================
# Width sweep with trial averaging
# ===============================================
chi2_train_means, chi2_val_means = [], []

for width in tqdm(width_list, desc="Width sweep", ncols=100):
    chi2_train_tmp, chi2_val_tmp = [], []

    for trial in range(n_trials):
        print(f"  ▶ Width={width:<4} | Trial={trial+1}/{n_trials} ...", end=" ", flush=True)

        result = kfold_mlp_chi2(
            x=x_sample,
            y=y_sample,
            hidden_layers=[width],
            num_steps=num_steps,
            sigma=sigma,
            n_splits=n_splits,
            base_seed=seed + trial * 123,
        )

        print("done.")
        chi2_train_tmp.append(result["chi2_train_median"])
        chi2_val_tmp.append(result["chi2_val_median"])

    chi2_train_means.append(np.mean(chi2_train_tmp))
    chi2_val_means.append(np.mean(chi2_val_tmp))

print("\n✅ Completed all widths successfully.\n")


# ===============================================
# Final model (for visualization)
# ===============================================
final_width = width_list[-1]
print(f"Training final model (width={final_width}) for visualization...")

params, model, x_mean, x_std, _, _ = train_mlp(
    x_train=x_sample.reshape(-1, 1),
    y_train=y_sample.reshape(-1, 1),
    x_val=x_sample.reshape(-1, 1),
    y_val=y_sample.reshape(-1, 1),
    hidden_layers=[final_width],
    num_steps=num_steps,
    sigma=sigma,
    seed=seed,
)

x_dense, y_pred_dense = predict_on_grid(params, model, x_mean, x_std)


# ===============================================
# Plot results
# ===============================================
plot_nn_chi2_double_descent(
    x_sample=x_sample,
    y_sample=y_sample,
    y_pred_dense=y_pred_dense,
    chi2_train_list=chi2_train_means,
    chi2_val_list=chi2_val_means,
    complexities=width_list,
    complexity_label="Width",
    sigma=sigma,
)

print("\n📊 Plot saved successfully. Experiment complete.\n")