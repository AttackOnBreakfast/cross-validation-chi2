# -----------------------------
# main.py
# -----------------------------
import numpy as np
import warnings
import matplotlib.pyplot as plt
from src.data import generate_dataset_pair, rescale
from src.fitting import fit_chebyshev, eval_chebyshev
from src.theory import theoretical_chi2
from src.plot import plot_fit_and_chi2
from src.utils import rescale, smooth_curve

warnings.simplefilter('ignore', np.RankWarning)

# Parameters
n_points = 300
sigma = 0.3
max_params = 20
n_trials = 100

# Chi2 accumulators
chi2_A_accum = np.zeros(max_params)
chi2_B_accum = np.zeros(max_params)

# Sample for plotting fit
(x_A_fit, y_A_fit), _ = generate_dataset_pair(n_points, sigma, seed=42)
x_plot = np.linspace(0, 1, 500)

# Loop over seeds
for seed in range(n_trials):
    (x_A, y_A), (x_B, y_B) = generate_dataset_pair(n_points, sigma, seed=seed)
    x_A_rescaled = rescale(x_A)
    x_B_rescaled = rescale(x_B)

    for m in range(1, max_params + 1):
        coefs = fit_chebyshev(x_A_rescaled, y_A, m)
        y_pred_A = eval_chebyshev(x_A_rescaled, coefs)
        y_pred_B = eval_chebyshev(x_B_rescaled, coefs)

        chi2_A = np.sum(((y_A - y_pred_A) / sigma) ** 2)
        chi2_B = np.sum(((y_B - y_pred_B) / sigma) ** 2)

        chi2_A_accum[m - 1] += chi2_A
        chi2_B_accum[m - 1] += chi2_B

# Averages
chi2_A_avg = chi2_A_accum / n_trials
chi2_B_avg = chi2_B_accum / n_trials

# Theory
degrees = np.arange(1, max_params + 1)
chi2_A_theory, chi2_B_theory = theoretical_chi2(n_points, degrees)

# Plot
plot_fit_and_chi2(x_A_fit, y_A_fit, x_plot, max_params,
                  chi2_A_avg, chi2_B_avg, chi2_A_theory, chi2_B_theory, degrees)