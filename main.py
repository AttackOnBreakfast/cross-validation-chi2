# -----------------------------
# main.py
# -----------------------------
import numpy as np
import warnings
from src.utils import generate_data
from src.fitting import fit_polynomial, compute_chi2
from src.theory import chi2_theory_A, chi2_theory_B, chi2_variance_A, chi2_variance_B
from src.plot import plot_results

# Suppress RankWarnings for high-degree polynomial fits
warnings.simplefilter('ignore', np.RankWarning)

# Parameters
n_points = 300
sigma = 0.3
max_params = 30
n_trials = 100

# Chi2 accumulators
chi2_A_accum = np.zeros(max_params)
chi2_B_accum = np.zeros(max_params)
chi2_A_all = np.zeros((n_trials, max_params))
chi2_B_all = np.zeros((n_trials, max_params))

# Sample for fit plot
x_fit_sample, y_fit_sample = generate_data(n_points, sigma, seed=42)

# Cross-validation loop
for seed in range(n_trials):
    x_A, y_A = generate_data(n_points, sigma, seed=seed)
    x_B, y_B = generate_data(n_points, sigma, seed=seed + 1000)

    for m in range(1, max_params + 1):
        p = fit_polynomial(x_A, y_A, m)
        chi2_a = compute_chi2(y_A, p(x_A), sigma)
        chi2_b = compute_chi2(y_B, p(x_B), sigma)

        chi2_A_accum[m - 1] += chi2_a
        chi2_B_accum[m - 1] += chi2_b

        chi2_A_all[seed, m - 1] = chi2_a
        chi2_B_all[seed, m - 1] = chi2_b

# Averages and theoretical predictions
degrees = np.arange(1, max_params + 1)
chi2_A_avg = chi2_A_accum / n_trials
chi2_B_avg = chi2_B_accum / n_trials
chi2_A_std = np.std(chi2_A_all, axis=0)
chi2_B_std = np.std(chi2_B_all, axis=0)
chi2_A_theory = chi2_theory_A(n_points, degrees)
chi2_B_theory = chi2_theory_B(n_points, degrees)
chi2_A_var_theory = chi2_variance_A(n_points, degrees)
chi2_B_var_theory = chi2_variance_B(n_points, degrees)

# Plot all results with error bars and theoretical bands
plot_results(
    sample=(x_fit_sample, y_fit_sample),
    max_degree=max_params,
    chi2_A=chi2_A_avg,
    chi2_B=chi2_B_avg,
    chi2_A_std=chi2_A_std,
    chi2_B_std=chi2_B_std,
    chi2_A_theory=chi2_A_theory,
    chi2_B_theory=chi2_B_theory,
    chi2_A_var_theory=chi2_A_var_theory,
    chi2_B_var_theory=chi2_B_var_theory
)
