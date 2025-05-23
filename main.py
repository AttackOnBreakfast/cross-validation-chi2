# -----------------------------
# main.py
# -----------------------------
import os
import numpy as np
import pandas as pd
import warnings
from src.utils import generate_data
from src.fitting import fit_polynomial, compute_chi2
from src.theory import (
    chi2_theory_A,
    chi2_theory_B,
    chi2_variance_A,
    chi2_variance_B
)
from src.prior import exponential_model_prior, posterior_over_models
from src.plot import (
    plot_figure1_fit_and_chi2,
    plot_figure2_variance_comparison,
    plot_figure3_bma_prediction
)

# Suppress polynomial fit warnings
warnings.simplefilter('ignore', np.RankWarning)

# Constants
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Parameters
n_points = 300
sigma = 0.3
max_params = 30
n_trials = 100
lam_map = 5.0

# Preallocate arrays
chi2_A_all = np.zeros((n_trials, max_params))
chi2_B_all = np.zeros((n_trials, max_params))

# Fixed sample for plotting
x_fit_sample, y_fit_sample = generate_data(n_points, sigma, seed=42)

# Cross-validation loop
for seed in range(n_trials):
    x_A, y_A = generate_data(n_points, sigma, seed=seed)
    x_B, y_B = generate_data(n_points, sigma, seed=seed + 1000)

    for m in range(1, max_params + 1):
        model = fit_polynomial(x_A, y_A, m)
        chi2_A_all[seed, m - 1] = compute_chi2(y_A, model(x_A), sigma)
        chi2_B_all[seed, m - 1] = compute_chi2(y_B, model(x_B), sigma)

# Statistics
degrees = np.arange(1, max_params + 1)
chi2_A_avg = np.mean(chi2_A_all, axis=0)
chi2_B_avg = np.mean(chi2_B_all, axis=0)
chi2_A_std = np.std(chi2_A_all, axis=0)
chi2_B_std = np.std(chi2_B_all, axis=0)
chi2_A_empirical_var = chi2_A_std**2
chi2_B_empirical_var = chi2_B_std**2

# Theory
chi2_A_theory = chi2_theory_A(n_points, degrees)
chi2_B_theory = chi2_theory_B(n_points, degrees)
chi2_A_var_theory = chi2_variance_A(n_points, degrees)
chi2_B_var_theory = chi2_variance_B(n_points, degrees)

# Priors and posteriors
prior = exponential_model_prior(max_params, lam=0.1)
posterior = posterior_over_models(chi2_B_avg, prior, sigma_squared=sigma**2)

# Plots
plot_figure1_fit_and_chi2(
    x_fit_sample, y_fit_sample,
    degrees,
    chi2_A_avg, chi2_B_avg,
    chi2_A_std, chi2_B_std,
    chi2_A_theory, chi2_B_theory,
    chi2_A_var_theory, chi2_B_var_theory,
    sigma
)

plot_figure2_variance_comparison(
    chi2_A_std, chi2_B_std,
    chi2_A_var_theory, chi2_B_var_theory
)

plot_figure3_bma_prediction(
    x_fit_sample, y_fit_sample, posterior, sigma
)

# Save summary
summary_df = pd.DataFrame({
    "Degree": degrees,
    "Chi2_A_avg": chi2_A_avg,
    "Chi2_A_std": chi2_A_std,
    "Chi2_A_var_theory": chi2_A_var_theory,
    "Chi2_B_avg": chi2_B_avg,
    "Chi2_B_std": chi2_B_std,
    "Chi2_B_var_theory": chi2_B_var_theory,
    "Prior": prior,
    "Posterior": posterior
})
summary_df.to_csv(f"{RESULTS_DIR}/chi2_summary.csv", index=False)

# Save LaTeX table
with open(f"{RESULTS_DIR}/chi2_table.tex", "w") as f:
    f.write("\\begin{tabular}{c c c c c c}\n")
    f.write("\\toprule\n")
    f.write("Degree & $\\chi^2_B$ & $\\sigma_B$ & $\\mathrm{Var}_B^{\\text{th}}$ & Prior & Posterior \\\\n")
    f.write("\\midrule\n")
    for i, m in enumerate(degrees):
        row = f"{m} & {chi2_B_avg[i]:.1f} & {chi2_B_std[i]:.2f} & {chi2_B_var_theory[i]:.1f} & {prior[i]:.3f} & {posterior[i]:.3f} \\\\n"
        f.write(row)
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")