# -----------------------------
# main.py
# -----------------------------
import os
import numpy as np
import pandas as pd
import warnings
from src.utils import generate_data
from src.fitting import fit_polynomial, compute_chi2
from src.theory import chi2_theory_A, chi2_theory_B, chi2_variance_A, chi2_variance_B
from src.plot import plot_results, plot_prior_posterior
from src.prior import exponential_model_prior, posterior_over_models

# Suppress RankWarnings for high-degree polynomial fits
warnings.simplefilter('ignore', np.RankWarning)

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

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

# Summary statistics
degrees = np.arange(1, max_params + 1)
chi2_A_avg = chi2_A_accum / n_trials
chi2_B_avg = chi2_B_accum / n_trials
chi2_A_std = np.std(chi2_A_all, axis=0)
chi2_B_std = np.std(chi2_B_all, axis=0)
chi2_A_theory = chi2_theory_A(n_points, degrees)
chi2_B_theory = chi2_theory_B(n_points, degrees)
chi2_A_var_theory = chi2_variance_A(n_points, degrees)
chi2_B_var_theory = chi2_variance_B(n_points, degrees)

# Prior and posterior
prior = exponential_model_prior(max_params, lam=0.1)
posterior = posterior_over_models(chi2_B_avg, prior, sigma_squared=sigma**2)

# Plot everything
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
plot_prior_posterior(degrees, prior, posterior)

# Save CSV
posterior_df = pd.DataFrame({
    "Degree": degrees,
    "Prior": prior,
    "Posterior": posterior,
    "Chi2_B_avg": chi2_B_avg,
    "Chi2_B_std": chi2_B_std,
    "Chi2_B_var_theory": chi2_B_var_theory
})
posterior_df.to_csv("results/prior_posterior_analysis.csv", index=False)

# Save LaTeX table
with open("results/chi2_table.tex", "w") as f:
    f.write("\\begin{tabular}{c c c c c c}\n")
    f.write("\\toprule\n")
    f.write("Degree & Prior & Posterior & $\\chi^2_B$ & $\\sigma_B$ & $\\mathrm{Var}_B^{\\text{th}}$ \\\\\n")
    f.write("\\midrule\n")
    for i, m in enumerate(degrees):
        row = f"{m} & {prior[i]:.3f} & {posterior[i]:.3f} & {chi2_B_avg[i]:.1f} & {chi2_B_std[i]:.2f} & {chi2_B_var_theory[i]:.1f} \\\\\n"
        f.write(row)
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")