# -----------------------------
# src/plot.py
# -----------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from src.fitting import fit_polynomial
from src.truth_function import f_truth
from src.utils import generate_data

def plot_figure1_fit_and_chi2(
    x_sample, y_sample,
    degrees,
    chi2_A, chi2_B,
    chi2_A_std, chi2_B_std,
    chi2_A_theory, chi2_B_theory,
    chi2_A_var_theory, chi2_B_var_theory,
    sigma
):
    truth_x = np.linspace(0, 1, 500)

    interp_A = interp1d(degrees, chi2_A, kind='cubic')
    interp_B = interp1d(degrees, chi2_B, kind='cubic')
    interp_A_theory = interp1d(degrees, chi2_A_theory, kind='linear')
    interp_B_theory = interp1d(degrees, chi2_B_theory, kind='linear')

    x_dense = np.linspace(1, degrees[-1], 500)
    chi2_A_smooth = interp_A(x_dense)
    chi2_B_smooth = interp_B(x_dense)
    chi2_A_theory_smooth = interp_A_theory(x_dense)
    chi2_B_theory_smooth = interp_B_theory(x_dense)

    plt.figure(figsize=(18, 9))
    plt.subplot(1, 2, 1)
    plt.errorbar(x_sample, y_sample, yerr=sigma, fmt='.', alpha=0.4, label="Sample A + noise")
    plt.plot(truth_x, f_truth(truth_x), 'k--', label="Truth function")
    fit_poly = fit_polynomial(x_sample, y_sample, degrees[-1])
    plt.plot(truth_x, fit_poly(truth_x), 'r-', label=f"Fit deg={degrees[-1]}")
    plt.title(f"Sample A Fit (deg={degrees[-1]})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.errorbar(degrees, chi2_A, yerr=chi2_A_std, fmt='o', label=r"$\chi^2(D_A, \theta_A)$", color='blue')
    plt.errorbar(degrees, chi2_B, yerr=chi2_B_std, fmt='s', label=r"$\chi^2(D_B, \theta_A)$", color='red')
    plt.plot(x_dense, chi2_A_smooth, 'b--')
    plt.plot(x_dense, chi2_B_smooth, 'r--')
    plt.plot(x_dense, chi2_A_theory_smooth, 'b:', label=r"Theory: $N - m$")
    plt.plot(x_dense, chi2_B_theory_smooth, 'r:', label=r"Theory: $N + m$")
    plt.fill_between(degrees, chi2_A + np.sqrt(chi2_A_var_theory), chi2_A - np.sqrt(chi2_A_var_theory), color='blue', alpha=0.2)
    plt.fill_between(degrees, chi2_B + np.sqrt(chi2_B_var_theory), chi2_B - np.sqrt(chi2_B_var_theory), color='red', alpha=0.2)
    plt.xlabel("Model Complexity (Polynomial Degree)")
    plt.ylabel(r"$\chi^2$ (log scale)")
    plt.title(r"Cross-validated $\chi^2$ with Error Bars and Variance Bands")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/chi2_cross_validation.png", dpi=300)
    plt.show()

def plot_figure2_variance_comparison(chi2_A_std, chi2_B_std, chi2_A_var_theory, chi2_B_var_theory):
    empirical_var_A = chi2_A_std ** 2
    empirical_var_B = chi2_B_std ** 2

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(chi2_A_var_theory, empirical_var_A, 'o-', color='blue', label=r"Training: $\chi^2_A$")
    ax.plot(chi2_B_var_theory, empirical_var_B, 's-', color='red', label=r"Test: $\chi^2_B$")

    # Identity line
    all_x = np.concatenate([chi2_A_var_theory, chi2_B_var_theory])
    all_y = np.concatenate([empirical_var_A, empirical_var_B])
    min_val = min(all_x.min(), all_y.min())
    max_val = max(all_x.max(), all_y.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label=r"$y = x$")

    # Labels and title
    ax.set_xlabel(r"Theoretical $\mathrm{Var}(\chi^2)$")
    ax.set_ylabel(r"Empirical $\mathrm{Var}(\chi^2)$")
    ax.set_title(r"Empirical vs Theoretical Variance of $\chi^2$")
    ax.grid(True)
    ax.legend()

    # Pearson r
    r_A, _ = pearsonr(chi2_A_var_theory, empirical_var_A)
    r_B, _ = pearsonr(chi2_B_var_theory, empirical_var_B)
    ax.text(0.05, 0.95, rf"$r_A = {r_A:.3f}$\n$r_B = {r_B:.3f}$",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

    # Smart zooming
    x_lo, x_hi = np.percentile(all_x, [5, 95])
    y_lo, y_hi = np.percentile(all_y, [5, 95])
    ax.set_xlim(x_lo - 0.1 * (x_hi - x_lo), x_hi + 0.1 * (x_hi - x_lo))
    ax.set_ylim(y_lo - 0.1 * (y_hi - y_lo), y_hi + 0.1 * (y_hi - y_lo))

    plt.tight_layout()
    plt.savefig("figures/chi2_var_vs_theory.png", dpi=300)
    plt.show()

def plot_figure3_bma_prediction(x_fit, y_fit, posterior, sigma):
    degrees = np.arange(1, len(posterior) + 1)
    x_test = np.linspace(0, 1, 200)
    y_bma_mean = np.zeros_like(x_test)
    y_bma_var = np.zeros_like(x_test)

    for m, weight in zip(degrees, posterior):
        model = fit_polynomial(x_fit, y_fit, m)
        pred = model(x_test)
        y_bma_mean += weight * pred
        y_bma_var += weight * pred**2

    y_bma_std = np.sqrt(y_bma_var - y_bma_mean**2)

    x_test_eval, y_test_eval = generate_data(len(x_fit), sigma, seed=777)
    y_bma_eval = np.zeros_like(x_test_eval)
    for m, weight in zip(degrees, posterior):
        model = fit_polynomial(x_fit, y_fit, m)
        y_bma_eval += weight * model(x_test_eval)
    chi2_bma_test = np.sum(((y_test_eval - y_bma_eval) / sigma) ** 2)

    truth_x = np.linspace(0, 1, 500)
    plt.figure(figsize=(10, 6))
    plt.fill_between(x_test, y_bma_mean - y_bma_std, y_bma_mean + y_bma_std,
                     color='gray', alpha=0.3, label='BMA ± std')
    plt.plot(x_test, y_bma_mean, label='BMA Mean', color='black', linewidth=2)
    plt.errorbar(x_fit, y_fit, yerr=sigma, fmt='.', alpha=0.4, label='Data')
    plt.plot(truth_x, f_truth(truth_x), 'k--', label="Truth")
    plt.title("BMA Fit and $\chi^2$ Test")
    plt.text(0.05, 0.95, rf"$\chi^2(D_B; f_{{\mathrm{{BMA}}}}) = {chi2_bma_test:.1f}$",
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))
    plt.xlabel("x")
    plt.ylabel("Predicted K")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/bma_prediction.png", dpi=300)
    plt.show()