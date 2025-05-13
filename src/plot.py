# -----------------------------
# src/plot.py
# -----------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from src.fitting import fit_polynomial
from src.truth_function import f_truth

def plot_results(
    sample,
    max_degree,
    chi2_A,
    chi2_B,
    chi2_A_std,
    chi2_B_std,
    chi2_A_theory,
    chi2_B_theory,
    chi2_A_var_theory,
    chi2_B_var_theory,
):
    x_sample, y_sample = sample
    truth_x = np.linspace(0, 1, 500)
    degrees = np.arange(1, max_degree + 1)

    # Smooth interpolations
    interp_A = interp1d(degrees, chi2_A, kind='cubic')
    interp_B = interp1d(degrees, chi2_B, kind='cubic')
    interp_A_theory = interp1d(degrees, chi2_A_theory, kind='linear')
    interp_B_theory = interp1d(degrees, chi2_B_theory, kind='linear')

    x_dense = np.linspace(1, max_degree, 500)
    chi2_A_smooth = interp_A(x_dense)
    chi2_B_smooth = interp_B(x_dense)
    chi2_A_theory_smooth = interp_A_theory(x_dense)
    chi2_B_theory_smooth = interp_B_theory(x_dense)

    # Plot 1: Fit + chi2 vs complexity
    plt.figure(figsize=(18, 9))

    # Left: Data and fit
    plt.subplot(1, 2, 1)
    plt.errorbar(x_sample, y_sample, yerr=0.3, fmt='.', alpha=0.4, label="Sample A + noise")
    plt.plot(truth_x, f_truth(truth_x), 'k--', label="Truth function")
    fit_poly = fit_polynomial(x_sample, y_sample, max_degree)
    plt.plot(truth_x, fit_poly(truth_x), 'r-', label=f"Fit deg={max_degree}")
    plt.title(f"Sample A Fit (deg={max_degree})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # Right: Chi^2 vs model complexity with error bars
    plt.subplot(1, 2, 2)
    plt.errorbar(degrees, chi2_A, yerr=chi2_A_std, fmt='o', label=r"$\chi^2(D_A, \theta_A)$", color='blue')
    plt.errorbar(degrees, chi2_B, yerr=chi2_B_std, fmt='s', label=r"$\chi^2(D_B, \theta_A)$", color='red')

    plt.plot(x_dense, chi2_A_smooth, 'b--')
    plt.plot(x_dense, chi2_B_smooth, 'r--')

    plt.plot(x_dense, chi2_A_theory_smooth, 'b:', label=r"Theory: $N - m$")
    plt.plot(x_dense, chi2_B_theory_smooth, 'r:', label=r"Theory: $N + m$")

    plt.fill_between(
        degrees,
        chi2_A + np.sqrt(chi2_A_var_theory),
        chi2_A - np.sqrt(chi2_A_var_theory),
        color='blue',
        alpha=0.2,
        label=r"Predicted $\pm\sqrt{{\rm Var}}(\chi^2_A)$"
    )
    plt.fill_between(
        degrees,
        chi2_B + np.sqrt(chi2_B_var_theory),
        chi2_B - np.sqrt(chi2_B_var_theory),
        color='red',
        alpha=0.2,
        label=r"Predicted $\pm\sqrt{{\rm Var}}(\chi^2_B)$"
    )

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

    # Plot 2: Zoomed-in region with Pearson r annotation
    empirical_var_A = chi2_A_std ** 2
    empirical_var_B = chi2_B_std ** 2

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(chi2_A_var_theory, empirical_var_A, 'o-', color='blue', label=r"Training: $\chi^2_A$")
    ax.plot(chi2_B_var_theory, empirical_var_B, 's-', color='red', label=r"Test: $\chi^2_B$")
    ax.plot([0, 1.05 * max(chi2_B_var_theory)], [0, 1.05 * max(chi2_B_var_theory)], 'k--', label=r"$y = x$")

    ax.set_xlabel(r"Theoretical $\mathrm{Var}(\chi^2)$")
    ax.set_ylabel(r"Empirical Variance of $\chi^2$")
    ax.set_title(r"Zoomed $\chi^2$ Variance vs Theoretical Prediction")
    ax.grid(True)
    ax.legend()

    # Smart zoom: keep points with small relative deviation
    threshold = 0.15
    mask_A = np.abs(empirical_var_A - chi2_A_var_theory) / chi2_A_var_theory < threshold
    mask_B = np.abs(empirical_var_B - chi2_B_var_theory) / chi2_B_var_theory < threshold
    combined_mask = mask_A | mask_B

    if np.any(combined_mask):
        x_vals = np.concatenate([chi2_A_var_theory[combined_mask], chi2_B_var_theory[combined_mask]])
        y_vals = np.concatenate([empirical_var_A[combined_mask], empirical_var_B[combined_mask]])
        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        margin_x = 0.1 * (x_max - x_min)
        margin_y = 0.1 * (y_max - y_min)
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)

    # Pearson r annotation
    from scipy.stats import pearsonr
    r_A, _ = pearsonr(chi2_A_var_theory, empirical_var_A)
    r_B, _ = pearsonr(chi2_B_var_theory, empirical_var_B)
    ax.text(0.05, 0.95,
            rf"$r_A = {r_A:.3f}$" + "\n" + rf"$r_B = {r_B:.3f}$",
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=12,
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig("figures/chi2_var_vs_theory.png", dpi=300)
    plt.show()