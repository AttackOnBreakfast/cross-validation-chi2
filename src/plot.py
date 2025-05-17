# -----------------------------
# src/plot.py
# -----------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from src.fitting import fit_polynomial, fit_polynomial_map
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

    interp_A = interp1d(degrees, chi2_A, kind='cubic')
    interp_B = interp1d(degrees, chi2_B, kind='cubic')
    interp_A_theory = interp1d(degrees, chi2_A_theory, kind='linear')
    interp_B_theory = interp1d(degrees, chi2_B_theory, kind='linear')

    x_dense = np.linspace(1, max_degree, 500)
    chi2_A_smooth = interp_A(x_dense)
    chi2_B_smooth = interp_B(x_dense)
    chi2_A_theory_smooth = interp_A_theory(x_dense)
    chi2_B_theory_smooth = interp_B_theory(x_dense)

    plt.figure(figsize=(18, 9))
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

    plt.subplot(1, 2, 2)
    plt.errorbar(degrees, chi2_A, yerr=chi2_A_std, fmt='o', label=r"$\chi^2(D_A, \theta_A)$", color='blue')
    plt.errorbar(degrees, chi2_B, yerr=chi2_B_std, fmt='s', label=r"$\chi^2(D_B, \theta_A)$", color='red')
    plt.plot(x_dense, chi2_A_smooth, 'b--')
    plt.plot(x_dense, chi2_B_smooth, 'r--')
    plt.plot(x_dense, chi2_A_theory_smooth, 'b:', label=r"Theory: $N - m$")
    plt.plot(x_dense, chi2_B_theory_smooth, 'r:', label=r"Theory: $N + m$")
    plt.fill_between(degrees, chi2_A + np.sqrt(chi2_A_var_theory), chi2_A - np.sqrt(chi2_A_var_theory), color='blue', alpha=0.2, label=r"Predicted $\pm\sqrt{{\rm Var}}(\chi^2_A)$")
    plt.fill_between(degrees, chi2_B + np.sqrt(chi2_B_var_theory), chi2_B - np.sqrt(chi2_B_var_theory), color='red', alpha=0.2, label=r"Predicted $\pm\sqrt{{\rm Var}}(\chi^2_B)$")
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

    r_A, _ = pearsonr(chi2_A_var_theory, empirical_var_A)
    r_B, _ = pearsonr(chi2_B_var_theory, empirical_var_B)
    ax.text(0.05, 0.95, rf"$r_A = {r_A:.3f}$\n$r_B = {r_B:.3f}$", transform=ax.transAxes, verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

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

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/chi2_var_vs_theory.png", dpi=300)
    plt.show()

def plot_prior_posterior(degrees, prior, posterior):
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, prior, 'k--', label="Prior", linewidth=2)
    plt.plot(degrees, posterior, 'b-', label="Posterior", linewidth=2)
    plt.xlabel("Polynomial Degree (Model Complexity)")
    plt.ylabel("Probability")
    plt.title("Prior vs Posterior over Model Degrees")
    plt.legend()
    plt.grid(True)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/prior_vs_posterior.png", dpi=300)
    plt.show()

def plot_mle_map_panel(x, y, deg_mle, deg_map, lam_map, degrees,
                        chi2_A, chi2_B, chi2_B_map, chi2_A_map,
                        chi2_A_std, chi2_B_std, chi2_B_map_std,
                        chi2_A_theory, chi2_B_theory,
                        chi2_A_var_theory, chi2_B_var_theory):
    truth_x = np.linspace(0, 1, 500)
    poly_mle = fit_polynomial(x, y, deg_mle)
    poly_map = fit_polynomial_map(x, y, deg_map, lam=lam_map)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.errorbar(x, y, yerr=0.3, fmt='.', alpha=0.3, label="Data")
    ax1.plot(truth_x, f_truth(truth_x), 'k--', label="Truth")
    ax1.plot(truth_x, poly_mle(truth_x), color='purple', linestyle='--', linewidth=2, label=f"MLE deg={deg_mle}")
    ax1.plot(truth_x, poly_map(truth_x), color='blue', linewidth=2.5, linestyle='-', label=f"MAP deg={deg_map}")
    ax1.set_title("MLE vs MAP Fits")
    ax1.set_ylabel("y")
    ax1.grid(True)
    ax1.legend()

    ax2.errorbar(degrees, chi2_A, yerr=chi2_A_std, fmt='o', label=r"$\chi^2(D_A, \theta_A)$", color='blue')
    ax2.errorbar(degrees, chi2_B, yerr=chi2_B_std, fmt='s', label=r"$\chi^2(D_B, \theta_A)$", color='red')
    ax2.plot(degrees, chi2_B_map, 'k--', label=r"$\chi^2(D_B, \theta^{(m)}_{\mathrm{MAP}})$")
    ax2.plot(degrees, chi2_A_map, 'k-.', label=r"$\chi^2(D_A, \theta^{(m)}_{\mathrm{MAP}})$")
    ax2.fill_between(degrees, chi2_B_map - chi2_B_map_std, chi2_B_map + chi2_B_map_std, color='black', alpha=0.15)
    ax2.plot(degrees, chi2_A_theory, 'b:', label=r"Theory: $N - m$")
    ax2.plot(degrees, chi2_B_theory, 'r:', label=r"Theory: $N + m$")
    ax2.fill_between(degrees, chi2_A + np.sqrt(chi2_A_var_theory), chi2_A - np.sqrt(chi2_A_var_theory), color='blue', alpha=0.2)
    ax2.fill_between(degrees, chi2_B + np.sqrt(chi2_B_var_theory), chi2_B - np.sqrt(chi2_B_var_theory), color='red', alpha=0.2)
    ax2.axvline(deg_mle, color='purple', linestyle='--', label=f"MLE deg={deg_mle}")
    ax2.axvline(deg_map, color='green', linestyle='--', label=f"MAP deg={deg_map}")
    ax2.set_xlabel("Polynomial Degree")
    ax2.set_ylabel(r"$\chi^2$")
    ax2.set_title(r"Cross-validated $\chi^2$ with MLE and MAP Fits")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--")
    ax2.legend()

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/combined_model_fit_and_chi2.png", dpi=300)
    plt.show()