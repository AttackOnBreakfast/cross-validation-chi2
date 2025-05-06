# -----------------------------
# src/plot.py
# -----------------------------
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

    # Plot
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

    # Theory lines
    plt.plot(x_dense, chi2_A_theory_smooth, 'b:', label=r"Theory: $N - m$")
    plt.plot(x_dense, chi2_B_theory_smooth, 'r:', label=r"Theory: $N + m$")

    # Variance bands
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
    plt.title("Cross-validated $\chi^2$ with Error Bars and Variance Bands")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")

    plt.tight_layout()
    plt.show()