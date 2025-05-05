# -----------------------------
# src/plot.py
# -----------------------------
import matplotlib.pyplot as plt
from src.truth import f_truth
from src.fitting import fit_chebyshev, eval_chebyshev
from src.data import rescale
import numpy as np

def plot_fit_and_chi2(x_fit, y_fit, x_plot, degree,
                      chi2_A, chi2_B, chi2_A_theory, chi2_B_theory, degrees):
    x_rescaled = rescale(x_fit)
    x_plot_rescaled = rescale(x_plot)
    coefs = fit_chebyshev(x_rescaled, y_fit, degree)
    y_fit_curve = eval_chebyshev(x_plot_rescaled, coefs)

    plt.figure(figsize=(12, 6))

    # Fit plot
    plt.subplot(1, 2, 1)
    plt.scatter(x_fit, y_fit, s=10, color='gray', alpha=0.5, label='Data A')
    plt.plot(x_plot, f_truth(x_plot), 'k--', lw=2, label='Truth')
    plt.plot(x_plot, y_fit_curve, 'r-', lw=2, label=f'Fit (deg={degree})')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Fit on Sample A")
    plt.legend()
    plt.grid(True)

    # Chi2 plot
    plt.subplot(1, 2, 2)
    plt.plot(degrees, chi2_A, 'b-', label=r'$\langle \chi^2(D_A, \theta_A) \rangle$')
    plt.plot(degrees, chi2_B, 'r-', label=r'$\langle \chi^2(D_B, \theta_A) \rangle$')
    plt.plot(degrees, chi2_A_theory, 'b--', label=r'Theory: $N - m$')
    plt.plot(degrees, chi2_B_theory, 'r--', label=r'Theory: $N + m$')
    plt.yscale('log')
    plt.xlabel("Model Complexity (deg)")
    plt.ylabel(r'$\chi^2$ (log scale)')
    plt.title("Cross-validated $\chi^2$ vs Model Complexity")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')

    plt.tight_layout()
    plt.show()