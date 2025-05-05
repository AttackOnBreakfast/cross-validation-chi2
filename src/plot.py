# -----------------------------
# src/plot.py
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from src.truth_function import f_truth


def plot_results(x_sample, y_sample, max_params, chi2_A_avg, chi2_B_avg,
                 chi2_A_theory, chi2_B_theory):

    degrees = np.arange(1, max_params + 1)
    truth_x = np.linspace(0, 1, 500)

    # Interpolation
    x_dense = np.linspace(1, max_params, 500)
    interp_A = interp1d(degrees, chi2_A_avg, kind='cubic')
    interp_B = interp1d(degrees, chi2_B_avg, kind='cubic')
    interp_A_th = interp1d(degrees, chi2_A_theory, kind='linear')
    interp_B_th = interp1d(degrees, chi2_B_theory, kind='linear')

    chi2_A_smooth = interp_A(x_dense)
    chi2_B_smooth = interp_B(x_dense)
    chi2_A_th_smooth = interp_A_th(x_dense)
    chi2_B_th_smooth = interp_B_th(x_dense)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Left: example fit
    plt.subplot(1, 2, 1)
    plt.scatter(x_sample[0], x_sample[1], s=10, color='gray', alpha=0.5, label='Data A (Sample)')
    plt.plot(truth_x, f_truth(truth_x), 'k--', lw=2, label='Truth Function')
    fit_poly = np.poly1d(np.polyfit(x_sample[0], x_sample[1], deg=max_params))
    plt.plot(truth_x, fit_poly(truth_x), 'r-', lw=2, label=f'Fit (deg={max_params})')
    plt.title(f"Fit to Sample A with Polynomial Degree {max_params}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # Right: chi^2 vs model complexity
    plt.subplot(1, 2, 2)
    plt.plot(x_dense, chi2_A_smooth, 'b-', label=r'$\langle \chi^2(D_A, \theta_A) \rangle$')
    plt.plot(x_dense, chi2_B_smooth, 'r-', label=r'$\langle \chi^2(D_B, \theta_A) \rangle$')
    plt.plot(x_dense, chi2_A_th_smooth, 'b--', label=r'Theory: $N - m$')
    plt.plot(x_dense, chi2_B_th_smooth, 'r--', label=r'Theory: $N + m$')
    plt.scatter(degrees, chi2_A_avg, color='blue', s=20)
    plt.scatter(degrees, chi2_B_avg, color='red', s=20)
    plt.xlabel("Model Complexity (Polynomial Degree)")
    plt.ylabel(r'$\chi^2$ Value (log scale)')
    plt.title("Average Chi-squared over Trials")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", linestyle='--')

    plt.tight_layout()
    plt.show()