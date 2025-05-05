import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from src.truth_function import f_truth
from src.data import generate_data
from src.theory import theoretical_chi2
from src.utils import smooth_curve

# Parameters
n_points = 300
sigma = 0.3
max_params = 30
n_trials = 100

# Initialize chi-squared accumulators
chi2_A_accum = np.zeros(max_params)
chi2_B_accum = np.zeros(max_params)

# Use a fixed seed to create a representative sample for plotting
x_fit_sample, y_fit_sample = generate_data(n_points, sigma, seed=42)
truth_x = np.linspace(0, 1, 500)

# Main trial loop
for seed in range(n_trials):
    x_A, y_A = generate_data(n_points, sigma, seed=seed)
    x_B, y_B = generate_data(n_points, sigma, seed=seed + 1000)

    for m in range(1, max_params + 1):
        coefs = np.polyfit(x_A, y_A, m)
        p = np.poly1d(coefs)

        chi2_a = np.sum(((y_A - p(x_A)) / sigma) ** 2)
        chi2_b = np.sum(((y_B - p(x_B)) / sigma) ** 2)

        chi2_A_accum[m - 1] += chi2_a
        chi2_B_accum[m - 1] += chi2_b

# Average chi-squared values
chi2_A_avg = chi2_A_accum / n_trials
chi2_B_avg = chi2_B_accum / n_trials
degrees = np.arange(1, max_params + 1)

# Theoretical chi-squared
chi2_A_theory, chi2_B_theory = theoretical_chi2(degrees, n_points)

# Smooth curves for plotting
x_dense = np.linspace(1, max_params, 500)
chi2_A_smooth = smooth_curve(degrees, chi2_A_avg, x_dense)
chi2_B_smooth = smooth_curve(degrees, chi2_B_avg, x_dense)
chi2_A_theory_smooth = smooth_curve(degrees, chi2_A_theory, x_dense)
chi2_B_theory_smooth = smooth_curve(degrees, chi2_B_theory, x_dense)

# ---------- Plotting ----------
plt.figure(figsize=(12, 6))

# Left: Fit on a sample
plt.subplot(1, 2, 1)
plt.scatter(x_fit_sample, y_fit_sample, s=10, alpha=0.5, label='Data A')
plt.plot(truth_x, f_truth(truth_x), 'k--', label='Truth Function')
fit_poly = np.poly1d(np.polyfit(x_fit_sample, y_fit_sample, deg=max_params))
plt.plot(truth_x, fit_poly(truth_x), 'r-', label=f'Fit (deg={max_params})')
plt.title(f"Fit to Sample A (Degree {max_params})")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# Right: Chi^2 plot
plt.subplot(1, 2, 2)
plt.plot(x_dense, chi2_A_smooth, 'b-', label=r'$\langle \chi^2(D_A, \theta_A) \rangle$')
plt.plot(x_dense, chi2_B_smooth, 'r-', label=r'$\langle \chi^2(D_B, \theta_A) \rangle$')
plt.plot(x_dense, chi2_A_theory_smooth, 'b--', label=r'Theory: $N - m$')
plt.plot(x_dense, chi2_B_theory_smooth, 'r--', label=r'Theory: $N + m$')
plt.scatter(degrees, chi2_A_avg, color='blue', s=20)
plt.scatter(degrees, chi2_B_avg, color='red', s=20)
plt.xlabel("Model Complexity (Polynomial Degree)")
plt.ylabel(r'$\chi^2$ Value (log scale)')
plt.yscale('log')
plt.title("Average Chi-squared vs Model Complexity")
plt.legend()
plt.grid(True, which="both", linestyle='--')

plt.tight_layout()
plt.show()