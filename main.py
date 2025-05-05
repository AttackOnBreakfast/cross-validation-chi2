import numpy as np
import matplotlib.pyplot as plt
from src.truth_function import f_truth
from src.data import generate_dataset_pair
from src.theory import theoretical_chi2
from src.utils import smooth_curve

# Parameters
n_points = 300
sigma = 0.3
max_params = 30
n_trials = 100

# Accumulators
chi2_A_accum = np.zeros(max_params)
chi2_B_accum = np.zeros(max_params)

# Representative sample for plotting fit
x_fit_sample, y_fit_sample = generate_dataset_pair(n_points, sigma, seed=42)[0]
x_plot = np.linspace(0, 1, 500)

# Loop over random seeds
for seed in range(n_trials):
    (x_A, y_A), (x_B, y_B) = generate_dataset_pair(n_points, sigma, seed)

    for m in range(1, max_params + 1):
        coefs = np.polyfit(x_A, y_A, m)
        p = np.poly1d(coefs)

        chi2_A = np.sum(((y_A - p(x_A)) / sigma) ** 2)
        chi2_B = np.sum(((y_B - p(x_B)) / sigma) ** 2)

        chi2_A_accum[m - 1] += chi2_A
        chi2_B_accum[m - 1] += chi2_B

# Average chi-squared
chi2_A_avg = chi2_A_accum / n_trials
chi2_B_avg = chi2_B_accum / n_trials

# Theory curves
degrees = np.arange(1, max_params + 1)
chi2_A_theory, chi2_B_theory = theoretical_chi2(n_points, degrees)

# Smooth
x_dense = np.linspace(1, max_params, 500)
chi2_A_smooth = smooth_curve(degrees, chi2_A_avg, x_dense)
chi2_B_smooth = smooth_curve(degrees, chi2_B_avg, x_dense)
chi2_A_th_smooth = smooth_curve(degrees, chi2_A_theory, x_dense, kind='linear')
chi2_B_th_smooth = smooth_curve(degrees, chi2_B_theory, x_dense, kind='linear')

# Plot
plt.figure(figsize=(12, 6))

# Fit plot
plt.subplot(1, 2, 1)
plt.scatter(x_fit_sample, y_fit_sample, s=10, color='gray', alpha=0.5, label='Data A (Sample)')
plt.plot(x_plot, f_truth(x_plot), 'k--', lw=2, label='Truth Function')
fit_poly = np.poly1d(np.polyfit(x_fit_sample, y_fit_sample, deg=max_params))
plt.plot(x_plot, fit_poly(x_plot), 'r-', lw=2, label=f'Fit (deg={max_params})')
plt.title(f"Fit to Sample A (deg={max_params})")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# Chi-squared plot
plt.subplot(1, 2, 2)
plt.plot(x_dense, chi2_A_smooth, 'b-', label=r'$\langle \chi^2(D_A, \theta_A) \rangle$')
plt.plot(x_dense, chi2_B_smooth, 'r-', label=r'$\langle \chi^2(D_B, \theta_A) \rangle$')
plt.plot(x_dense, chi2_A_th_smooth, 'b--', label=r'Theory: $N - m$')
plt.plot(x_dense, chi2_B_th_smooth, 'r--', label=r'Theory: $N + m$')
plt.scatter(degrees, chi2_A_avg, color='blue', s=20)
plt.scatter(degrees, chi2_B_avg, color='red', s=20)
plt.xlabel("Model Complexity (Polynomial Degree)")
plt.ylabel(r'$\chi^2$ Value (log scale)')
plt.yscale('log')
plt.title(r"Cross-validated $\chi^2$ vs Model Complexity")
plt.legend()
plt.grid(True, which="both", linestyle='--')

plt.tight_layout()
plt.show()