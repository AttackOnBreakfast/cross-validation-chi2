import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Complex truth function
def f_truth(x):
    return 3 * (x + 0.2)**1.2 * (1.2 - x)**1.2 * (1 + 2.3 * x)

# Data generator
def generate_data(n_points=300, sigma=0.4, seed=0):
    np.random.seed(seed)
    x = np.sort(np.random.rand(n_points))
    y = f_truth(x) + sigma * np.random.randn(n_points)
    return x, y

# Parameters
n_points = 300
sigma = 0.3
max_params = 30
n_trials = 1

# Chi2 accumulators
chi2_A_accum = np.zeros(max_params)
chi2_B_accum = np.zeros(max_params)

# One representative sample to visualize the fit
x_fit_sample, y_fit_sample = generate_data(n_points, sigma, seed=42)
truth_x = np.linspace(0, 1, 500)

# Accumulate chi2 values
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

# Averages
chi2_A_avg = chi2_A_accum / n_trials
chi2_B_avg = chi2_B_accum / n_trials

# Theory
degrees = np.arange(1, max_params + 1)
chi2_A_theory = n_points - degrees
chi2_B_theory = n_points + degrees

# Smooth interpolation
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
plt.scatter(x_fit_sample, y_fit_sample, s=10, color='gray', alpha=0.5, label='Data A (Sample)')
plt.plot(truth_x, f_truth(truth_x), 'k--', lw=2, label='Truth Function')
fit_poly = np.poly1d(np.polyfit(x_fit_sample, y_fit_sample, deg=max_params))
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
plt.title("Average Chi-squared over 100 Seeds")
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", linestyle='--')

plt.tight_layout()
plt.show()