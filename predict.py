# ----------------
# predict.py
# ----------------

import os
import numpy as np
import matplotlib.pyplot as plt
from src.fitting import fit_polynomial, fit_polynomial_map
from src.prior import exponential_model_prior, posterior_over_models
from src.utils import generate_data

# Parameters
n_points = 300
sigma = 0.3
max_params = 30
lam_map = 5.0
lam_prior = 0.01  # Looser prior for better BMA spread

# Generate synthetic data
x_data, y_data = generate_data(n_points, sigma, seed=123)

# Model degrees
degrees = np.arange(1, max_params + 1)
x_test = np.linspace(0, 1, 100)

# Fit models and collect predictions
chi2_mle = []
chi2_map = []
y_pred_mle = []
y_pred_map = []

for m in degrees:
    mle_model = fit_polynomial(x_data, y_data, m)
    map_model = fit_polynomial_map(x_data, y_data, m, lam=lam_map)
    y_pred_mle.append(mle_model(x_test))
    y_pred_map.append(map_model(x_test))
    chi2_mle.append(np.sum(((y_data - mle_model(x_data)) / sigma) ** 2))
    chi2_map.append(np.sum(((y_data - map_model(x_data)) / sigma) ** 2))

chi2_mle = np.array(chi2_mle)
chi2_map = np.array(chi2_map)

# Prior and Posterior
prior = exponential_model_prior(max_params, lam=lam_prior)
posterior = posterior_over_models(chi2_mle, prior, sigma_squared=sigma**2)

print("Posterior weights:", np.round(posterior, 4))
print("Sum of posterior:", np.sum(posterior))

# Bayesian Model Averaging (BMA)
y_bma_mean = np.zeros_like(x_test)
y_bma_var = np.zeros_like(x_test)

for i, m in enumerate(degrees):
    map_model = fit_polynomial_map(x_data, y_data, m, lam=lam_map)
    y_pred = map_model(x_test)
    y_bma_mean += posterior[i] * y_pred
    y_bma_var += posterior[i] * (y_pred ** 2)

y_bma_std = np.sqrt(np.maximum(0, y_bma_var - y_bma_mean ** 2))  # Numerical safety

print("BMA std range:", y_bma_std.min(), y_bma_std.max())

# Plot results
plt.figure(figsize=(10, 6))
plt.fill_between(
    x_test,
    y_bma_mean - y_bma_std,
    y_bma_mean + y_bma_std,
    facecolor='gray',
    edgecolor='black',
    linewidth=0.7,
    alpha=0.3,
    label='BMA ± std'
)
plt.plot(x_test, y_bma_mean, label='BMA Mean', color='black', linewidth=2)
plt.plot(x_test, y_pred_mle[np.argmin(chi2_mle)], '--', label='MLE Best Model', color='purple')
plt.plot(x_test, y_pred_map[np.argmax(posterior)], '-', label='MAP Best Model', color='blue')
plt.errorbar(x_data, y_data, yerr=sigma, fmt='.', alpha=0.4, label='Data')

plt.xlabel("x")
plt.ylabel("Predicted K")
plt.title("Prediction of Goodness-of-Fit Statistic (K)")
plt.legend()
plt.grid(True)

os.makedirs("figures", exist_ok=True)
plt.tight_layout()
plt.savefig("figures/predicted_K.png", dpi=300)
plt.show()

# Print predictions at select points
print("\nSample predictions:")
for x_val in [0.1, 0.25, 0.5, 0.75, 0.9]:
    idx = np.abs(x_test - x_val).argmin()
    print(f"x = {x_test[idx]:.3f} --> BMA prediction: {y_bma_mean[idx]:.4f} ± {y_bma_std[idx]:.4f}")