# -----------------------------
# main.py
# -----------------------------
from src.truth_function import truth_function
from src.utils import generate_data, rescale
from src.plot import plot_chi2_vs_model_complexity
from numpy.polynomial.chebyshev import chebfit, chebval
import numpy as np
import warnings

# Suppress RankWarnings from Chebyshev fit instability (optional)
warnings.simplefilter('ignore', np.RankWarning)

# Parameters
num_points = 1000
x_range = (0, 10)
max_params = 20

# Generate synthetic dataset
x_data, y_data, y_errors = generate_data(truth_function, num_points, x_range)

# Split data into two sets
split_index = int(0.7 * num_points)
x_A, x_B = x_data[:split_index], x_data[split_index:]
y_A, y_B = y_data[:split_index], y_data[split_index:]
y_err_A, y_err_B = y_errors[:split_index], y_errors[split_index:]

# Initialize chi-squared results
chi2_A_on_A = []
chi2_B_on_A = []

# Loop over model complexities (Chebyshev degrees)
for m in range(1, max_params + 1):
    # Rescale x to [-1, 1] for Chebyshev stability
    x_A_rescaled = rescale(x_A)
    x_B_rescaled = rescale(x_B)

    # Fit model to dataset A using Chebyshev
    coefs = chebfit(x_A_rescaled, y_A, m)

    # Evaluate on A and B
    y_fit_A = chebval(x_A_rescaled, coefs)
    y_fit_B = chebval(x_B_rescaled, coefs)

    # Compute chi-squared
    chi2_A = np.sum(((y_A - y_fit_A) / y_err_A) ** 2)
    chi2_B = np.sum(((y_B - y_fit_B) / y_err_B) ** 2)

    chi2_A_on_A.append(chi2_A)
    chi2_B_on_A.append(chi2_B)

# Plot results
plot_chi2_vs_model_complexity(chi2_A_on_A, chi2_B_on_A, max_params)
