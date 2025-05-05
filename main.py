# -----------------------------
# main.py
# -----------------------------
import numpy as np
from src.truth_function import f_truth
from src.utils import generate_data, rescale
from src.plot import plot_chi2_vs_model_complexity
from numpy.polynomial.chebyshev import chebfit, chebval
import warnings

warnings.simplefilter('ignore', np.RankWarning)

# Parameters
num_points = 100
x_range = (-0.2, 1.2)
noise_level = 0.2
max_degree = 15
random_seed = 42

# Generate datasets A and B
rng = np.random.default_rng(random_seed)
x_data = np.linspace(x_range[0], x_range[1], num_points)
rng.shuffle(x_data)
x_A, x_B = np.array_split(x_data, 2)

y_A, y_A_truth = generate_data(x_A, f_truth, noise_level, random_state=0)
y_B, y_B_truth = generate_data(x_B, f_truth, noise_level, random_state=1)

# Chebyshev basis fitting with rescaled x values
x_A_scaled = rescale(x_A)
x_B_scaled = rescale(x_B)

degrees = np.arange(1, max_degree + 1)
chi2_A = []
chi2_B = []

for deg in degrees:
    coefs = chebfit(x_A_scaled, y_A, deg)
    y_fit_A = chebval(x_A_scaled, coefs)
    y_fit_B = chebval(rescale(x_B), coefs)

    chi2_A_val = np.sum(((y_A - y_fit_A) / (noise_level * y_A_truth))**2) / len(y_A)
    chi2_B_val = np.sum(((y_B - y_fit_B) / (noise_level * y_B_truth))**2) / len(y_B)

    chi2_A.append(chi2_A_val)
    chi2_B.append(chi2_B_val)

plot_chi2_vs_model_complexity(degrees, chi2_A, chi2_B)