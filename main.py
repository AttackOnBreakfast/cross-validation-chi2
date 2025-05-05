# -----------------------------
# main.py
# -----------------------------
# main.py

import numpy as np
import matplotlib.pyplot as plt
from src.truth_function import f_truth
from src.utils import generate_data, rescale
from src.plot import plot_chi2_vs_model_complexity

# Parameters
num_points = 500
x_range = (0, 10)
noise_level = 0.1
max_params = 15
random_state = 42

# Generate x and data
x_data = np.linspace(*x_range, num_points)
y_data, y_truth, y_errors = generate_data(x_data, f_truth, noise_level=noise_level, random_state=random_state)

# Split into two datasets
split_index = int(0.7 * num_points)
x_A, y_A, err_A = x_data[:split_index], y_data[:split_index], y_errors[:split_index]
x_B, y_B, err_B = x_data[split_index:], y_data[split_index:], y_errors[split_index:]

# Run cross-validation and plot
plot_chi2_vs_model_complexity(x_A, y_A, err_A, x_B, y_B, err_B, max_params)