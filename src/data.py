# -----------------------------
# data.py
# -----------------------------
import numpy as np
from src.truth_function import f_truth

def generate_dataset_pair(n_points, sigma, seed):
    """
    Generate two datasets A and B with independent noise.
    """
    np.random.seed(seed)
    x_A = np.sort(np.random.rand(n_points))
    y_A = f_truth(x_A) + sigma * np.random.randn(n_points)

    np.random.seed(seed + 1000)
    x_B = np.sort(np.random.rand(n_points))
    y_B = f_truth(x_B) + sigma * np.random.randn(n_points)

    return (x_A, y_A), (x_B, y_B)

def generate_fit_sample(n_points, sigma, seed):
    """
    Generate a representative data sample to visualize polynomial fitting.
    """
    np.random.seed(seed)
    x = np.sort(np.random.rand(n_points))
    y = f_truth(x) + sigma * np.random.randn(n_points)
    return x, y
