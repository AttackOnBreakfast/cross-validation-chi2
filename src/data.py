# -----------------------------
# data.py
# -----------------------------
import numpy as np
from src.truth_function import f_truth

def generate_fit_sample(n_points, sigma, seed):
    """
    Generate a representative data sample to visualize polynomial fitting.
    """
    np.random.seed(seed)
    x = np.sort(np.random.rand(n_points))
    y = f_truth(x) + sigma * np.random.randn(n_points)
    return x, y
