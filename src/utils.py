# -----------------------------
# src/utils.py
# -----------------------------
import numpy as np
from src.truth_function import f_truth

def generate_data(n_points=300, sigma=0.4, seed=0):
    np.random.seed(seed)
    x = np.sort(np.random.rand(n_points))
    y = f_truth(x) + sigma * np.random.randn(n_points)
    return x, y