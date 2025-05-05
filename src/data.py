# -----------------------------
# src/data.py
# -----------------------------
import numpy as np
from src.truth import f_truth

def generate_dataset_pair(n_points, sigma, seed):
    np.random.seed(seed)
    x = np.sort(np.random.rand(n_points))
    y1 = f_truth(x) + sigma * np.random.randn(n_points)
    y2 = f_truth(x) + sigma * np.random.randn(n_points)
    return (x, y1), (x, y2)

def rescale(x):
    return 2 * (x - x.min()) / (x.max() - x.min()) - 1
