# -----------------------------
# src/truth_fuction.py 
# -----------------------------
import numpy as np

def f_truth(x: np.ndarray) -> np.ndarray:
    """
    The ground-truth function used to generate synthetic data.
    """
    return np.exp(-0.3 * x) * (1 + 0.2 * np.sin(3 * x))