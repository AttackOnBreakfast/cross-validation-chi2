# -----------------------------
# src/truth_fuction.py 
# -----------------------------
import numpy as np

def f_truth(x: np.ndarray) -> np.ndarray:
    """
    The ground-truth function used to generate synthetic data.
    """
    return 3 * (x + 0.2)**1.2 * (1.2 - x)**1.2 * (1 + 2.3 * x)