# -----------------------------
# src/truth_fuction.py 
# -----------------------------
import numpy as np

def truth_function(x: np.ndarray) -> np.ndarray:
    """
    Defines the ground truth function f_truth(x).
    For example, a 7th-degree polynomial.
    """
    return 2 + x - 0.5 * x**2 + 0.1 * x**3 - 0.05 * x**4 + 0.02 * x**5 - 0.005 * x**6 + 0.001 * x**7