# -----------------------------
# src/truth_fuction.py 
# -----------------------------
import numpy as np

def f_truth(x: np.ndarray) -> np.ndarray:
    """
    Defines the theoretical 'truth' function for the model.
    """
    return 3 + 0.5 * x - 0.2 * x**2 + 0.03 * x**3 - 0.001 * x**4