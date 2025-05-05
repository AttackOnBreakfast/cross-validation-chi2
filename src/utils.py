# -----------------------------
# src/utils.py
# -----------------------------
import numpy as np
from typing import Tuple

def generate_data(x: np.ndarray, truth_fn, noise_level: float, random_state=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic data around a truth function with noise.
    The noise is proportional to the function value.
    """
    rng = np.random.default_rng(random_state)
    y_truth = truth_fn(x)
    noise = rng.normal(loc=0.0, scale=noise_level * y_truth)
    y_noisy = y_truth + noise
    y_noisy = np.clip(y_noisy, 0, None)  # ensure non-negative
    return y_noisy, y_truth

def rescale(x: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
    """
    Rescale x to the interval [new_min, new_max].
    """
    old_min = np.min(x)
    old_max = np.max(x)
    return new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)