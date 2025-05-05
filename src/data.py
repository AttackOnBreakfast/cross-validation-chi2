# src/data.py

import numpy as np

def f_truth(x):
    """
    Complex 'true' function used to generate synthetic data.
    """
    return 3 * (x + 0.2)**1.2 * (1.2 - x)**1.2 * (1 + 2.3 * x)

def generate_data(n_points=300, sigma=0.4, seed=None):
    """
    Generate noisy data around the true function.
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.sort(np.random.rand(n_points))
    y = f_truth(x) + sigma * np.random.randn(n_points)
    return x, y

def generate_dataset_pair(n_points=300, sigma=0.4, seed=None):
    """
    Generate two independent datasets A and B using different seeds.
    """
    x_A, y_A = generate_data(n_points=n_points, sigma=sigma, seed=seed)
    x_B, y_B = generate_data(n_points=n_points, sigma=sigma,
                             seed=seed + 1000 if seed is not None else None)
    return (x_A, y_A), (x_B, y_B)