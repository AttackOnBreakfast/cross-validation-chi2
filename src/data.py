# src/data.py

import numpy as np

def f_truth(x):
    return 3 * (x + 0.2)**1.2 * (1.2 - x)**1.2 * (1 + 2.3 * x)

def generate_dataset_pair(n_points=300, sigma=0.3, seed_a=0, seed_b=1):
    def generate(seed):
        np.random.seed(seed)
        x = np.sort(np.random.rand(n_points))
        y = f_truth(x) + sigma * np.random.randn(n_points)
        return x, y
    return generate(seed_a), generate(seed_b)