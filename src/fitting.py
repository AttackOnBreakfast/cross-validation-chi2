# -----------------------------
# src/fitting.py
# -----------------------------
import numpy as np

def fit_polynomial(x, y, degree):
    coefs = np.polyfit(x, y, degree)
    return np.poly1d(coefs)


def compute_chi2(y_true, y_pred, sigma):
    return np.sum(((y_true - y_pred) / sigma) ** 2)