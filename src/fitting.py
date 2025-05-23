# -----------------------------
# src/fitting.py
# -----------------------------
import numpy as np
from numpy.linalg import inv

def fit_polynomial(x, y, degree):
    """
    Maximum Likelihood Estimation (MLE) polynomial fit.
    Fits a polynomial of specified degree to (x, y) data using least squares.
    """
    coefs = np.polyfit(x, y, degree)
    return np.poly1d(coefs)

def compute_chi2(y_true, y_pred, sigma):
    """
    Computes the chi-squared statistic for the fit.
    
    Parameters:
        y_true -- true observed values
        y_pred -- predicted values from the model
        sigma  -- assumed noise standard deviation

    Returns:
        Chi-squared value as a float
    """
    return np.sum(((y_true - y_pred) / sigma) ** 2)