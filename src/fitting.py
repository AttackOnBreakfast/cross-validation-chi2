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

def fit_polynomial_map(x, y, degree, lam):
    """
    Maximum A Posteriori (MAP) polynomial fit with L2 prior (ridge regularization).
    
    Parameters:
        x      -- 1D array of input features
        y      -- 1D array of observed data
        degree -- degree of the polynomial
        lam    -- regularization strength (lambda)

    Returns:
        poly1d object representing the fitted polynomial
    """
    X = np.vander(x, degree + 1, increasing=True)
    XtX = X.T @ X
    XtY = X.T @ y
    I = np.eye(degree + 1)
    coefs = inv(XtX + lam * I) @ XtY
    return np.poly1d(coefs[::-1])  # poly1d expects highest degree first

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