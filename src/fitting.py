# -----------------------------
# src/fitting.py
# -----------------------------
from numpy.polynomial.chebyshev import chebfit, chebval

def fit_chebyshev(x, y, degree):
    return chebfit(x, y, degree)

def eval_chebyshev(x, coeffs):
    return chebval(x, coeffs)