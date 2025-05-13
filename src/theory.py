# -----------------------------
# src/theory.py
# -----------------------------

import numpy as np

def chi2_theory_A(N, m):
    """Expected chi-squared for training set"""
    return N - m

def chi2_theory_B(N, m):
    """Expected chi-squared for test set"""
    return N + m

def chi2_variance_A(N, m):
    """Theoretical variance of chi^2 on training set"""
    return 2 * (N - m)

def chi2_variance_B(N, m):
    """Theoretical variance of chi^2 on test set"""
    return 2 * (N + 3 * m)