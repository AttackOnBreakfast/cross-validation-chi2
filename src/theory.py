# -----------------------------
# src/theory.py
# -----------------------------

import numpy as np

# === Polynomial Model Theory ===
def chi2_theory_A(N, m):
    """Expected chi-squared for training set (polynomial case)"""
    return N - m

def chi2_theory_B(N, m):
    """Expected chi-squared for test set (polynomial case)"""
    return N + m

def chi2_variance_A(N, m):
    """Theoretical variance of chi^2 on training set (polynomial case)"""
    return 2 * (N - m)

def chi2_variance_B(N, m):
    """Theoretical variance of chi^2 on test set (polynomial case)"""
    return 2 * (N + 3 * m)
