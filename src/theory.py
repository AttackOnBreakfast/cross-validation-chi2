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

def m_eff_from_width(width, N, alpha=0.6, m_max_ratio=3.0):
    """
    Effective model complexity m_eff as a nonlinear function of network width.
    For small widths, growth is sublinear; for large widths, it saturates around m_max_ratio * N.
    """
    width = np.asarray(width, dtype=float)
    width_norm = width / np.max(width)
    m_eff = N * m_max_ratio * (width_norm ** alpha) / (1 + width_norm ** alpha)
    return m_eff

def chi2_theory_A_NN(N, width):
    m_eff = m_eff_from_width(width, N)
    return N - m_eff

def chi2_theory_B_NN(N, width):
    m_eff = m_eff_from_width(width, N)
    return N + m_eff