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

# === Neural Network Theory (with m_eff(step)) ===
def m_eff_from_step(step, max_step, max_m):
    """
    Define effective complexity m_eff as a function of training step.
    Maps step in [0, max_step] → m_eff in [1, max_m].
    Linear mapping for now, can be refined later.
    """
    step = np.asarray(step)
    return 1 + (max_m - 1) * (step / max_step)

def chi2_theory_A_step(N, step, max_step, max_m):
    m_eff = m_eff_from_step(step, max_step, max_m)
    return N - m_eff

def chi2_theory_B_step(N, step, max_step, max_m):
    m_eff = m_eff_from_step(step, max_step, max_m)
    return N + m_eff

def chi2_variance_A_step(N, step, max_step, max_m):
    m_eff = m_eff_from_step(step, max_step, max_m)
    return 2 * (N - m_eff)

def chi2_variance_B_step(N, step, max_step, max_m):
    m_eff = m_eff_from_step(step, max_step, max_m)
    return 2 * (N + 3 * m_eff)