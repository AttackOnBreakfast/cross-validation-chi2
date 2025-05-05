# -----------------------------
# src/theory.py
# -----------------------------
import numpy as np

def chi2_theory_A(n_points, degrees):
    return n_points - degrees

def chi2_theory_B(n_points, degrees):
    return n_points + degrees