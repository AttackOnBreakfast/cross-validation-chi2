# -----------------------------
# src/theory.py
# -----------------------------
def theoretical_chi2(n_points, degrees):
    chi2_A = n_points - degrees
    chi2_B = n_points + degrees
    return chi2_A, chi2_B