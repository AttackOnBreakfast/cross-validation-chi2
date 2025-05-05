# -----------------------------
# src/plot.py
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from src.truth_function import f_truth

def plot_chi2_vs_model_complexity(x_A, y_A, err_A, x_B, y_B, err_B, max_params):
    chi2_values_A = []
    chi2_values_B = []

    for m in range(1, max_params + 1):
        # Fit on dataset A
        coefs = np.polyfit(x_A, y_A, m)
        poly = np.poly1d(coefs)

        # Chi-squared on A
        chi2_A = np.sum(((y_A - poly(x_A)) / err_A)**2)
        chi2_values_A.append(chi2_A / len(y_A))

        # Chi-squared on B
        chi2_B = np.sum(((y_B - poly(x_B)) / err_B)**2)
        chi2_values_B.append(chi2_B / len(y_B))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_params + 1), chi2_values_A, label=r"$\chi^2_A$", marker='o')
    plt.plot(range(1, max_params + 1), chi2_values_B, label=r"$\chi^2_B$", marker='s')
    plt.xlabel("Polynomial Degree (Model Complexity)")
    plt.ylabel("Reduced $\chi^2$")
    plt.title("Cross-validated $\chi^2$ vs Model Complexity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()