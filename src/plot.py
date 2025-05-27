# -----------------------------
# src/plot.py
# -----------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from src.fitting import fit_polynomial
from src.truth_function import f_truth
from src.utils import generate_data

plt.rcParams.update({
    'font.size': 16,               # Master font size (affects most things)
    'axes.titlesize': 16,          # Subplot title
    'axes.labelsize': 16,          # Axis labels
    'xtick.labelsize': 16,         # Tick labels (x)
    'ytick.labelsize': 16,         # Tick labels (y)
    'legend.fontsize': 16,         # Legend text
    'figure.titlesize': 16,        # Global figure suptitle
    'text.color': 'black',         # Default text color
    'axes.edgecolor': 'black',     # Axis border color
    'axes.linewidth': 1.2,         # Thicker axes border
    'font.family': 'sans-serif',   # Optional: you can choose 'serif' or 'DejaVu Sans'
})

def plot_figure1_fit_and_chi2(
    x_sample, y_sample,
    degrees,
    chi2_A, chi2_B,
    chi2_A_std, chi2_B_std,
    chi2_A_theory, chi2_B_theory,
    chi2_A_var_theory, chi2_B_var_theory,
    sigma
):
    truth_x = np.linspace(0, 1, 500)
    x_dense = np.linspace(1, degrees[-1], 500)

    interp_A = interp1d(degrees, chi2_A, kind='cubic')
    interp_B = interp1d(degrees, chi2_B, kind='cubic')
    interp_A_theory = interp1d(degrees, chi2_A_theory, kind='linear')
    interp_B_theory = interp1d(degrees, chi2_B_theory, kind='linear')

    chi2_A_smooth = interp_A(x_dense)
    chi2_B_smooth = interp_B(x_dense)
    chi2_A_theory_smooth = interp_A_theory(x_dense)
    chi2_B_theory_smooth = interp_B_theory(x_dense)

    os.makedirs("figures", exist_ok=True)

    # === One figure, two side-by-side plots, total size 24 x 13.5 inches ===
    fig_fit_and_chi2, (ax1, ax2) = plt.subplots(
        ncols=2,
        figsize=(19, 10),
        gridspec_kw={'width_ratios': [10, 9]}
    )

    # --- Left plot: Sample fit ---
    ax1.errorbar(x_sample, y_sample, yerr=sigma, fmt='.', alpha=0.4, label="Sample A + noise")
    ax1.plot(truth_x, f_truth(truth_x), 'k--', label="Truth function")
    fit_poly = fit_polynomial(x_sample, y_sample, degrees[-1])
    ax1.plot(truth_x, fit_poly(truth_x), 'r-', label=f"Fit deg={degrees[-1]}")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(bottom=0)
    ax1.set_title(f"Sample A Fit (deg={degrees[-1]})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.text(
        0.02, 0.95,
        r"$f_{\mathrm{truth}}(x) = 3(x + 0.2)^{1.2}(1.2 - x)^{1.2}(1 + 2.3x)$",
        transform=ax1.transAxes,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'),
    )
    ax1.grid(True)

    # --- Right plot: Chi-squared curves ---
    ax2.errorbar(degrees, chi2_A, yerr=chi2_A_std, fmt='o', color='blue', label=r"$\chi^2(D_A, \theta_A)$")
    ax2.errorbar(degrees, chi2_B, yerr=chi2_B_std, fmt='s', color='red', label=r"$\chi^2(D_B, \theta_A)$")
    ax2.plot(x_dense, chi2_A_smooth, 'b--')
    ax2.plot(x_dense, chi2_B_smooth, 'r--')
    ax2.plot(x_dense, chi2_A_theory_smooth, 'b:', label=r"Theory: $N - m$")
    ax2.plot(x_dense, chi2_B_theory_smooth, 'r:', label=r"Theory: $N + m$")
    ax2.fill_between(degrees, chi2_A + np.sqrt(chi2_A_var_theory), chi2_A - np.sqrt(chi2_A_var_theory), color='blue', alpha=0.2)
    ax2.fill_between(degrees, chi2_B + np.sqrt(chi2_B_var_theory), chi2_B - np.sqrt(chi2_B_var_theory), color='red', alpha=0.2)
    ax2.set_yscale("log")
    ax2.set_ylim(200, 500)
    ax2.set_xlim(left=0)
    ax2.set_title(r"Cross-validated $\chi^2$ with Error Bars and Variance Bands")
    ax2.set_xlabel("Model Complexity (Polynomial Degree)")
    ax2.set_ylabel(r"$\chi^2$ (log scale)")
    ax2.grid(True, which="both", linestyle="--")
    ax2.legend(loc='lower left')

    # Annotate deg=1 outlier
    ax2.annotate(
    r'$\chi^2 \approx 1100$', 
    xy=(4, 470),      # destination point (arrowhead)
    xytext=(5, 420),  # starting point (text label)
    arrowprops=dict(arrowstyle='->', color='gray', lw=1),
    ha='center', color='gray', fontsize=16
    )

    # Final spacing and save
    fig_fit_and_chi2.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.10, wspace=0.25)
    fig_fit_and_chi2.savefig("figures/combined_fit_and_chi2.png", dpi=300)
    plt.show()

def plot_figure2_variance_comparison(chi2_A_std, chi2_B_std, chi2_A_var_theory, chi2_B_var_theory):
    empirical_var_A = chi2_A_std ** 2
    empirical_var_B = chi2_B_std ** 2

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(chi2_A_var_theory, empirical_var_A, 'o-', color='blue', label=r"Training: $\chi^2_A$")
    ax.plot(chi2_B_var_theory, empirical_var_B, 's-', color='red', label=r"Test: $\chi^2_B$")

    # Identity line
    all_x = np.concatenate([chi2_A_var_theory, chi2_B_var_theory])
    all_y = np.concatenate([empirical_var_A, empirical_var_B])
    min_val = min(all_x.min(), all_y.min())
    max_val = max(all_x.max(), all_y.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label=r"$y = x$")

    # Labels and title
    ax.set_xlabel(r"Theoretical $\mathrm{Var}(\chi^2)$")
    ax.set_ylabel(r"Empirical $\mathrm{Var}(\chi^2)$")
    ax.set_title(r"Empirical vs Theoretical Variance of $\chi^2$")
    ax.grid(True)
    ax.legend()

    # Pearson r
    r_A, _ = pearsonr(chi2_A_var_theory, empirical_var_A)
    r_B, _ = pearsonr(chi2_B_var_theory, empirical_var_B)
    ax.text(0.025, 0.975, rf"$r_A = {r_A:.3f}$" + "\n" + rf"$r_B = {r_B:.3f}$",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

    # Smart zooming
    x_lo, x_hi = np.percentile(all_x, [2.5, 97.5])
    y_lo, y_hi = np.percentile(all_y, [2.5, 97.5])
    ax.set_xlim(x_lo - 0.1 * (x_hi - x_lo), x_hi + 0.1 * (x_hi - x_lo))
    ax.set_ylim(y_lo - 0.1 * (y_hi - y_lo), y_hi + 0.1 * (y_hi - y_lo))

    plt.tight_layout()
    plt.savefig("figures/chi2_var_vs_theory.png", dpi=300)
    plt.show()

def plot_figure3_expectation_and_std_vs_degree(
    n_trials,
    n_points,
    sigma,
    max_degree,
    x_targets=[0.25, 0.5, 0.75]
):
    os.makedirs("figures", exist_ok=True)
    degrees = np.arange(0, max_degree + 1)
    x_targets = np.asarray(x_targets)

    means = {x: [] for x in x_targets}
    stds = {x: [] for x in x_targets}
    truth_vals = {x: f_truth(x) for x in x_targets}

    for m in degrees:
        preds = {x: [] for x in x_targets}
        for seed in range(n_trials):
            x_train, y_train = generate_data(n_points, sigma, seed)
            model = fit_polynomial(x_train, y_train, m)
            for x in x_targets:
                preds[x].append(model(x))

        for x in x_targets:
            values = np.array(preds[x])
            means[x].append(np.mean(values))
            stds[x].append(np.std(values))

    fig, axes = plt.subplots(len(x_targets), 1, figsize=(14, 3.5 * len(x_targets)), sharex=True)
    if len(x_targets) == 1:
        axes = [axes]

    colors = plt.get_cmap("tab10")

    for i, (x, ax) in enumerate(zip(x_targets, axes)):
        mean_arr = np.array(means[x])
        std_arr = np.array(stds[x])
        truth_val = truth_vals[x]
        color = colors(i)

        ax.plot(degrees, mean_arr, label=fr"$\mathbb{{E}}[y(x={x})]$", color=color)
        ax.fill_between(degrees, mean_arr - std_arr, mean_arr + std_arr, alpha=0.2, color=color)
        ax.axhline(truth_val, linestyle=":", color=color, alpha=0.8,
                   label=fr"$f_{{\mathrm{{truth}}}}(x={x})$")
        ax.set_ylabel("y value")
        ax.set_title(fr"Prediction mean and std at $x = {x}$ vs Number of Parameters (m)")
        ax.grid(True, which="both", linestyle="--")
        ax.legend(loc="center right")
        ax.set_xlim(left=0, right=max_degree + 1)

    axes[-1].set_xlabel("Model Complexity (Polynomial Degree)")

    secax = axes[0].secondary_xaxis('top', functions=(lambda d: d + 1, lambda p: p - 1))

    fig.tight_layout()
    fig.savefig("figures/figure3_expectation_std_combined.png", dpi=300)
    plt.show()