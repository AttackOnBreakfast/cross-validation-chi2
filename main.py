import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from src.data_generation import generate_data
from src.truth_functions import f_truth
from src.chi2_analysis import compute_chi2_over_trials
from src.plotting import plot_fit_and_chi2

# Parameters
n_points = 300
sigma = 0.3
max_params = 30
n_trials = 100

# Run chi-squared computation
chi2_A_avg, chi2_B_avg = compute_chi2_over_trials(n_points, sigma, max_params, n_trials)

# Generate one sample for visualizing the fit
x_sample, y_sample = generate_data(n_points, sigma, seed=42)

# Generate plot
plot_fit_and_chi2(
    x_sample=x_sample,
    y_sample=y_sample,
    chi2_A=chi2_A_avg,
    chi2_B=chi2_B_avg,
    n_points=n_points,
    sigma=sigma,
    max_params=max_params,
    truth_function=f_truth
)  
