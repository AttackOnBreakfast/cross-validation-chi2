# Cross-Validation Chi-Squared Analysis

This repository performs a cross-validation study of model fitting and $\chi^2$ analysis using synthetic data. The setup simulates experimental uncertainties and compares fitted vs predicted chi-squared values across varying model complexities.

---

## Features

- **Truth function defined as:**

  $$f_{\text{truth}}(x) = 3(x + 0.2)^{1.2}(1.2 - x)^{1.2}(1 + 2.3x)$$

- Synthetic data generated with configurable Gaussian noise  
- Polynomial fitting using `np.polyfit` (MLE) and ridge-regularized MAP estimation  
- Reduced $\chi^2$ computed on training and testing datasets  
- Theoretical predictions for:
  - Mean $\chi^2$
  - Variance of $\chi^2$
- Empirical error bars and theoretical variance bands on $\chi^2$ plots  
- Empirical vs theoretical variance plots with Pearson $r$ annotation
- **New:**
  - MAP curve $\chi^2(D_B, \theta_{\mathrm{MAP}}^{(m)})$ shown across degrees
  - MAP training curve $\chi^2(D_A, \theta_{\mathrm{MAP}}^{(m)})$ now included
  - Vertical lines at MLE and MAP-selected degrees

---

## Output Files

- 📄 `results/chi2_dispersion_variance.csv`  
  Table of standard deviations and theoretical variances for each polynomial degree

- 📄 `results/chi2_table.tex`  
  LaTeX-formatted table showing $\sigma_A$, $\sigma_B$, and their predicted variances

- 📊 `figures/chi2_cross_validation.png`  
  Log-scaled plot of $\chi^2$ vs model complexity with error bars and variance bands

- 📊 `figures/combined_model_fit_and_chi2.png`  
  Two-panel plot: (top) truth vs MLE vs MAP fit; (bottom) $\chi^2$ curves for MLE, MAP, and theory

- 📊 `figures/chi2_var_vs_theory.png`  
  Zoomed-in plot comparing empirical and theoretical variance with correlation coefficients

- 📊 `figures/prior_vs_posterior.png`  
  Plot comparing exponential prior and posterior probability over degrees

---

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- SciPy
- Pandas

---

## Installation

Clone the repo and install dependencies with pip:

```bash
pip install -r requirements.txt
```

---

## Usage

To run the simulation and generate cross-validation plots and statistical analysis:

```bash
python main.py
```

---

## Output

Upon running the project, the following files are generated:

- `results/chi2_dispersion_variance.csv`  
  Contains a table of chi-squared standard deviations (dispersions) and theoretical variance predictions

- `results/chi2_table.tex`  
  LaTeX-formatted table showing $\sigma_A$, $\sigma_B$ and their theoretical variance predictions

- `figures/chi2_plot.png`  
  Graph of $\chi^2$ vs model complexity with error bars and theoretical variance bands

---

## 📁 Project Structure

```bash
cross-validation-chi2/
│
├── main.py                        # Main simulation and analysis script
├── cross-validation-chi2.py       # Reference/legacy version
│
├── src/                           # Core source code
│   ├── data.py                    # Data generation
│   ├── fitting.py                 # Polynomial fitting & chi² computation (MLE + MAP)
│   ├── plot.py                    # Visualization code including MAP chi² curves
│   ├── theory.py                  # Theoretical mean and variance formulas
│   ├── truth_function.py          # Truth function definition
│   └── utils.py                   # Generic helper utilities
│
├── results/                       # Numeric and LaTeX outputs
│   ├── chi2_dispersion_variance.csv
│   └── chi2_table.tex
│
├── figures/                       # Generated visualizations
│   ├── chi2_cross_validation.png
│   ├── combined_model_fit_and_chi2.png
│   ├── chi2_var_vs_theory.png
│   └── prior_vs_posterior.png
│
├── chi2_dispersion.tex            # Notes on empirical dispersion calculation
├── chi2_variance.tex              # Notes on theoretical chi² variance
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Author

**Artemiy Filippov**  
Physics Research @ Michigan State University 2025
