# Cross-Validation Chi-Squared Analysis

This repository performs a cross-validation study of model fitting and $\chi^2$ analysis using synthetic data. The setup simulates experimental uncertainties and compares fitted vs. predicted chi-squared values across varying model complexities.

---

## 🔍 Features

- **Truth function defined as:**

  $$
  f_{\text{truth}}(x) = 3(x + 0.2)^{1.2}(1.2 - x)^{1.2}(1 + 2.3x)
  $$

- Synthetic data generation with configurable Gaussian noise  
- Polynomial fitting using `np.polyfit` (MLE) and ridge-regularized MAP estimation  
- Reduced $\chi^2$ computed on both training and testing datasets  
- Theoretical predictions for:
  - Mean $\chi^2$
  - Variance of $\chi^2$
- Empirical error bars and theoretical variance bands  
- Empirical vs. theoretical variance plots with Pearson $r$ annotations  
- Bayesian model averaging (BMA) using prior and posterior weights  

---

## 🆕 New Additions

- MAP-based $\chi^2(D_B, \theta_{\mathrm{MAP}}^{(m)})$ curve
- Training curve for MAP: $\chi^2(D_A, \theta_{\mathrm{MAP}}^{(m)})$
- BMA predictions with epistemic uncertainty bands
- Posterior distributions reflecting prior beliefs over model complexity

---

## 📁 Project Structure

```bash
cross-validation-chi2/
│
├── main.py                        # Main simulation and analysis script
├── predict.py                     # BMA-based prediction and uncertainty analysis
├── cross-validation-chi2.py       # Minimal legacy version of the project (keep)
│
├── src/                           # Core source code
│   ├── data.py                    # Data generation (A and B sets)
│   ├── fitting.py                 # Polynomial fits and chi² computations (MLE & MAP)
│   ├── plot.py                    # Visualization tools (fit plots, χ² curves, BMA, etc.)
│   ├── prior.py                   # Priors and posteriors over model degrees
│   ├── theory.py                  # Theoretical formulas for χ² mean and variance
│   ├── truth_function.py          # Defines f_truth(x)
│   └── utils.py                   # Helper functions for pipeline support
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
├── README.md
├── requirements.txt
└── .gitignore

---

## Output Files

- 📄 `results/chi2_dispersion_variance.csv`  
  Table of standard deviations and theoretical variances for each polynomial degree

- 📄 `results/chi2_table.tex`  
  LaTeX-formatted table showing $\sigma_A$, $\sigma_B$, their predicted variances, and posterior weights

- 📊 `figures/combined_fit_and_chi2.png`  
  Two-panel plot: (left) data + truth + fit; (right) $\chi^2$ curves for MLE, MAP, and theory

- 📊 `figures/chi2_var_vs_theory.png`  
  Comparison between empirical and theoretical $\mathrm{Var}(\chi^2)$ with Pearson $r$ annotation

- 📊 `figures/figure3_expectation_std_combined.png`  
  Multi-panel plot of predictive mean ± std at $x = 0.25$, $0.5$, $0.75$ vs. model complexity

- 📊 `figures/predicted_K.png`  
  Bayesian model average prediction of $K$ with uncertainty band and MAP/MLE overlays

---

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- SciPy
- Pandas

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/cross-validation-chi2.git
cd cross-validation-chi2
pip install -r requirements.txt

---

## Usage

To run the full simulation and generate figures and statistical outputs:

```bash
python main.py

---

## Author

**Artemiy Filippov**  
Undergraduate Researcher  
Department of Physics and Astronomy  
Michigan State University (MSU)  
Class of 2027
