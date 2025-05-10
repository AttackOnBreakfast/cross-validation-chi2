# Cross-Validation Chi-Squared Analysis

This repository performs a cross-validation study of model fitting and $\chi^2$ analysis using synthetic data. The setup simulates experimental uncertainties and compares fitted vs predicted chi-squared values across varying model complexities.

---

## Features

- Truth function defined as:

  $$
  f_{\text{truth}}(x) = 3(x + 0.2)^{1.2}(1.2 - x)^{1.2}(1 + 2.3x)
  $$

- Synthetic data generated with configurable Gaussian noise
- Polynomial fitting using `np.polyfit` and `np.poly1d`
- Reduced $\chi^2$ computation on training and testing datasets
- Error bars and variance bands on $\chi^2$ plots
- Output files:
  - `results/chi2_dispersion_variance.csv`: numeric table of dispersions and variances
  - `results/chi2_table.tex`: LaTeX-formatted version
  - `figures/chi2_plot.png`: visualization of cross-validated chi-squared with error bars

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
```bash

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
├── main.py                        # Main script: generates data, fits models, plots
├── cross-validation-chi2.py       # Working reference file (legacy)
│
├── src/                           # Source code
│   ├── data.py                    # Data generation utilities
│   ├── fitting.py                 # Polynomial fit and chi2 calculation
│   ├── plot.py                    # Plotting logic
│   ├── theory.py                  # Theoretical expectations for <χ²> and Var(χ²)
│   ├── truth_function.py          # Defines the underlying truth function
│   └── utils.py                   # Generic helper functions
│
├── results/                       # Generated output
│   ├── chi2_dispersion_variance.csv
│   └── chi2_table.tex
│
├── figures/                       # Plot images
│   └── chi2_plot.png
│
├── chi2_dispersion.tex            # LaTeX explanation: empirical dispersion
├── chi2_variance.tex              # LaTeX explanation: theoretical variance
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## License

This project is licensed under the **MIT License** – feel free to use, modify, and distribute.

---

## Author

**Artemiy Filippov**  
Physics Research @ Michigan State University 2025
