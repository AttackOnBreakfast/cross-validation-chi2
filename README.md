# Cross-Validation Chi-Squared Analysis

This repository performs a cross-validation study of model fitting and $\chi^2$ analysis using synthetic data. The setup simulates experimental uncertainties and compares fitted vs predicted chi-squared values across varying model complexities.

## Features

- Truth function defined as:  
  $$f_{\text{truth}}(x) = 3(x + 0.2)^{1.2}(1.2 - x)^{1.2}(1 + 2.3x)$$
- Synthetic data generated with configurable Gaussian noise
- Polynomial fitting using $\texttt{np.polyfit}$ and $\texttt{np.poly1d}$
- Reduced $\chi^2$ computation on training and testing datasets
- Error bars and variance bands on $\chi^2$ plots
- Output of:
  - `chi2_dispersion_variance.csv`: Numeric data table
  - `chi2_table.tex`: LaTeX-formatted table

## Folder Structure

cross-validation-chi2/
|
├── main.py                  # Entry point: runs data generation, fitting, cross-validation, plotting
|
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── data.py              # Sample generators for training/validation
│   ├── fitting.py           # Polynomial fitting and chi-squared calculations
│   ├── plot.py              # All plotting logic (data + chi2 curves)
│   ├── theory.py            # Theoretical curves: <χ2>, Var(χ2)
│   ├── truth_function.py    # Defines the underlying f_truth(x)
│   └── utils.py             # General helpers (data generation, etc.)
|
├── chi2_dispersion_variance.csv     # Exported numeric results
├── chi2_table.tex                   # LaTeX table for χ2 std and variance
├── requirements.txt                # (Optional) Python dependencies
└── README.md

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- SciPy
- Pandas

## Installation

Clone the repo and install dependencies with pip:

```bash
pip install -r requirements.txt

## Usage

To run the simulation and generate cross-validation plots and statistical analysis:

```bash
python main.py

## Output

Upon running the project, the following files are generated:

- **`chi2_dispersion_variance.csv`**  
  Contains a table of chi-squared standard deviations (dispersions) and theoretical variances for both training and testing datasets.

- **`chi2_table.tex`**  
  LaTeX-formatted table showing $\sigma_A$, $\sigma_B$ and their theoretical variance predictions, suitable for inclusion in reports.

---

## License

This project is licensed under the **MIT License** – feel free to use, modify, and distribute.

---

## Author

**Artemiy Filippov**  
Physics Research @ Michigan State University 2025
