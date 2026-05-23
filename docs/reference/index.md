# Package Index • py-flexplot

---

## Creating Explanations

These functions are the bread and butter of py-flexplot and are used to create intelligent visualizations from data and models.

| Function | Description |
|----------|-------------|
| [`flexplot()`](flexplot.html) | Intelligent multivariate graphics via formulas |
| [`visualize()`](visualize.html) | Provides a visual representation of a fitted statistical object |
| [`compare_fits()`](compare_fits.html) | Visually compare the fit of two different models |
| [`added_plot()`](added_plot.html) | Generates an added variable plot (partial regression plot) |

---

## Statistical Utilities

Functions for model comparison, effect sizes, and data formatting.

| Function | Description |
|----------|-------------|
| [`model_comparison()`](model_comparison.html) | Statistically compares the fits of two models (AIC, BIC, LRT) |
| [`estimates()`](estimates.html) | Reports effect sizes and model summaries |
| [`p_format()`](p_format.html) | Formats p-values (e.g., <.001) |
| [`eliminated_columns()`](eliminated_columns.html) | Removes columns with too many missing values |
| [`color_table()`](color_table.html) | Returns a styled pandas dataframe with gradient |

---

## Empirical Bayes Estimation

Functions for empirical Bayes binomial estimation (ported from ebbr).

| Function | Description |
|----------|-------------|
| [`fit_beta_prior()`](fit_beta_prior.html) | Fits a beta prior to binomial data using MLE or moments |
| [`add_ebb_estimate()`](add_ebb_estimate.html) | Adds empirical Bayes estimates to a dataframe |

---

## Structural Equation Modeling

Visualization tools for SEM models (ported from flexplavaan).

| Function | Description |
|----------|-------------|
| [`hopper_plot()`](hopper_plot.html) | Visualize residuals from variance/covariance matrix |
| [`disturbance_plot()`](disturbance_plot.html) | Visualize association between two variables after removing model-implied fit |
| [`measurement_plot()`](measurement_plot.html) | Visualize relationship between a latent variable and its indicator |

---

## Installation

```bash
pip install py-flexplot
```

## Quick Start

```python
import pandas as pd
from pyflexplot import flexplot, visualize

# Load data
df = pd.read_csv("your_data.csv")

# Create intelligent visualization
p = flexplot("outcome ~ predictor | grouping_var", data=df)
p.show()

# Visualize a fitted model
import statsmodels.api as sm
model = sm.OLS(df['y'], df['x']).fit()
visualize(model, data=df)
```

---

## See Also

- [GitHub Repository](https://github.com/ezraair555/py-flexplot)
- [Examples](../examples/index.html)
- [API Reference](api/pyflexplot.html)

---

*py-flexplot is a Python port of Dustin Fife's R flexplot package with additional tools for empirical Bayes estimation and SEM visualization.*
