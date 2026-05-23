# Model Comparison • model_comparison

## Description

`model_comparison()` statistically compares the fits of two models, reporting AIC, BIC, log-likelihood, and performing a likelihood ratio test where applicable.

## Usage

```python
model_comparison(model1, model2)
```

## Arguments

| Argument | Description |
|----------|-------------|
| `model1` | First fitted model (typically the simpler/nested model) |
| `model2` | Second fitted model (typically the more complex model) |

## Details

The function computes:

1. **Information Criteria**: AIC and BIC for both models
2. **Log-Likelihood**: Log-likelihood values
3. **Likelihood Ratio Test**: If models are nested, performs LRT with chi-squared distribution

The LRT statistic is calculated as: `2 * (llf_model2 - llf_model1)` with degrees of freedom equal to the difference in model parameters.

## Returns

A tuple containing:
- `DataFrame`: Model comparison statistics (AIC, BIC, LogLik)
- `float`: p-value from likelihood ratio test (if applicable)

## Examples

### Compare Nested Models

```python
import pandas as pd
import statsmodels.api as sm
from pyflexplot import model_comparison

# Load data
df = pd.read_csv("mtcars.csv")

# Fit nested models
model1 = sm.OLS(df['mpg'], sm.add_constant(df[['wt']])).fit()
model2 = sm.OLS(df['mpg'], sm.add_constant(df[['wt', 'hp']])).fit()

# Compare models
comparison, p_value = model_comparison(model1, model2)

print(comparison)
print(f"LRT p-value: {p_value:.4f}")
```

### Interpret Results

```python
# Lower AIC/BIC indicates better fit
# Significant p-value (< 0.05) suggests model2 provides better fit

if p_value < 0.05:
    print("Model 2 provides significantly better fit")
else:
    print("No significant improvement from Model 2")
```

## See Also

- [`estimates()`](estimates.html) - Model effect sizes
- [`compare_fits()`](compare_fits.html) - Visual model comparison
- [`p_format()`](p_format.html) - Format p-values
