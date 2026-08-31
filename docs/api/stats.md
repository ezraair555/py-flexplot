# Biostatistical Utilities (Fifer Port)

These utilities are ported from the `fifer` and `fifer2` R packages, providing standard biostatistical tools for Python.

> **Coverage:** See [coverage.md](coverage.md) for the full coverage matrix
> vs R's `fifer` / `fifer2`. Some features are partial or v0.7.0 todos.

## `model_comparison`

Compare two statistical models (usually OLS or GLM) and report comparative fit statistics.

```python
from pyflexplot import model_comparison
stats_df, p_value = model_comparison(model1, model2)
```

### Returns
A 2-tuple `(stats_df, p_value)`:

- **stats_df**: A `pandas.DataFrame` (index: `["Model 1", "Model 2"]`) with columns:
  - `AIC`, `BIC`, `LogLik` — always present.
  - `R.squared`, `Adj.R.squared` — present when both models expose them (OLS, GLM).
  - `BayesFactor` — derived from BIC via the Kass & Raftery (1995)
    approximation: `BF = exp((BIC_worse - BIC_better) / 2)`. The better
    model (lower BIC) gets `BF >= 1`; the worse model gets `1/BF`. This
    mirrors R's `fifer::model.comparison()`.
- **p_value**: Likelihood-ratio test p-value (chi-squared). `ValueError`
  if the models are not nested (negative or zero df difference).

### Example

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pyflexplot import model_comparison

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "y": rng.normal(size=200),
    "x1": rng.normal(size=200),
    "x2": rng.normal(size=200),
})
X1 = sm.add_constant(df[["x1"]])
X2 = sm.add_constant(df[["x1", "x2"]])
m1 = sm.OLS(df["y"], X1).fit()
m2 = sm.OLS(df["y"], X2).fit()
stats_df, p = model_comparison(m1, m2)
print(stats_df)
print(f"LRT p = {p:.4f}")
```

---

## `estimates`

Compute a structured effect-size report for a fitted OLS model. A port of
R's `fifer::estimates()` / `flexplot::estimates.lm()`.

```python
from pyflexplot import estimates
report = estimates(ols_model)
```

### Returns
A `dict` with the following keys (v0.6.3+):

| Key | Type | Notes |
|---|---|---|
| `r.squared` | float | OLS R². |
| `adj.r.squared` | float | OLS adjusted R². |
| `sigma` | float | Residual standard error. |
| `n` | int | Number of observations used. |
| `r.squared.ci` | tuple or None | R² confidence interval (Olkin & Finn 1995 non-central-F inversion, v0.7.3+). Real interval, no longer a placeholder. |
| `coef` | DataFrame | `estimate`, `std.error`, `t`, `p.value`, `ci.lower`, `ci.upper` per coefficient. |
| `standardized` | Series | Standardized betas per predictor: `b_j * sd(x_j) / sd(y)`. |
| `semi.p.r2` | Series | Semi-partial R² per predictor, computed via reduced-model fits. |
| `factors` | list of str | Categorical predictor names. |
| `numbers` | list of str | Numeric predictor names. |
| `formula` | str | The fitted formula string. |

### Example

```python
import statsmodels.formula.api as smf
from pyflexplot import estimates

m = smf.ols("y ~ x1 + x2", data=df).fit()
report = estimates(m)
print(report["coef"])
print(report["standardized"])
print(report["semi.p.r2"])
```

---

## `eta_squared` (v0.7.3+)

Partial eta-squared (η²_p) for a fitted OLS model, with a confidence
interval via the same non-central-F inversion that powers
`estimates()["r.squared.ci"]`. Port of R's `sjstats::eta_sq()` /
`fifer::eta_squared()`.

```python
from pyflexplot import eta_squared
df = eta_squared(ols_model)        # one-row DataFrame indexed by "model"
df = eta_squared(ols_model, level=0.90)   # custom CI level
```

### Returns
A `pandas.DataFrame` with one row (`"model"`) and columns:
- `eta_sq` (float): partial eta-squared, `(F * df1) / (F * df1 + df2)`.
- `eta_sq_ci_low` (float | None): CI lower bound (None if degenerate).
- `eta_sq_ci_high` (float | None): CI upper bound (None if degenerate).
- `F` (float): the model's overall F-statistic.

### Notes
- statsmodels' OLS exposes a single model-F, not per-term Fs. So
  `eta_squared()` returns one row indexed by `"model"`, not one row
  per predictor. For per-predictor semi-partial R² (a related but
  distinct quantity), use `estimates()["semi.p.r2"]`.
- η²_p can exceed R² when predictors are correlated; it estimates the
  variance in y explained by *each* predictor after controlling for
  the others.

---

## `compare_fits` (v0.6.3+)

Visually compare how well two pre-fit models match the data, with optional
return of the predictions DataFrame for downstream tooling.

```python
from pyflexplot import compare_fits
p = compare_fits(formula, data, model1, model2)
preds_df = compare_fits(formula, data, model1, model2, return_preds=True)
preds_df = compare_fits(
    formula, data, model1, model2,
    return_preds=True, pred_type="link",  # for GLM
)
```

- `return_preds: bool` — when `True`, returns a DataFrame with the original
  data plus a `pred_<model_name>` column per model.
- `pred_type: {"response", "link"}` — on the response scale (default, applies
  inverse-link for GLMs) or the linear-predictor scale.

For comparing multiple *smoothers* (not pre-fit models), use
`flexplot(..., overlay=[...])` instead — see [`core.md`](core.md).

---

## `p_format`

Formats p-values into standard APA/journal format.

```python
from pyflexplot import p_format
print(p_format(0.000123))  # Outputs: "<.001"
print(p_format(0.0456))    # Outputs: ".046"
```

---

## `eliminated_columns`

Removes columns from a DataFrame that exceed a certain threshold of missing data.

```python
from pyflexplot import eliminated_columns
clean_df = eliminated_columns(df, threshold=0.5)
```

---

## `color_table`

Quickly apply a gradient style to a pandas DataFrame for heat-map visualization of tables.

```python
from pyflexplot import color_table
styled_df = color_table(df, cmap="viridis")
```
*(Requires Jupyter to display styling)*
