# Biostatistical Utilities (Fifer Port)

These utilities are ported from the `fifer` and `fifer2` R packages, providing standard biostatistical tools for Python.

> **Coverage:** See [coverage.md](coverage.md) for the full coverage matrix
> vs R's `fifer` / `fifer2`. Some features are partial or v0.7.0 todos.

## `model_comparison`

Compare two statistical models (usually OLS or GLM) and report comparative fit statistics.

```python
from pyflexplot import model_comparison
stats_df, p_value = model_comparison(model1, model2)

# With pred.difference (v0.8.0+, R parity):
stats_df, p_value, pred_diff = model_comparison(model1, model2, return_pred_difference=True)
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
- **p_value**: Likelihood-ratio test p-value (chi-squared) when the
  models are nested; `None` for non-nested pairs (v0.8.0+, R parity —
  AIC/BIC/BF remain valid without nesting). v0.7.x raised `ValueError`.
- **pred_diff** (only with `return_pred_difference=True`): Series of
  prediction-difference quantiles (0/25/50/75/100%), or None if the
  models' predictions can't be aligned.

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
| `factor_estimates` | DataFrame | Per-factor-level fitted means with CIs (v0.8.0+; R's "Estimates for Factors"). |
| `mean_differences` | DataFrame | Pairwise factor-level contrasts with CIs and `cohens.d` (v0.8.0+; gated on `mc=`). |
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

## `eta_squared` (v0.7.5+)

Per-term partial eta-squared (η²_p) for a fitted OLS model, with a CI via
the same non-central-F inversion that powers `estimates()["r.squared.ci"]`.
Port of R's `sjstats::eta_sq()` / `fifer::eta_squared()`.

```python
from pyflexplot import eta_squared
df = eta_squared(ols_model)                   # type-III SS (R's default)
df = eta_squared(ols_model, typ=2)             # type-II SS (hierarchical)
df = eta_squared(ols_model, level=0.90)        # custom CI level
```

### Returns
A `pandas.DataFrame` with **one row per non-intercept term** (v0.7.5+;
prior versions returned one row indexed by `"model"`):

| Column | Type | Notes |
|---|---|---|
| `eta_sq` | float | Partial eta-squared for the term: `(F * df_term) / (F * df_term + df_resid)`. |
| `eta_sq_ci_low` | float \| None | CI lower bound (None if degenerate). |
| `eta_sq_ci_high` | float \| None | CI upper bound (None if degenerate). |
| `F` | float | Per-term F-statistic. |
| `p_value` | float | Per-term p-value. |
| `df` | int | Per-term degrees of freedom. |

### Method
v0.7.5: uses `statsmodels.stats.anova.anova_lm(model, typ=typ)` to compute
type-I, II, or III sums of squares per term. Type III is the default
and matches R's `car::Anova(..., type=3)` semantics. For categorical
predictors, `df` reflects the number of levels minus 1.

### Notes
- v0.7.3 returned a single-row DataFrame indexed by `"model"` (the
  overall model partial η², not per-term). v0.7.5 changed the surface
  to return per-term rows. If you need the old behavior, sum the
  per-term `eta_sq` weighted by their `df` (which recovers the model
  total).
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


---

## Standalone accessors (v0.8.0+)

Thin, R-named accessors over quantities `estimates()` already computes:

### `standardized_beta(model)`

```python
from pyflexplot import standardized_beta
betas = standardized_beta(ols_model)   # Series indexed by predictor
```

Note: entries for categorical dummy columns use the dummy column's SD —
treat those with care (R reports factor levels differently).

### `rsq_change(reduced_model, full_model)`

```python
from pyflexplot import rsq_change
delta = rsq_change(m_reduced, m_full)   # float
```

### `bf_bic(model1, model2)`

```python
from pyflexplot import bf_bic
bf12 = bf_bic(m1, m2)   # exp((BIC2 - BIC1)/2); >1 favors m1
```
