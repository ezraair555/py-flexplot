# Quality / Diagnostics Module (v0.6.0+)

Auto data-quality diagnostics for regression-style formulas. Designed to
surface *why* a fit might be off, not to gate-keep model usage.

## Public API

```python
from pyflexplot.quality import diagnose, format_summary
```

---

## `diagnose`

```python
diagnose(formula, data, verbose=True, outlier_threshold=None)
```

Run a diagnostic suite on a flexplot formula + data. Surfaces four
standard regression diagnostics in one call.

### Parameters

- **formula** (str): flexplot formula of the form `y ~ x1 + x2 [+ ...]`. Only the outcome and predictors after `~` are used. Categorical predictors are accepted as part of the formula but excluded from the regression design matrix.
- **data** (pd.DataFrame): non-empty data frame holding the referenced columns.
- **verbose** (bool, default True): if True, prints a one-paragraph summary to stdout.
- **outlier_threshold** (float, optional): Cook's distance cutoff. Default is the conventional `4/n` value. Pass an explicit float to override.

### Returns

`dict` with keys:

- `n_obs` (int): total rows in `data`.
- `n_complete` (int): rows with no missing values across the formula's columns.
- `columns` (list[str]): columns referenced by the formula (numeric only).
- `missing` (dict): per-column missing counts and pattern heuristic (`"none"`, `"concentrated (likely MNAR/MAR)"`, `"spread (likely MCAR)"`).
- `outliers` (dict): Cook's distance count, threshold, max Cook's D, and indices of influential points.
- `linearity` (dict): Ramsey RESET test statistic, p-value, reject/keep, plain-English interpretation.
- `heteroscedasticity` (dict): Breusch-Pagan LM statistic, p-value, reject/keep, plain-English interpretation.
- `_r_squared` (float): OLS R-squared on the complete-case subset (underscored to mark it as internal — not part of the public contract).

### Raises

- `ValueError` if the formula has no outcome or no predictors, or if no numeric predictors are present.

### Examples

```python
import pandas as pd
import numpy as np
from pyflexplot import diagnose, format_summary

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "y": rng.normal(size=200),
    "x": rng.normal(size=200),
})

# Verbose (prints to stdout, returns the dict):
diag = diagnose("y ~ x", data=df)

# Quiet (returns the dict without printing):
diag = diagnose("y ~ x", data=df, verbose=False)
print(diag["linearity"]["reject_linearity"])  # False for clean linear data

# Pre-formatted summary:
print(format_summary(diag))
```

---

## `format_summary`

```python
format_summary(diag)
```

Format a diagnosis dict as a one-paragraph human-readable summary. Suitable
for terminal output, log lines, or email bodies.

### Parameters

- **diag** (dict): output of `diagnose(..., verbose=False)`.

### Returns

`str` — a multi-line summary including the formula's columns, sample
sizes, missingness pattern, outlier count, linearity test verdict, and
heteroscedasticity test verdict.

### Example output

```
Diagnostic for y, x (n=200, complete cases=200)
  Missingness: 0 missing values total; pattern = none.
  Outliers: 0 influential points.
  Linearity: Fail to reject linearity at alpha=0.05.
  Heteroscedasticity: Fail to reject homoscedasticity at alpha=0.05.
  R-squared (OLS reference): 0.812
```

---

## Diagnostics at a glance

| Diagnostic | What it checks | Test |
|---|---|---|
| **Missingness** | Which columns have NAs and whether the pattern looks random | count + heuristic |
| **Outliers** | Influential observations via Cook's distance | Cook's D > `4/n` (or custom threshold) |
| **Linearity** | Whether the functional form is misspecified | Ramsey RESET |
| **Heteroscedasticity** | Whether the residual variance is constant across fitted values | Breusch-Pagan |

### When to use it

- Before fitting: `diagnose("y ~ x1 + x2", data=df)` → check for outliers
  and missingness before training a model.
- After fitting: if the model residuals look strange, run `diagnose()` and
  see whether linearity or heteroscedasticity is rejected.
- For model selection: `diagnose()` is a quick sanity check; a rejected
  linearity test on a simple-LM residual plot is a strong signal that you
  need a non-linear term or transformation.

### Limitations

- Categorical predictors are accepted in the formula but excluded from
  the regression design matrix. If the formula's only predictor is
  categorical, `diagnose()` raises `ValueError`.
- The pattern heuristic for missingness is a rough first cut. For formal
  missing-data tests, see `statsmodels`.
- With `n < 30`, the test statistics may be unreliable (small-sample
  behavior). `diagnose()` still runs but treat results with skepticism.

See `docs/api/core.md` for the `flexplot()` reference and
`docs/examples/diagnostics_workflow.md` for the combined workflow.