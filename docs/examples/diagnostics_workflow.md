# Diagnostic + Visualization Workflow

This example shows the typical py-flexplot workflow for fitting and
diagnosing a regression-style model: run data-quality checks first,
then visualize the fit with uncertainty bands and overlay smoothers.

## 1. Set up the data

```python
import numpy as np
import pandas as pd
from pyflexplot import flexplot, diagnose, format_summary

rng = np.random.default_rng(42)
n = 300
df = pd.DataFrame({
    "x1": rng.normal(size=n),
    "x2": rng.choice(["control", "treatment"], size=n),
    "y": (
        1.5 * rng.normal(size=n)
        + 0.8 * (rng.normal(size=n) ** 2)   # non-linear term
        + (rng.normal(size=n) * (1 + 1.5 * (rng.choice(["control", "treatment"], size=n) == "treatment")))  # heteroscedastic
    ),
})
```

This data has two known issues: a quadratic relationship and a
group-dependent variance. `diagnose()` should flag both.

## 2. Run data-quality diagnostics

```python
diag = diagnose("y ~ x1 + x2", data=df, verbose=True)
```

Output (one paragraph, terminal/email/log-friendly):

```
Diagnostic for y, x1, x2 (n=300, complete cases=300)
  Missingness: 0 missing values total; pattern = none.
  Outliers: 4 influential points (max Cook's D = 0.183).
  Linearity: Reject linearity at alpha=0.05; functional form may be misspecified.
  Heteroscedasticity: Reject homoscedasticity at alpha=0.05; variance is non-constant.
  R-squared (OLS reference): 0.412
```

The linearity and heteroscedasticity rejections confirm what we baked into
the synthetic data. The dict returned by `diagnose(..., verbose=False)`
also includes the raw test statistics (`p_value`, `statistic`) for each
test — drill in if you want to confirm the rejection thresholds or run
your own analysis.

## 3. Visualize with uncertainty bands

A single 95% CI is the default. For Tufte-style multi-ribbon display:

```python
p = flexplot(
    "y ~ x1 + x2", data=df,
    bands=[0.5, 0.8, 0.95],   # nested coverage ribbons
)
p.draw()
```

For a prediction interval (band on new observations, not the mean):

```python
p = flexplot(
    "y ~ x1 + x2", data=df,
    method="lm", uncertainty="prediction", level=0.95,
)
p.draw()
```

For a case-resampled bootstrap CI on the loess smoother (the theoretical
CI on loess is weak; bootstrap is the honest path):

```python
p = flexplot(
    "y ~ x1 + x2", data=df,
    method="loess", uncertainty="bootstrap",
)
p.draw()
```

## 4. Overlay competing smoothers

The `overlay=` parameter draws multiple smoothers on the same axes.
With `label=`, a legend distinguishes them.

```python
p = flexplot(
    "y ~ x1", data=df,
    method="loess",  # primary smoother
    overlay=[
        "lm",   # additive linear (parallel slopes per group)
        {"method": "loess", "span": 0.3, "label": "LOESS (span=0.3)"},
        {"method": "rlm",   "label": "Robust regression"},
    ],
)
p.draw()
```

Use `overlay` to *see* which fit the data prefers — for our quadratic +
heteroscedastic data, the LOESS overlay should track the curvature while
the LM overlay stays straight.

## 5. Disable the fit (scatter only)

Pass `uncertainty = None` to draw the scatter without any fitted line
(useful when you only want to inspect the raw data).

```python
p = flexplot("y ~ x1 + x2", data=df, uncertainty=None)
p.draw()
```

## 6. R-style interaction syntax (forward-compatible)

`flexplot` accepts R-style interaction operators since v0.6.2:

```python
import warnings

with warnings.catch_warnings():
    # v0.6.x emits a UserWarning noting that the fit is additive.
    warnings.simplefilter("ignore", UserWarning)
    p = flexplot("y ~ x1*x2", data=df)
    p.draw()
```

The parser expands `x1*x2` to `x1 + x2 + x1:x2` for column lookup and
preserves `x1:x2` in the term list for v0.7.0 (where non-parallel slopes
per group will become available).

## Summary

`py-flexplot` now covers a full modeling diagnostic → visualization
loop:

1. `diagnose()` — four standard regression diagnostics in one call
2. `flexplot()` with `uncertainty=` — first-class band types
3. `flexplot()` with `bands=` — Tufte-style nested ribbons
4. `flexplot()` with `overlay=` — multi-smoother comparison
5. R-style formula syntax with forward-compatible interaction handling

See the `flexplot()` docstring (`help(flexplot)`) for the full parameter
reference, or `docs/api/core.md` for the API reference.