# Uncertainty Module (v0.4.0+)

Helpers used internally by `flexplot()` for confidence intervals, prediction
intervals, and case-resampled bootstrap CIs. Also usable directly for custom
plotting or downstream analysis.

## Public API

```python
from pyflexplot.uncertainty import (
    VALID_UNCERTAINTY,
    validate_uncertainty_params,
    compute_bootstrap_ci,
    compute_prediction_band,
    format_band_label,
)
```

---

## `VALID_UNCERTAINTY`

```python
VALID_UNCERTAINTY = frozenset({None, "ci", "prediction", "bootstrap"})
```

The set of valid `uncertainty=` values accepted by `flexplot()`. `None`
disables the fit (scatter only); the other three correspond to band types.

---

## `validate_uncertainty_params`

```python
validate_uncertainty_params(uncertainty, level, bands, method)
```

Validate the uncertainty-related parameters and their method compatibility.
Raises `ValueError` on the first violation. Used at the top of
`flexplot()` to fail fast on bad inputs.

### Parameters

- **uncertainty**: one of `VALID_UNCERTAINTY`. `bootstrap` requires `method="loess"` or `"auto"`; passing it with `method="lm"` raises.
- **level**: float in `(0, 1)`. `0`, `1`, or out-of-range values raise.
- **bands**: list of floats in `(0, 1)`, or `None`. Each entry must satisfy the same bounds as `level`.
- **method**: `"auto"`, `"lm"`, or `"loess"`. Combined with `uncertainty` to enforce compatibility (e.g., `"bootstrap"` requires a non-parametric smoother).

### Raises

- `ValueError` on any invalid input, with a precise message.

---

## `compute_bootstrap_ci`

```python
compute_bootstrap_ci(x, y, smooth_fn, n_resamples=200, level=0.95, random_state=None)
```

Case-resampled bootstrap confidence interval for a smoother.

### Parameters

- **x, y** (1-D arrays): equal-length input data.
- **smooth_fn** (callable): `smooth_fn(x_eval, y_at_x_eval_sorted_by_x) -> yhat`. Fitted values evaluated at `x_eval`. Called once on the full data plus `n_resamples` times on bootstrap samples.
- **n_resamples** (int, default 200): number of bootstrap samples.
- **level** (float, default 0.95): coverage probability in `(0, 1)`.
- **x_eval** (1-D array, optional): evaluation grid; defaults to sorted unique `x` values.
- **random_state** (int, optional): seed for reproducibility.

### Returns

- `(x_eval, lower, upper)`: three 1-D arrays of equal length. Percentile-based
  CI bounds on the bootstrap distribution of fitted values at each `x_eval`.

### Design notes

- Uses case (row) resampling rather than residual resampling to be robust to
  model misspecification.
- Failed bootstrap samples (e.g., singular fits) fall back to the full-data
  fit so percentile bounds are still computable.

### Example

```python
import numpy as np
from pyflexplot.uncertainty import compute_bootstrap_ci
from statsmodels.nonparametric.smoothers_lowess import lowess

rng = np.random.default_rng(0)
x = rng.normal(size=100)
y = x ** 2 + rng.normal(scale=0.3, size=100)
x_eval = np.linspace(-2, 2, 50)

def loess_at(x_eval, y_sorted):
    return lowess(y_sorted, np.sort(x), return_sorted=False)

x_out, lo, hi = compute_bootstrap_ci(
    x, y, loess_at, n_resamples=100, level=0.95, random_state=0,
)
```

---

## `compute_prediction_band`

```python
compute_prediction_band(y_true, y_pred, level=0.95)
```

Symmetric residual-based prediction interval. Assumes approximately normal
residuals with constant variance. Uses residual standard error with `ddof=0`
(matches OLS residual-variance convention).

### Parameters

- **y_true, y_pred** (1-D arrays): must have the same shape.
- **level** (float, default 0.95): coverage probability.

### Returns

- `(lower, upper)`: 1-D arrays of the same length as `y_pred`, where
  `lower = y_pred - z * sigma` and `upper = y_pred + z * sigma`, with `z`
  the standard-normal critical value.

### Example

```python
import numpy as np
from pyflexplot.uncertainty import compute_prediction_band

rng = np.random.default_rng(0)
y_true = rng.normal(scale=2.0, size=200)
y_pred = np.zeros(200)
lower, upper = compute_prediction_band(y_true, y_pred, level=0.95)
# half-width is z * sigma ≈ 1.96 * 2.0 ≈ 3.92
```

---

## `format_band_label`

```python
format_band_label(level, kind="ci")
```

Format a legend label for a band.

### Parameters

- **level** (float): the level coverage probability.
- **kind** (str, default `"ci"`): one of `"ci"`, `"prediction"`, `"bootstrap"`, or any other string.

### Returns

- `str`: e.g., `"95% CI"`, `"80% PI"`, `"95% bootstrap CI"`.

### Example

```python
>>> from pyflexplot.uncertainty import format_band_label
>>> format_band_label(0.95)
'95% CI'
>>> format_band_label(0.80, kind="prediction")
'80% PI'
>>> format_band_label(0.95, kind="bootstrap")
'95% bootstrap CI'
```

---

## When to use which band

| Use case | Band type | Why |
|---|---|---|
| "What's the uncertainty in the *mean* response?" | `"ci"` | Standard, well-calibrated for LM; plotnine built-in for loess. |
| "Where will new observations fall?" | `"prediction"` | Wider than CI; reflects both parameter uncertainty AND residual variance. |
| Loess smoother with no theoretical CI | `"bootstrap"` | Case-resampling is the honest path when the smoother has no closed-form SE. |
| "Compare fits visually" | overlay + bands | Let the user see which fit the data prefers. |

See `docs/api/core.md` for the `flexplot()` parameter reference and
`docs/examples/diagnostics_workflow.md` for the combined workflow.