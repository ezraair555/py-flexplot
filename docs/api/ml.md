# Machine-Learning Adapters (sklearn integration)

This module provides thin adapters that make scikit-learn estimators
(``RandomForestRegressor``, ``RandomForestClassifier``, ``GradientBoosting*``,
and any estimator with a ``.predict()`` method) usable with py-flexplot's
visualization API: ``compare_fits()``, ``flexplot(overlay=...)``,
``visualize()``, and ``estimates()``.

> **Optional dependency:** scikit-learn is **not** declared in
> py-flexplot's ``pyproject.toml`` — install it separately
> (``pip install scikit-learn``) when you want to use this module.

## Why an adapter?

scikit-learn estimators expose ``.predict(X)`` but not the predictor
names; py-flexplot's visualization layer needs to know which columns
the model used so it can build evaluation DataFrames without losing
row alignment. ``RFAdapter`` carries that metadata alongside the
fitted model.

This is intentionally a thin glue layer — R-flexplot's
``flexplot::compare.fits()`` accepted any R model with a ``predict()``
method, and that's the surface this module recreates in Python.

## `RFAdapter`

```python
from sklearn.ensemble import RandomForestRegressor
from pyflexplot.ml import RFAdapter

rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X, y)
fit = RFAdapter(rf, response_var="y", predictor_names=list(X.columns))
```

### Parameters
- **estimator** (`sklearn.base.BaseEstimator`): A fitted scikit-learn
  estimator with a `.predict(X)` method. `RandomForestRegressor` /
  `RandomForestClassifier` are the primary targets, but any regressor
  or classifier works.
- **response_var** (`str`): Name of the response variable in the
  original DataFrame. Used by py-flexplot to build evaluation DataFrames.
- **predictor_names** (`list[str]`): Column names of the predictors
  used during fitting. Order matters; must match the column order in
  the X matrix passed to `.fit()`.

### Methods

- **`predict(X)`** — Accepts a `pandas.DataFrame` (column-aligned by
  name when possible) or a 2-D `numpy.ndarray`. Returns a 1-D
  `numpy.ndarray`.
- **`predict_df(data)`** — Predicts on a DataFrame and returns a
  single-column DataFrame. The column is named `pred_<response_var>`
  to match the convention used by `compare_fits(return_preds=True)`.

### Example: use RFAdapter in `compare_fits`

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from pyflexplot import compare_fits
from pyflexplot.ml import make_rf_adapter

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "x1": rng.normal(size=200),
    "x2": rng.normal(size=200),
    "y":  rng.normal(size=200) + 0.5 * rng.normal(size=200),
})

# Fit a statsmodels OLS and a RandomForest on the same data.
ols = smf.ols("y ~ x1 + x2", data=df).fit()
rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(
    df[["x1", "x2"]], df["y"],
)
adapter = make_rf_adapter(rf, data=df, response_var="y")

# Overlay both fits.
p = compare_fits("y ~ x1 + x2", data=df, model1=ols, model2=adapter)
```

## `make_rf_adapter`

Convenience constructor that pulls predictor names from a DataFrame
when not provided explicitly:

```python
adapter = make_rf_adapter(rf, data=df, response_var="y")
# Equivalent to:
# RFAdapter(rf, response_var="y", predictor_names=[c for c in df.columns if c != "y"])
```

## Limitations and out-of-scope items

- **Mixed-effects / hierarchical models** for `flexplot()` now exist in the
  core API (`method="mixedlm"|"lmer"|"glmer"` + `random_effects=`), but they
  are still outside this **ml adapter** module's scope. `RFAdapter` is for
  sklearn-style estimators with `.predict()`, not mixed-model fitting.
- **`estimates()` does not work on RFAdapter** — it expects a
  statsmodels model. Use `model_comparison()` to compare a statsmodels
  fit against an RFAdapter, but don't expect `.rsquared` or
  `.params` on the latter.
- **Feature importances** are exposed via `adapter.estimator.feature_importances_`
  (raw sklearn accessor); py-flexplot doesn't wrap these yet. A future
  version may add a `feature_importance_plot()` helper.
- **Partial-dependence plots** — a common companion to random forests.
  Out of scope; sklearn's `sklearn.inspection.partial_dependence` is the
  standard reference.

## What this is NOT

This module is **not** a modeling layer. It does not fit, tune, or
cross-validate models. For those, use scikit-learn directly:

- [`sklearn.ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [`sklearn.model_selection.cross_val_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
- [`sklearn.inspection.partial_dependence`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html)

py-flexplot's role here is purely visualization glue.
