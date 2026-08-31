# Core Visualization API

The core of `py-flexplot` is the `flexplot` function, which uses a formula-based syntax to decide how to best represent your data.

## `flexplot`

```python
from pyflexplot import flexplot
p = flexplot(formula, data, method="auto", uncertainty="ci", level=0.95)
```

### Parameters
- **formula** (str): A formula of the form `y ~ x + color | panel`.
  - `y`: The outcome variable (y-axis).
  - `x`: The primary predictor (x-axis).
  - `color`: (Optional) Second predictor mapped to color and grouping.
  - `panel`: (Optional) Variables after `|` used for faceting (row/column panels).
  - **R-style interaction syntax** (v0.6.2+): `y ~ x*z` expands to `y ~ x + z + x:z` for column lookup; `y ~ x:z` is also accepted. Numeric binary `[0, 1]` y routes to the binomial GLM branch (v0.6.1+). See [Interactions](#interactions) below.
- **data** (pd.DataFrame): The dataset to plot.
- **method** (str): The smoothing method for the numeric-vs-numeric branch. Options: `"auto"`, `"lm"` (linear model), `"loess"` (locally weighted regression), `"polynomial"` / `"cubic"` (degree-3 OLS in x, v0.6.4+), `"logistic"` (GLM with logit link on numeric binary y, v0.6.4+).
- **uncertainty** (str, v0.4.0+): `{None, "ci", "prediction", "bootstrap"}`, default `"ci"`. Type of uncertainty band around the fitted line.
  - `None`: no fit, just the scatter.
  - `"ci"`: confidence interval on the mean response (plotnine built-in).
  - `"prediction"`: residual-based prediction interval on new observations (LM only).
  - `"bootstrap"`: case-resampled CI (loess only; `n_resamples=200`).
- **level** (float, v0.4.0+): Coverage probability for a single band, default `0.95`. Ignored when `bands` is given.
- **bands** (list of float, v0.4.0+): Nested coverage levels (e.g., `[0.5, 0.8, 0.95]`) for Tufte-style multi-ribbon display. Overrides `level` when provided.
- **overlay** (list, v0.5.0+): Additional smoothers to overlay on the same axes. Each entry is a method name (`"lm"`, `"loess"`, `"rlm"`, ...) or a dict with `method`, `color`, `label`, `uncertainty`, `level`. See [Overlay](#overlay) below.
- **bins** (int, v0.6.4+): Discretize a numeric x into `bins` equal-width intervals before plotting. Routes to the discrete-style summary branch (geom_jitter + spread marker). Mutually exclusive with `breaks` (which wins with a `UserWarning`).
- **labels** (list of str, v0.6.4+): Custom labels for the discrete x levels produced by `bins` / `breaks`. Length must equal `bins` or `len(breaks) - 1`.
- **breaks** (list of float, v0.6.4+): Explicit cut points for numeric-x binning. Takes precedence over `bins` when both are given.
- **spread** ({None, "ci", "stdev", "range", "iqr", "no"}, v0.6.4+): Dispersion marker for the discrete-x branch. `None` (default) preserves the legacy bootstrap CI; `"stdev"` / `"range"` / `"iqr"` use a pointrange; `"no"` omits the summary.
- **sample** (int, v0.6.5+): Subsample N rows for the plotnine layers; smoother fits still see the full DataFrame. No-op when `N >= len(data)`. Deterministic via `np.random.default_rng(0)`.
- **ghost_line** ({None, "red", "dashed"}, v0.6.5+): Reference `geom_hline` at y=0. `"red"` for a solid red threshold; `"dashed"` for a black dashed reference.
- **plot_type** ({None, "scatter", "line", "boxplot", "bar"}, v0.6.5+): Explicit geom override. Bypasses the auto-dispatch.
- **return_data** (bool, v0.6.5+): When `True`, returns `{"plot": ggplot, "data": DataFrame}` instead of just the plot.
- **ghost_reference** (pd.DataFrame, v0.6.6+): Reference dataset to overlay. Two patterns detected by column shape: `(x, y)` → gray geom_point (reference scatter); `(x, "pred")` → red dashed geom_line (prediction line).
- **plot_string** (dict, v0.6.6+): Override axis/legend labels. Accepts keys `x`, `y`, `title`, `subtitle`, `caption`, `color`.
- **related** (bool, v0.6.6+): R-flexplot's panel-linking flag. Currently a no-op on the Python side (plotnine already shares scales by default); accepted for R-parity, rejected if non-bool.

### Intelligent Mapping
- **Numeric y ~ Numeric x**: Scatterplot + trend line. Smoother controlled by `method` (`"auto"` / `"lm"` / `"loess"` / `"polynomial"` / `"cubic"` / `"logistic"`).
- **Numeric y ~ Categorical x**: Jittered dot plot + dispersion marker (bootstrap CI by default; `spread` controls the marker type).
- **Numeric binary `[0, 1]` y ~ Numeric x** (v0.6.1+): Scatterplot + binomial GLM (logistic curve) — even when y is `int`/`float`, the binary pre-check routes to the binomial branch. Explicit `method="logistic"` bypasses this and uses the parametric branch.
- **Non-numeric y ~ Numeric x** (e.g., string `["yes","no"]`): Scatterplot + binomial GLM.
- **Categorical y ~ Categorical x**: Jittered dot plot of counts.
- **Auto-binned numeric x** (v0.6.4+): When `bins` or `breaks` is given, numeric x is discretized via `pd.cut` and the discrete-x branch applies.

### Examples

```python
import pandas as pd
import numpy as np
from pyflexplot import flexplot

rng = np.random.default_rng(0)
df = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})

# Basic numeric-vs-numeric plot with default uncertainty.
flexplot("y ~ x", data=df)

# Multiple nested uncertainty bands (Tufte-style).
flexplot("y ~ x", data=df, bands=[0.5, 0.8, 0.95])

# Overlay multiple smoothers on the same chart.
flexplot(
    "y ~ x", data=df,
    overlay=[
        {"method": "loess", "label": "LOESS smoother"},
        {"method": "rlm",   "label": "Robust regression"},
    ],
)

# Disable the fit (scatter only).
flexplot("y ~ x", data=df, uncertainty=None)

# Polynomial fit on a non-linear signal.
flexplot(
    "y ~ x", data=df.assign(y=lambda d: d["x"] ** 2 + rng.normal(scale=0.3, size=200)),
    method="polynomial",
)

# Auto-bin a numeric predictor into 4 equal-width groups (v0.6.4+).
df2 = pd.DataFrame({
    "x": rng.uniform(0, 100, size=200),
    "y": rng.normal(size=200),
})
flexplot("y ~ x", data=df2, bins=4)

# Use stdev as the dispersion marker (v0.6.4+).
flexplot("y ~ x", data=df2, bins=4, spread="stdev")

# Subsample a large dataset for rendering (v0.6.5+).
flexplot("y ~ x", data=df2, sample=50)

# Get both the plot and the data (v0.6.5+).
result = flexplot("y ~ x", data=df, return_data=True)
result["plot"].save("plot.png")
print(result["data"].head())

# Overlay a reference dataset (v0.6.6+).
ref = pd.DataFrame({"x": np.linspace(-3, 3, 20), "pred": np.linspace(-3, 3, 20) ** 2})
flexplot("y ~ x", data=df, ghost_reference=ref)

# Override labels (v0.6.6+).
flexplot(
    "y ~ x", data=df,
    plot_string={"x": "Predictor (s)", "y": "Response (V)", "title": "Experiment 1"},
)
```

### <a name="overlay"></a>Overlay

The `overlay` parameter adds additional smoothers on top of the primary
`method`. Useful for visually comparing fits side-by-side (e.g., LM vs LOESS
vs robust regression):

```python
flexplot(
    "y ~ x", data=df,
    overlay=[
        "loess",                       # default color, default ci
        {"method": "rlm", "color": "#ff7f0e", "label": "Robust"},
        {"method": "lm",  "level": 0.80, "label": "Linear (80%)"},
    ],
)
```

If any overlay entry has an explicit `label`, a manual color scale is added
so the legend distinguishes the primary smoother from each overlay. The
binomial branch restricts overlay to `method="glm"` (other methods raise).

### <a name="interactions"></a>Interactions (v0.6.2+)

Formulas accept R-style interaction syntax:

| Syntax | Parsed as |
|--------|-----------|
| `y ~ x + z` | additive model, `x` and `z` as separate predictors |
| `y ~ x*z` | expanded to `x + z + x:z`; column lookup uses `x` and `z` |
| `y ~ x:z` | interaction term only (no main effects); first atom `x` used as the x-axis variable |

The v0.6.x fit remains **additive** (parallel slopes per color group). A
`UserWarning` is emitted whenever `*` or `:` appears in the formula so users
aren't misled. v0.7.0 will add `interaction_model=True` for true non-parallel
slopes.

---

## `visualize` (v0.6.3+)

Visualize a fitted model's predictions and residuals.

```python
from pyflexplot import visualize

# Predicted-vs-observed scatter (default).
result = visualize(formula, data, model)
# {"plot": ggplot, ...}

# Residual plots: returns a dict of two ggplots.
residuals = visualize(formula, data, model, plot="residuals")
# {"rvf": ggplot, "hist": ggplot}  (residual-vs-fitted + histogram)

# All-in-one layout (cowplot if available, else dict).
all_plots = visualize(formula, data, model, plot="all")
```

| `plot=` | Returns | Notes |
|---|---|---|
| `"model"` (default) | `{"plot": ggplot}` | Predicted-vs-observed scatter + fitted line. |
| `"residuals"` | `{"rvf": ggplot, "hist": ggplot}` | Residual-vs-fitted (geom_hline at y=0) + residual histogram. |
| `"all"` | `cowplot`-joined 2-column layout, or dict `{"model", "rvf", "hist"}` if cowplot isn't installed | Always useful even when cowplot is missing. |

Raises `ValueError` for unknown `plot=` values.

---

## `added_plot`

Generates an **Added Variable Plot** (Partial Regression Plot) to visualize the unique relationship between Y and X after controlling for other variables in the formula.

```python
from pyflexplot import added_plot
p = added_plot("y ~ x1 + x2", data=df)
```

- If the formula has multiple predictors, `added_plot` calculates the residuals and plots the "clean" relationship for the first predictor.

---

## `compare_fits`

Visually compare how well two different models fit the data by overlaying their prediction lines.

```python
from pyflexplot import compare_fits
p = compare_fits(formula, data, model1, model2)
```

For comparing multiple *smoothers* (not pre-fit models), use `flexplot(..., overlay=[...])` instead — see the Overlay section above.

---

## `diagnose` (v0.6.0+)

```python
from pyflexplot import diagnose
diag = diagnose("y ~ x + z", data=df)        # verbose=True prints to stdout
diag = diagnose("y ~ x + z", data=df, verbose=False)
```

Run a data-quality diagnostic on a flexplot formula + data. Surfaces:

- **Missingness**: per-column counts and pattern heuristic (`none` /
  `concentrated (likely MNAR/MAR)` / `spread (likely MCAR)`).
- **Outliers**: Cook's distance count and threshold (default `4/n`).
- **Linearity**: Ramsey RESET test for functional-form misspecification.
- **Heteroscedasticity**: Breusch-Pagan test for non-constant variance.

Returns a structured `dict`; pass `verbose=True` for a one-paragraph
terminal/email/log summary. See `pyflexplot.quality.diagnose` for details.

### Example

```python
from pyflexplot import diagnose, format_summary

df = pd.DataFrame({"y": rng.normal(size=200), "x": rng.normal(size=200)})
diag = diagnose("y ~ x", data=df, verbose=False)
print(format_summary(diag))
```

---

## Uncertainty helpers (v0.4.0+)

```python
from pyflexplot.uncertainty import (
    compute_bootstrap_ci,
    compute_prediction_band,
    format_band_label,
    validate_uncertainty_params,
    VALID_UNCERTAINTY,
)
```

Public helpers used internally by `flexplot()` but also usable directly.