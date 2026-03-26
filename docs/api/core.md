# Core Visualization API

The core of `py-flexplot` is the `flexplot` function, which uses a formula-based syntax to decide how to best represent your data.

## `flexplot`

```python
from pyflexplot import flexplot
p = flexplot(formula, data, method="auto")
```

### Parameters
- **formula** (str): A formula of the form `y ~ x + color | panel`.
  - `y`: The outcome variable (y-axis).
  - `x`: The primary predictor (x-axis).
  - `color`: (Optional) Second predictor mapped to color and grouping.
  - `panel`: (Optional) Variables after `|` used for faceting (row/column panels).
- **data** (pd.DataFrame): The dataset to plot.
- **method** (str): The smoothing method. Options: `"auto"`, `"lm"` (linear model), `"loess"` (locally weighted regression).

### Intelligent Mapping
- **Numeric y ~ Numeric x**: Scatterplot + trend line.
- **Numeric y ~ Categorical x**: Jittered dot plot + bootstrapped means/CIs.
- **Categorical y ~ Numeric x**: Scatterplot + logistic regression curve.
- **Categorical y ~ Categorical x**: Jittered dot plot of counts.

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
*(Implementation in progress)*
