# Descriptive-Statistics Visualizations

This module is a port of R's `fifer::meansplot()`. It provides descriptive
visualizations for `y ~ group` formulas — useful for showing the central
tendency (and dispersion) of a numeric outcome across categorical or
ordinal groups.

## `meansplot`

Plot the mean of `y` per level of `x` with an error bar.

```python
from pyflexplot import meansplot

# Default: mean ± standard error, with a connecting line.
p = meansplot("weight ~ diet", data=df)

# Custom error type.
p = meansplot("weight ~ diet", data=df, error="sd")     # SD instead of SE
p = meansplot("weight ~ diet", data=df, error="ci")      # 95% CI on the mean
p = meansplot("weight ~ diet", data=df, error="range")   # min-max range
p = meansplot("weight ~ diet", data=df, error="iqr")     # Q1-Q3 IQR
p = meansplot("weight ~ diet", data=df, error="no")      # no error bar

# Suppress the connecting line.
p = meansplot("weight ~ diet", data=df, connect=False)
```

### Parameters
- **formula** (`str`): Formula of the form `y ~ group`. `group` may be a
  single categorical variable (string/object) or a numeric variable with
  few unique values (treated as discrete). Color terms and `|` facets
  are not supported and raise `ValueError`.
- **data** (`pd.DataFrame`): The dataset.
- **error** (`{"se", "sd", "ci", "range", "iqr", "no"}`, default `"se"`):
  Kind of error bar to draw around each mean:
  - `"se"`: standard error of the mean.
  - `"sd"`: standard deviation (ddof=1).
  - `"ci"`: `level` confidence interval on the mean (t-distribution).
  - `"range"`: min-max range.
  - `"iqr"`: Q1-Q3 IQR.
  - `"no"`: no error bar.
- **level** (`float`, default `0.95`): Coverage probability for `error="ci"`.
- **connect** (`bool`, default `True`): Draw a line connecting the
  per-group means. Useful for ordinal predictors.

### Returns
A `plotnine.ggplot`. The summary DataFrame (per-group mean, count, std,
and error-bar bounds) is exposed at `p.data` for downstream inspection.

### Example

```python
import pandas as pd
import numpy as np
from pyflexplot import meansplot

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "weight": rng.normal(size=120),
    "diet": rng.choice(["A", "B", "C"], size=120),
})

p = meansplot("weight ~ diet", data=df)
print(p.data)
#   diet  count      mean       std    __lower    __upper
# 0    A     40  ...       ...       ...        ...
# 1    B     40  ...       ...       ...        ...
# 2    C     40  ...       ...       ...        ...
```

### Differences from R's `fifer::meansplot()`

- The Python port requires a 2-term formula (`y ~ group`); R accepts
  more complex formulas with `+ color` or `| given` that the Python
  port rejects explicitly.
- The connecting line is always drawn with `linetype="dashed"` and
  `color="gray"` in the Python port; R's `meansplot()` allows finer
  control over the connector's appearance.
- Numeric predictors are coerced to discrete levels (string) for
  consistent rendering. R's `meansplot()` keeps numeric predictors as
  numeric.

These differences are intentional: the Python port keeps the surface
small and explicit, and delegates richer customization to plotnine's
standard layer composition.