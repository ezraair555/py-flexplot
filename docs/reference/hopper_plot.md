# Hopper Plot • hopper_plot

## Description

`hopper_plot()` visualizes residuals from the variance/covariance matrix in structural equation modeling (SEM). It shows the discrepancy between observed and model-implied correlations.

## Usage

```python
hopper_plot(model, **kwargs)
```

## Arguments

| Argument | Description |
|----------|-------------|
| `model` | Fitted SEM model (from semopy) |
| `**kwargs` | Additional arguments passed to plotnine |

## Details

The hopper plot displays the residual covariance matrix (observed - implied) as a heatmap. Large residuals indicate areas where the model does not fit the data well.

**Interpretation:**
- **Red cells**: Observed correlation > Implied correlation
- **Blue cells**: Observed correlation < Implied correlation
- **White cells**: Good fit (residual ≈ 0)

## Returns

A `plotnine.ggplot` heatmap object.

## Examples

### SEM Hopper Plot

```python
import pandas as pd
from semopy import Model
from pyflexplot import hopper_plot

# Define SEM model
desc = """
    y1 =~ x1 + x2 + x3
    y2 =~ x4 + x5 + x6
    y1 ~ y2
"""

# Fit model
model = Model(desc)
model.fit(data)

# Create hopper plot
p = hopper_plot(model)
p.show()
```

## See Also

- [`disturbance_plot()`](disturbance_plot.html) - Disturbance dependence plots
- [`measurement_plot()`](measurement_plot.html) - Measurement model plots

## References

Ported from Dustin Fife's R `flexplavaan` package
