# SEM Visualization API (flexplavaan Port)

The `sem` module brings the visualization strategies of `flexplavaan` to Python, integrating with the `semopy` package.

## `hopper_plot`

Visualize the residuals from the variance/covariance matrix. This helps identify where the model fails to capture the relationships between variables.

```python
from pyflexplot import hopper_plot
p = hopper_plot(model)
```

- **model**: A fitted `semopy.Model` object.
- **Visual**: A heatmap of residuals (Observed - Implied). Blue indicates underestimation, Red indicates overestimation by the model.

---

## `disturbance_plot`

Visualize the association between two observed variables after the model-implied fit has been removed.

```python
from pyflexplot import disturbance_plot
p = disturbance_plot(model, var1="x1", var2="x2", data=df)
```

- **Visual**: A scatterplot of residuals with a blue LOESS line and a red dashed line at y=0. If the LOESS line deviates significantly from zero, it suggests an unmodeled association (e.g., a missing path or correlated error).

---

## `measurement_plot`

Visualize the relationship between a latent variable and one of its indicators.

```python
from pyflexplot import measurement_plot
p = measurement_plot(model, latent_var="F1", indicator="x1", data=df)
```

- **latent_var**: The name of the latent factor.
- **indicator**: The name of the observed variable.
- **Visual**: A scatterplot of the predicted factor scores against the raw indicator values, including a linear regression line.
