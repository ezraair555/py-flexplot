# Biostatistical Utilities (Fifer Port)

These utilities are ported from the `fifer` and `fifer2` R packages, providing standard biostatistical tools for Python.

## `model_comparison`

Compare two statistical models (usually OLS or GLM) and report comparative fit statistics.

```python
from pyflexplot import model_comparison
stats_df, p_value = model_comparison(model1, model2)
```

### Returns
- **stats_df**: A DataFrame containing AIC, BIC, and Log-Likelihood for both models.
- **p_value**: If the models are nested, a Likelihood Ratio Test (LRT) p-value indicating if the more complex model is significantly better.

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
