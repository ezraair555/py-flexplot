# Intelligent Multivariate Graphics • flexplot

## Description

`flexplot()` creates intelligent multivariate graphics via formulas. It automatically determines appropriate plot types based on variable types (numeric vs. categorical) and supports faceting by conditioning variables.

## Usage

```python
flexplot(formula: str, data: pd.DataFrame, method: str = "auto", **kwargs)
```

## Arguments

| Argument | Description |
|----------|-------------|
| `formula` | Formula of the form `outcome ~ predictor1 + predictor2 \| given1 + given2` |
| `data` | Pandas DataFrame containing the variables |
| `method` | Smoothing method: `"auto"`, `"lm"`, or `"loess"` |
| `**kwargs` | Additional arguments passed to plotnine |

## Formula Syntax

The formula syntax supports:

- **Outcome variable** (left side of `~`)
- **Predictor variables** (right side of `~`, separated by `+`)
- **Conditioning variables** (after `\|`, separated by `+`)

Examples:
- `"y ~ x"` - Simple scatter plot
- `"y ~ x + color_var"` - With color grouping
- `"y ~ x \| group"` - Faceted by one variable
- `"y ~ x \| group1 + group2"` - Faceted grid (group2 × group1)

## Details

`flexplot()` automatically selects the appropriate plot type based on variable types:

| Y Type | X Type | Plot Type |
|--------|--------|-----------|
| Numeric | Numeric | Scatter plot with regression line |
| Numeric | Categorical | Jitter plot with mean ± CI |
| Categorical | Numeric | Logistic regression smooth |
| Categorical | Categorical | Jitter plot with transparency |

## Returns

A `plotnine.ggplot` object that can be displayed with `.show()` or further customized.

## Examples

### Basic Scatter Plot

```python
import pandas as pd
from pyflexplot import flexplot

# Load data
df = pd.read_csv("mtcars.csv")

# Create scatter plot with regression line
p = flexplot("mpg ~ wt", data=df)
p.show()
```

### Multiple Predictors with Color

```python
# Add color by transmission type
p = flexplot("mpg ~ wt + am", data=df)
p.show()
```

### Faceted Plot

```python
# Facet by number of cylinders
p = flexplot("mpg ~ wt | cyl", data=df)
p.show()

# Facet grid by two variables
p = flexplot("mpg ~ wt | am + cyl", data=df)
p.show()
```

### Custom Smoothing Method

```python
# Use LOESS instead of linear regression
p = flexplot("mpg ~ wt", data=df, method="loess")
p.show()
```

## See Also

- [`visualize()`](visualize.html) - Visualize fitted models
- [`compare_fits()`](compare_fits.html) - Compare model fits visually
- [`added_plot()`](added_plot.html) - Added variable plots

## References

Ported from Dustin Fife's R `flexplot` package: https://cran.r-project.org/package=flexplot
