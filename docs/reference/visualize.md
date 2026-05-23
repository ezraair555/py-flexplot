# Model Visualization • visualize

## Description

`visualize()` provides a visual representation of a fitted statistical object. It supports statsmodels (OLS, GLM) and scikit-learn models.

## Usage

```python
visualize(model, data: Optional[pd.DataFrame] = None, **kwargs)
```

## Arguments

| Argument | Description |
|----------|-------------|
| `model` | Fitted statistical model (statsmodels or sklearn) |
| `data` | Optional DataFrame for visualization (extracted from model if not provided) |
| `**kwargs` | Additional arguments passed to plotnine |

## Details

`visualize()` automatically extracts the necessary information from fitted models to create diagnostic visualizations:

- **Statsmodels models**: Extracts prediction data and plots predicted vs. observed
- **Scikit-learn models**: Uses provided data to show model predictions
- **Automatic variable selection**: Uses the first predictor for X-axis

## Returns

A `plotnine.ggplot` object showing the model fit.

## Examples

### Statsmodels OLS

```python
import pandas as pd
import statsmodels.api as sm
from pyflexplot import visualize

# Load data
df = pd.read_csv("mtcars.csv")

# Fit model
model = sm.OLS(df['mpg'], sm.add_constant(df[['wt', 'hp']])).fit()

# Visualize
p = visualize(model, data=df)
p.show()
```

### Scikit-learn Regression

```python
from sklearn.linear_model import LinearRegression
from pyflexplot import visualize

# Fit model
X = df[['wt', 'hp']]
y = df['mpg']
sklearn_model = LinearRegression().fit(X, y)

# Visualize (data required for sklearn)
p = visualize(sklearn_model, data=df)
p.show()
```

## See Also

- [`flexplot()`](flexplot.html) - Formula-based graphics
- [`compare_fits()`](compare_fits.html) - Compare two models visually
- [`added_plot()`](added_plot.html) - Partial regression plots
