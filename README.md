# py-flexplot

A Python port of Dustin Fife's [`flexplot`](https://github.com/dustinfife/flexplot) and related R packages (`fifer`, `flexplavaan`, etc.).

`py-flexplot` provides intelligent data visualization using a formula-based syntax, similar to the original R implementation but powered by `plotnine` for a consistent "grammar of graphics" look and feel in Python.

![Titanic Example](docs/assets/titanic/plot2_sex.png)

## Included R Packages
- **flexplot**: Intelligent multivariate graphics via formulas.
- **fifer/fifer2**: Biostatistical toolbox for data cleanup and analysis.
- **flexplavaan**: Visualizing latent variable models (SEM).
- **flex_nn**: Neural network visualization (In progress).
- **flexifiers**: Data transformation utilities (In progress).

## Installation

> **PyPI status:** `py-flexplot` is not yet published on PyPI. Until it is released, install directly from the Git repository.

### Source install (recommended until PyPI is live)

```bash
pip install git+https://github.com/ezraair555/py-flexplot.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/ezraair555/py-flexplot.git
cd py-flexplot
pip install -e .
```

## Quick Start

```python
import pandas as pd
from pyflexplot import flexplot, visualize
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("data.csv")

# 1. Formula-based visualization
# y ~ x | z (y by x, faceted by z)
p = flexplot("y ~ x | z", data=df)
p.draw()

# 2. Model visualization
model = smf.ols("y ~ x", data=df).fit()
p_viz = visualize(model, data=df)
p_viz.draw()
```

## Features
- **Formula Syntax**: Uses `y ~ x + z | a` to automatically determine plot types.
- **Model Visualization**: Directly `visualize(model)` to see predicted vs actuals.
- **Model Comparison**: Use `compare_fits(formula, data, m1, m2)` to see performance side-by-side.
- **Biostats Utilities**: Ported functions from `fifer` for common statistical tasks.

## Changelog

### 0.1.1 (2026-06-20)
- Hardened `parse_flexplot_formula()` validation (exactly one `~`, at most one `|`, trimmed tokens, intercept-only handling, empty outcome/predictor rejection).
- Added input validation to `flexplot()` for empty DataFrames, missing columns, and numeric column types; color/group aesthetics are now included in the initial `aes()` so all geoms receive them.
- Fixed `hopper_plot()` against current `semopy` (`calc_sigma()` and `mx_cov` handling) and added real `semopy` smoke tests.
- Hardened `fit_beta_prior()` with range checks for `successes`/`totals`, zero-variance guard, optimizer success, and finite parameter validation.
- Fixed `added_plot()` residual alignment using index-aware `pd.concat(..., join='inner')` with length validation.
- Fixed `model_comparison()` LRT to enforce correct order and validate required model attributes.
- Fixed `visualize()` to identify the first non-intercept term robustly and raise exceptions instead of returning strings.
- Fixed `compare_fits()` prediction alignment to `data.index` with length validation.
- Fixed `add_ebb_estimate()` to use scalar/array addition and validate columns/dtypes.
- Fixed `sem.py` functions to raise typed exceptions instead of returning error strings.
- Expanded test coverage for all P0 and selected P1 paths.

### 0.1.0
- Initial package skeleton with `flexplot`, `visualize`, `compare_fits`, SEM helpers, and `fifer` utilities.

## License
MIT
