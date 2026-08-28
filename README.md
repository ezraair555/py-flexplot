# py-flexplot

A Python port of Dustin Fife's [`flexplot`](https://github.com/dustinfife/flexplot) and related R packages (`fifer`, `flexplavaan`, etc.).

`py-flexplot` provides intelligent data visualization using a formula-based syntax, similar to the original R implementation but powered by `plotnine` for a consistent "grammar of graphics" look and feel in Python.

![Titanic Example](docs/assets/titanic/plot2_sex.png)

## Included R Packages
- **flexplot**: Intelligent multivariate graphics via formulas.
- **fifer/fifer2**: Biostatistical toolbox for data cleanup and analysis.
- **flexplavaan**: Visualizing latent variable models (SEM).
- **flex_nn**: Neural-network visualization wrappers. **torch** is the default backend; **Keras 3** is supported transparently via the same `NeuralNetFit` class. Drop any `torch.nn.Module` or `keras.Model` (Sequential, Functional, or subclassed) into `compare_fits()` alongside statsmodels fits.
- **bluepill**: Synthetic mixed-model data generator. `mixed_model(...)` produces clustered data with fixed and random effects, interactions, and polynomial terms.

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

### Optional backends for `flex_nn`

- **torch** is the default and is required for the torch paths to run. `pip install torch`.
- **Keras 3** is supported opportunistically. Install `pip install "keras[jax]"` (or `keras[tensorflow]` / `keras[torch]`), set `KERAS_BACKEND=jax` (or your chosen backend), and `from pyflexplot.flex_nn import NeuralNetFit` will route Keras models through the same wrapper. No keras import is required when torch is the only backend.
- The `tests/test_flex_nn_keras.py` and the keras section of `examples/notebooks/flex_nn_example.ipynb` exercise the keras path; both skip cleanly when keras isn't installed.

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
- **Neural-Network Integration (torch + Keras 3)**: Wrap a fitted `torch.nn.Module` or `keras.Model` with `NeuralNetFit` to drop it into `compare_fits` next to a statsmodels fit. Keras 3 models are evaluated with `training=False` so Dropout/BatchNorm behave deterministically; torch models use `torch.no_grad()`.
- **Synthetic Data Generation**: `mixed_model(...)` produces clustered data with fixed + random effects for demos, teaching, and power analyses.
- **Biostats Utilities**: Ported functions from `fifer` for common statistical tasks.

## Changelog

### 0.2.0 (2026-08-28)
- Added `pyflexplot.flex_nn` — torch-default wrappers for fitting and visualizing neural networks. `NeuralNetFit` bundles a fitted `torch.nn.Module` (or `keras.Model`) with the metadata needed to plug into `compare_fits`. `permutation_importance()` provides column-shuffling variable importance.
- Added `pyflexplot.bluepill` — port of Dustin Fife's `bluepill` R package. `estimate_sd()` recovers an SD from a known mean and min/max range; `mixed_model()` generates clustered synthetic data with fixed + random effects, interactions, and polynomial terms.
- Dropped the aspirational `flexifiers` bullet from the "Included R Packages" list (no corresponding R package was found).
- Added 52 new tests across the two modules (60 → 112). Test suite uses `pytest.importorskip("torch")` so the package still imports cleanly without torch installed, but `flex_nn` tests skip when torch is absent.

### 0.2.1 (2026-08-28)
- Hardened the Keras 3 path in `pyflexplot.flex_nn`: predictions now go through a dedicated `_keras_predict()` helper that passes `training=False` (so `Dropout`/`BatchNorm` behave deterministically) and falls back gracefully for custom `Model` subclasses whose `predict()` doesn't accept the `training` kwarg. Added 14 keras-specific tests in `tests/test_flex_nn_keras.py` (skip when keras isn't installed; verified against `keras==3.15.1` + `jax` backend). Extended the example notebook with a Keras 3 walk-through and added a README section describing the optional install + backend selection.

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
