# py-flexplot

A **partial** Python port of Dustin Fife's [`flexplot`](https://github.com/dustinfife/flexplot) and related R packages (`fifer`, `flexplavaan`, `ebbr`, `bluepill`).

`py-flexplot` provides intelligent data visualization using a formula-based syntax, similar to the original R implementation but powered by `plotnine` for a consistent "grammar of graphics" look and feel in Python.

![Titanic Example](docs/assets/titanic/plot2_sex.png)

## What's covered (and what isn't)

This is **not** a 1:1 port. The Python port covers the parts of R's `flexplot` and friends that translate cleanly onto `plotnine` + `statsmodels`; some R-only features are deferred or unsupported. See [`docs/api/coverage.md`](docs/api/coverage.md) for the full coverage matrix vs the R packages. Highlights:

- ✅ `flexplot()` core dispatch + `bins` / `breaks` / `labels` (auto-bin), `spread`, `overlay`, `uncertainty` (CI / prediction / bootstrap), `ghost_line` / `ghost_reference`, `plot.string`, `plot_type` override, `sample`, `return_data`.
- ✅ `model_comparison()` (AIC / BIC / R² / adj.R² / **Bayes factor**), `estimates()` (structured effect-size reporter), `compare_fits()` (with `return_preds` / `pred_type`).
- ✅ `visualize()` with `plot='model' | 'residuals' | 'all'`.
- ✅ `diagnose()` (missingness, Cook's D, Ramsey RESET, Breusch-Pagan).
- ⚠️ R-style interaction syntax (`y ~ x*z`) is parsed but the fit remains additive — pass `interaction_model=True` (v0.7.0+) for non-parallel slopes per color group.
- ✅ `randomForest` (and any sklearn estimator with `.predict()`) — use `pyflexplot.ml.RFAdapter` to wrap a fitted estimator and pass it to `compare_fits()`. See [`docs/api/ml.md`](docs/api/ml.md).
- ❌ Mixed-effects models (`lme4` / `glmer`) are not ported; `statsmodels.MixedLM` is not a drop-in for `lme4`. See [`docs/api/coverage.md`](docs/api/coverage.md) for the plan.

## Included R Packages
- **flexplot**: Intelligent multivariate graphics via formulas.
- **fifer/fifer2**: Biostatistical toolbox for data cleanup and analysis.
- **flexplavaan**: Visualizing latent variable models (SEM).
- **flex_nn**: Neural-network visualization wrappers. **torch** is the default backend; **Keras 3** is supported transparently via the same `NeuralNetFit` class. Drop any `torch.nn.Module` or `keras.Model` (Sequential, Functional, or subclassed) into `compare_fits()` alongside statsmodels fits.
- **bluepill**: Synthetic mixed-model data generator. `mixed_model(...)` produces clustered data with fixed and random effects, interactions, and polynomial terms.
- **descriptives** (Python-native, port of `fifer::meansplot()`): `meansplot(formula, data, error=...)` for mean + error-bar visualizations across categorical or ordinal groups.
- **ml** (Python-native, no R analog): Adapters so scikit-learn estimators (`RandomForestRegressor`, `RandomForestClassifier`, and any estimator with `.predict()`) can be used with `compare_fits()`. Optional — requires `pip install scikit-learn`.

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
from pyflexplot import flexplot, visualize, compare_fits
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

# 3. Compare two models side-by-side (statsmodels or scikit-learn)
p_cmp = compare_fits("y ~ x", data=df, model1=model, model2=model)

# 4. Drop a fitted neural network into compare_fits
from pyflexplot.flex_nn import NeuralNetFit, set_response_var
import torch

torch_model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU(),
                                  torch.nn.Linear(8, 1)).eval()
set_response_var(torch_model, "y")
nn_fit = NeuralNetFit(model=torch_model, response_var="y",
                      predictor_names=["x1", "x2", "x3"])
p_nn = compare_fits("y ~ x1", data=df, model1=model, model2=nn_fit)

# 5. Generate a synthetic clustered dataset for demos or power analyses
from pyflexplot import mixed_model

df_sim = mixed_model(
    fixed=[0.0, 0.2, 0.5, 0.3, 0.2],
    random=[0.1, 0.1, 0.0, 0.2, 0.1],
    sigma=0.3, clusters=15, n_per=[11, 3],
    vars={"depression": (10.0, 3.0, 0),
          "stress":     (22.0, 7.0, 0),
          "life_events": ["no", "yes"],
          "ses":        (55.0, 15.0, 0),
          "therapist":  [f"Dr. {chr(65 + i)}" for i in range(15)]},
    seed=42,
)
```

See `examples/notebooks/flex_nn_example.ipynb` for an end-to-end
walk-through of the new functionality.

## Features
- **Formula Syntax**: Uses `y ~ x + z | a` to automatically determine plot types.
- **Model Visualization**: Directly `visualize(model)` to see predicted vs actuals.
- **Model Comparison**: Use `compare_fits(formula, data, m1, m2)` to see performance side-by-side.
- **Uncertainty Layers (v0.4.0+)**: First-class confidence / prediction / bootstrap bands around every fitted line via `uncertainty=`, `level=`, and `bands=` on `flexplot()`. Pick the band type that fits your modeling claim.
- **Model-Compare Overlay (v0.5.0+)**: Overlay multiple smoothers (`lm`, `loess`, `rlm`, etc.) on the same chart via `overlay=...` so the user can *see* which fit the data prefers.
- **Auto Data-Quality Diagnostics (v0.6.0+)**: `diagnose("y ~ x + z", data)` runs missingness / Cook's distance / Ramsey RESET / Breusch-Pagan and prints a one-paragraph summary of why your fit might be off.
- **R-Style Interaction Syntax (v0.6.2+)**: Formulas accept `y ~ x*z` and `y ~ x:z` (parsed, validated, with a `UserWarning` noting that the v0.6.x fit is additive; v0.7.0 will add `interaction_model=True`).
- **Neural-Network Integration (torch + Keras 3)**: Wrap a fitted `torch.nn.Module` or `keras.Model` with `NeuralNetFit` to drop it into `compare_fits` next to a statsmodels fit. Keras 3 models are evaluated with `training=False` so Dropout/BatchNorm behave deterministically; torch models use `torch.no_grad()`. `permutation_importance()` provides column-shuffling variable ranking that works against either backend.
- **Synthetic Data Generation**: `mixed_model(...)` produces clustered data with fixed + random effects for demos, teaching, and power analyses. `estimate_sd(mean, min, max)` recovers an SD from a known range.
- **Biostats Utilities**: Ported functions from `fifer` for common statistical tasks.

## Typical workflow (v0.6.x)

```python
import pandas as pd
from pyflexplot import flexplot, diagnose

df = pd.read_csv("data.csv")

# 1. Diagnose the model fit before plotting.
diag = diagnose("y ~ x + z", data=df)

# 2. Plot with uncertainty bands and overlay competing smoothers.
p = flexplot(
    "y ~ x + z", data=df,
    uncertainty="ci",        # or "prediction" / "bootstrap"
    level=0.95,
    bands=[0.5, 0.8, 0.95],  # nested ribbons (Tufte-style)
    overlay=[
        {"method": "loess", "label": "LOESS smoother"},
        {"method": "rlm",   "label": "Robust regression"},
    ],
)
p.draw()
```

See `docs/examples/diagnostics_workflow.md` for a longer walk-through.

## Continuous Integration

Three GitHub Actions workflows cover the test surface, kept independent so
each runs in its own clean environment:

* `.github/workflows/python-app.yml` -- core test matrix across Python
  3.10, 3.11, 3.12, 3.13. No torch or keras required; tests that need
  them skip cleanly via `pytest.importorskip`.
* `.github/workflows/torch.yml` -- installs `torch` (CPU build) and runs
  the torch-flex_nn tests. Triggered on every push to `main`, on PRs
  touching `src/pyflexplot/flex_nn.py` or the torch tests, and on a
  weekly schedule so we catch upstream torch regressions.
* `.github/workflows/keras3.yml` -- installs `keras[jax]` and runs
  `tests/test_flex_nn_keras.py` against a Keras 3 install. Same
  trigger pattern as `torch.yml` plus a weekly schedule.

All workflows upload coverage via `pytest-cov`.

## Changelog

### 0.6.2 (2026-08-30)
- **R-style interaction syntax accepted by the formula parser.** `y ~ x*z` and `y ~ x:z` no longer raise "missing column"; the parser expands `*` to `+` + `:` for column lookup and preserves interaction terms in `all_x`. `flexplot()` emits a `UserWarning` reminding the user that v0.6.x fits remain additive; v0.7.0 will add `interaction_model=True`. 6 new tests in `tests/test_core.py`.

### 0.6.1 (2026-08-30)
- **Fixed dead binomial branch in `flexplot()`.** `pd.api.types.is_numeric_dtype([0, 1])` returns True, so int/float binary y was always routed to the LM/loess branch and the binomial GLM branch was unreachable. Added a binary pre-check that detects unique values ⊆ {0, 1} *before* the numeric-dtype dispatch. Numeric `[0, 1]` y now draws a sigmoid curve (was a straight LM line); string `["yes", "no"]` and multi-level numeric `[0, 1, 2]` behavior unchanged. 3 new tests + 1 updated regression test.

### 0.6.0 (2026-08-30)
- **`diagnose(formula, data)` — auto data-quality diagnostics.** Runs missingness (per-column counts and pattern heuristic), Cook's distance for outliers (default `4/n`), Ramsey RESET for functional form, and Breusch-Pagan for heteroscedasticity. Returns a structured dict; pass `verbose=True` for a one-paragraph terminal/email/log summary. New module `pyflexplot.quality`. 19 new tests.

### 0.5.0 (2026-08-30)
- **`overlay` parameter on `flexplot()`.** Overlay multiple smoothers (`lm`, `loess`, `rlm`, `glm`, ...) on the same axes with per-smoother uncertainty bands. Each entry takes a `color` (cycles through a 5-color palette) and optional `label` / `uncertainty` / `level`. When any entry has a `label`, a manual color scale adds a legend. The binomial branch restricts overlay to `method="glm"`; other methods raise. 14 new tests.

### 0.4.0 (2026-08-30)
- **`uncertainty` parameter on `flexplot()`.** First-class confidence / prediction / bootstrap bands around every fitted line. New module `pyflexplot.uncertainty` exposes `validate_uncertainty_params`, `compute_bootstrap_ci`, `compute_prediction_band`, `format_band`.`..- 35 new tests, full suite 199 passed / 1 skipped (keras not installed), no regressions.

### 0.3.0 (2026-08-28)
- **`visualize()` now accepts `NeuralNetFit` wrappers** (DESIGN-7 from the v0.2.2 review). The duck-type dispatch avoids importing `flex_nn` at module load time, so the core module stays cheap when neural-net support isn't needed. The output mirrors the statsmodels `visualize()`: predicted-vs-actual line on top of a scatter. 7 new tests in `tests/test_design_followups.py::TestVisualizeNeuralNetFit`.
- **`flexplot()` method validation** (DESIGN-4) — unknown `method` values now raise `ValueError` instead of silently producing no smooth. The `method` parameter is checked against a `{auto, lm, loess}` whitelist at entry.
- **`flexplot()` given-variable validation** (DESIGN-3) — formulas with 3+ variables after `|` now raise `ValueError` instead of silently dropping `given[2:]`. Two-given is the maximum; `facet_grid` only supports row+column.
- **`bluepill.mixed_model(polynomials=...)` no longer requires `to`** (DESIGN-6). Split the interaction/polynomial validator into two: interactions still require `from`/`to`/`coef`; polynomials only need `from`/`coef`. The R-compatible shape (`from`/`to`/`coef`) is still accepted on polynomials but `to` is ignored for backwards compatibility.
- **Hypothesis property tests** — 14 new property-based tests in `tests/test_property_based.py` covering the formula parser (round-trip identity, deterministic parsing, malformed-input rejection across hundreds of generated formulas) and `mixed_model` rescaling invariants (output mean/SD match the declared spec within sampling tolerance; categorical columns only take declared levels). Each test runs 10-50 generated examples.
- Total test surface: 132 → 164 (32 new). All tests pass; no API breakage.

### 0.2.2 (2026-08-28)
- **Critical bug fix (bluepill)**: `mixed_model()` had an off-by-one column index that made the last predictor a constant column (its declared mean, zero variance) and shifted all other predictors by one column. The README's example produced `ses = 55.0` for every row. Fixed.
- **Critical bug fix (flex_nn)**: `permutation_importance()` crashed with `UnboundLocalError` on five of the eleven declared metric names (`auc`, `precision`, `recall`, `f1`, `loss`) because the scorer dispatch branches were missing. Added rank-based AUC, thresholded binary precision/recall/F1, and `loss` (MSE) scorers; the unreachable `if direction is None:` fallback block is gone.
- **Critical bug fix (bluepill)**: tuple-of-strings categorical specs (valid per the `VarSpec` type hint) were misidentified as continuous specs and crashed with `ValueError`. Extracted the numeric-detection logic into a shared `_is_continuous_spec()` helper so validation and execution agree.
- Added 20 contract-level regression tests in `tests/test_bluepill_correctness.py` and `tests/test_flex_nn_correctness.py`. They check that predictors have non-zero variance, that the strongest coefficient ranks first in permutation importance, that all declared metrics work end-to-end, and that tuple specs round-trip. Each of these tests fails on the pre-v0.2.2 code path; all 20 pass now. Total: 132 tests passing.
- Other cleanups from the v0.2.2 review: replaced `from plotnine import *` with explicit imports in `core.py` and `sem.py`, removed the unused `patsy` import, restored the model's original `training` flag in `_keras_predict()` (was permanently mutating caller state), and added an explicit "experimental / not yet implemented" note to `estimates()`.

### 0.2.1 (2026-08-28)
- Hardened the Keras 3 path in `pyflexplot.flex_nn`: predictions now go through a dedicated `_keras_predict()` helper that passes `training=False` (so `Dropout`/`BatchNorm` behave deterministically) and falls back gracefully for custom `Model` subclasses whose `predict()` doesn't accept the `training` kwarg.
- Added `tests/test_flex_nn_keras.py` with 14 keras-specific tests (skip when keras isn't installed; verified against `keras==3.15.1` + `jax` backend). Total test surface: 110 (core + torch) + 14 (keras when available) = 124.
- CI: split into three workflows -- `python-app.yml` (core, no optional deps, Python 3.10-3.13), `torch.yml` (torch CPU install, weekly schedule to catch upstream regressions), `keras3.yml` (keras[jax] install, weekly schedule, PRs touching flex_nn).
- Extended the example notebook with a Keras 3 walk-through and added a README section describing the optional install + backend selection.

### 0.2.0 (2026-08-28)
- Added `pyflexplot.flex_nn` — torch-default wrappers for fitting and visualizing neural networks. `NeuralNetFit` bundles a fitted `torch.nn.Module` (or `keras.Model`) with the metadata needed to plug into `compare_fits`. `permutation_importance()` provides column-shuffling variable importance.
- Added `pyflexplot.bluepill` — port of Dustin Fife's `bluepill` R package. `estimate_sd()` recovers an SD from a known mean and min/max range; `mixed_model()` generates clustered synthetic data with fixed + random effects, interactions, and polynomial terms.
- Dropped the aspirational `flexifiers` bullet from the "Included R Packages" list (no corresponding R package was found).
- Added 50 new tests across the two modules (60 → 110). Test suite uses `pytest.importorskip("torch")` so the package still imports cleanly without torch installed, but `flex_nn` tests skip when torch is absent.

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
