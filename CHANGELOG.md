# Changelog

All notable changes to py-flexplot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Recent highlights

- **0.6.2** — R-style interaction syntax (`y ~ x*z`, `y ~ x:z`) accepted by the formula parser; `flexplot()` emits a `UserWarning` when interaction syntax is present (v0.7.0 will add `interaction_model=True` for non-parallel slopes).
- **0.6.1** — Fixed dead binomial branch in `flexplot()`; numeric `[0, 1]` y now routes to the binomial GLM smoother (was straight LM line).
- **0.6.0** — New `diagnose(formula, data)` for auto data-quality diagnostics (missingness, Cook's D, Ramsey RESET, Breusch-Pagan).
- **0.5.0** — New `overlay=` parameter on `flexplot()` for multi-smoother comparison.
- **0.4.0** — First-class uncertainty layer on `flexplot()` (`uncertainty=`, `level=`, `bands=`); new `pyflexplot.uncertainty` module.
- **0.3.0** — `visualize()` accepts `NeuralNetFit` wrappers; formula parser validation hardened.

## [0.6.2] - 2026-08-30

### Added

- **R-style interaction syntax accepted by the formula parser.**
  `parse_flexplot_formula()` now recognizes `*` and `:` operators in the
  right-hand side of a formula. `y ~ x*z` is expanded to `y ~ x + z + x:z`
  for column lookup purposes; the interaction term `x:z` is preserved in
  `all_x` for forward-compatibility with v0.7.0.
- **`has_interaction` flag on the parsed formula dict** — boolean set
  to True when the formula contains `*` or `:`.
- **`UserWarning` emitted by `flexplot()` when interaction syntax is
  detected** — explicit notice that v0.6.x fits remain additive
  (parallel slopes per color group) and that v0.7.0 will add
  `interaction_model=True` for non-parallel slopes. To suppress the
  warning, write the formula without `*` or `:`.
- **`_expand_r_formula()` helper** — public, expands `a*b` to
  `a + b + a:b` recursively (handles `a*b*c`).
- **`_first_atom()` helper** — public, returns the first atom of a
  possibly-interacted term (`x:z` → `x`).
- **6 new tests in `tests/test_core.py`** — parser accepts `*` and `:`
  syntax, `flexplot()` warns when interaction syntax is present, no
  warning for plain `+` formulas, column lookup strips interaction
  suffixes.

### Notes

- The fit behavior is **unchanged** for non-interaction formulas.
- For interaction formulas, the fit is **additive by design** in
  v0.6.x; users who need non-parallel slopes should fit their own
  statsmodels model and use `visualize()` instead.

## [0.6.1] - 2026-08-30

### Fixed

- **Binomial GLM branch in `flexplot()` is now reachable for numeric binary
  outcomes.** Previously, `pd.api.types.is_numeric_dtype([0, 1])` returned
  True, so int/float binary y always routed to the LM/loess branch and the
  binomial GLM branch was dead code (only reachable for string y, where the
  internal `.astype(float)` raised first). Added a binary pre-check that
  detects unique values ⊆ {0, 1} BEFORE the numeric-dtype dispatch and
  routes that case to the binomial branch. Behavioral change: numeric
  `[0, 1]` y now produces a logistic/sigmoid curve (was a straight LM
  line). String `["yes", "no"]` and multi-level numeric `[0, 1, 2]`
  behavior unchanged.

### Tests

- Updated `tests/test_uncertainty.py::test_flexplot_binomial_ci_renders_smooth`
  to actually assert binomial branch parameters (`method="glm"`,
  `method_args={"family": "binomial"}`).
- Added 3 new tests to `tests/test_core.py`:
  - `test_flexplot_binary_y_routes_to_binomial_branch`
  - `test_flexplot_binary_y_as_float_also_routes_to_binomial`
  - `test_flexplot_non_binary_numeric_y_still_uses_lm`

## [0.6.0] - 2026-08-30

### Added

- **Auto data-quality diagnostics via `diagnose()`** — one-paragraph
  summary of a flexplot formula + data. Surfaces four diagnostics in a
  single text output:
  - **Missingness**: per-column counts and pattern heuristic
    (none / concentrated / spread).
  - **Outliers**: Cook's distance count and threshold (default `4/n`).
  - **Linearity**: Ramsey RESET test for functional-form misspecification.
  - **Heteroscedasticity**: Breusch-Pagan test for non-constant variance.
- **New module `pyflexplot.quality`** — public helpers `diagnose()`
  and `format_summary()`.
- **`__init__.py`** now exports `diagnose` and `format_summary` at the
  package level.
- **New test file `tests/test_quality.py`** — 19 tests covering
  validation, missingness patterns, outlier detection on clean and
  contaminated data, Ramsey RESET linearity detection, Breusch-Pagan
  heteroscedasticity detection, multi-predictor formulas, color/given
  variables, verbose vs quiet output, and `format_summary`.

## [0.5.0] - 2026-08-30

### Added

- **`overlay` parameter on `flexplot()`** — overlay multiple smoothers on
  the same axes for visual model comparison. Accepts a list of method
  names (``"lm"``, ``"loess"``, ``"rlm"``, ``"glm"``, etc.) or dicts
  with keys ``method``, ``color``, ``label``, ``uncertainty``, ``level``.
  Distinct colors cycle through a 5-color palette; labels appear in the
  legend when provided. Lets the user SEE which fit the data prefers.
- New helpers ``_normalize_overlay()``, ``_add_overlay_numeric()``,
  ``_add_overlay_binomial()`` in ``pyflexplot.core``. Overlay on the
  binomial branch only accepts ``"glm"``; other methods raise.
- **New test file `tests/test_overlay.py`** — 14 tests covering
  string/dict input, validation, per-entry propagation of method/color/
  level, primary-with-overlay interaction, and legend integration.

## [0.4.0] - 2026-08-30

### Added

- **Uncertainty layer for `flexplot()`** — first-class, configurable band
  around every fitted line. New parameters on `flexplot()`:
  - `uncertainty`: `{None, "ci", "prediction", "bootstrap"}`, default `"ci"`.
    `None` disables the fit (scatter only). `"ci"` draws the confidence
    interval on the mean response (plotnine built-in). `"prediction"`
    draws a residual-based prediction interval on new observations.
    `"bootstrap"` runs a case-resampled CI (loess branch only,
    n_resamples=200).
  - `level`: float in `(0, 1)`, default `0.95`. Coverage probability for
    a single band. Ignored when `bands` is given.
  - `bands`: list of floats in `(0, 1)`, optional. Nested coverage levels
    (e.g., `[0.5, 0.8, 0.95]`) for Tufte-style multi-ribbon display.
    Overrides `level` when provided.
- **New module `pyflexplot.uncertainty`** — public helpers:
  `validate_uncertainty_params`, `compute_bootstrap_ci`,
  `compute_prediction_band`, `format_band_label`, `VALID_UNCERTAINTY`.
- **New test file `tests/test_uncertainty.py`** — 35 tests covering
  parameter validation, bootstrap CI shape/bounds/reproducibility,
  prediction-band symmetry, and end-to-end `flexplot()` integration.

### Known issue (pre-existing, not introduced by this release)

- The "binomial GLM" branch in `flexplot()` is unreachable for `int` and
  `float` binary outcomes. `pd.api.types.is_numeric_dtype([0, 1])` returns
  `True`, so the LM branch is always taken for `y in {0, 1}`. The
  validation logic in the binomial branch never fires. This release adds
  a regression test (`test_flexplot_binomial_ci_renders_smooth`) that
  documents the current behavior; a future release should re-route
  numeric-binary `y` to the binomial branch explicitly.

## [0.3.0] - 2026-07-12

### Changed

- Addressed top-5 design follow-ups from v0.2.2 review.
- See commit `fec2bc5` for details.

## [0.2.2] - 2026-07-08

### Fixed

- Three critical bugs found in v0.2.2 code review.
- See commit `2fc408a` for details.

## [0.2.1] - 2026-06-23

### Fixed

- Hardened Keras 3 path in `flex_nn`.

## [0.2.0] - 2026-06-22

### Added

- Marshall Goldsmith coaching-insight wiring (Sigma integration).

## [0.1.0] - 2026-05-30

### Added

- Initial Python port of Dustin Fife's R `flexplot` and `fifer` packages.
- Core visualization: `flexplot`, `visualize`, `compare_fits`, `added_plot`.
- Biostatistics: `model_comparison`, `estimates`, `p_format`,
  `eliminated_columns`, `color_table`.
- Empirical Bayes shrinkage: `fit_beta_prior`, `add_ebb_estimate`.
- SEM visualization: `hopper_plot`, `disturbance_plot`, `measurement_plot`.
- Simulated datasets: `estimate_sd`, `mixed_model`.
- Neural-network integration: `flex_nn` module.