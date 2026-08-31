# Changelog

All notable changes to py-flexplot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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