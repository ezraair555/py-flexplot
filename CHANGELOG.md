# Changelog

All notable changes to py-flexplot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Recent highlights

- **0.7.4** — New `pyflexplot.meansplot()` ports R's `fifer::meansplot()` for descriptive-statistics visualizations (mean + error bar per group, with `error="se"|"sd"|"ci"|"range"|"iqr"|"no"`).
- **0.7.3** — `estimates()` returns a real R² confidence interval via non-central-F inversion (was `None` placeholder in v0.6.x); new `eta_squared()` for partial eta-squared; `ghost_line="slope1"` adds a diagonal slope=1 reference for prediction-vs-observed overlays.
- **0.7.0** — `flexplot()` gains `interaction_model=True` for non-parallel slopes per color group when the formula uses R-style interaction syntax (`*` or `:`). Closes the largest semantic gap in the v0.6.2 R-audit.
- **0.6.7** — New `pyflexplot.ml.RFAdapter` lets scikit-learn estimators (random forests, gradient boosting, any `.predict()`-bearing estimator) participate in `compare_fits()` alongside statsmodels fits. Documentation honesty pass: README + `docs/index.md` + `docs/api/core.md` + `docs/api/stats.md` now reflect what's actually in the package; new `docs/api/coverage.md` gives a per-feature matrix vs the R packages.
- **0.6.6** — `flexplot()` gains `ghost.reference` (DataFrame overlay for reference scatter or prediction line), `plot.string` (label override dict), and `related` (no-op on Python side; plotnine already shares scales by default).
- **0.6.5** — `flexplot()` gains `sample` (subsample rows for plotting, full data still used for fits), `ghost_line` ("red" or "dashed" reference line), `plot_type` override ("scatter", "line", "boxplot", "bar"), and `return_data` (returns `{"plot", "data"}` instead of just plot). Closes the remaining easy-wins from the v0.6.x R-audit.
- **0.6.4** — `flexplot()` gains `bins` / `labels` / `breaks` (numeric-x auto-discretization, R-flexplot parity), `spread` (stdev/range/iqr/ci/no dispersion markers), and `method='polynomial'` / `'cubic'` / `'logistic'` parametric smoothers. Closes the auto-bin gap from the v0.6.2 R-audit.
- **0.6.3** — `model_comparison()` now exposes Bayes factor + adj.R²; `compare_fits()` gains `return_preds` + `pred_type`; `estimates()` is a real structured effect-size reporter (R², sigma, coef DataFrame, standardized betas, semi-partial R², factors/numbers split); `visualize()` accepts `plot='residuals'` / `'all'`.
- **0.6.2** — R-style interaction syntax (`y ~ x*z`, `y ~ x:z`) accepted by the formula parser; `flexplot()` emits a `UserWarning` when interaction syntax is present (v0.7.0 will add `interaction_model=True` for non-parallel slopes).
- **0.6.1** — Fixed dead binomial branch in `flexplot()`; numeric `[0, 1]` y now routes to the binomial GLM smoother (was straight LM line).
- **0.6.0** — New `diagnose(formula, data)` for auto data-quality diagnostics (missingness, Cook's D, Ramsey RESET, Breusch-Pagan).
- **0.5.0** — New `overlay=` parameter on `flexplot()` for multi-smoother comparison.
- **0.4.0** — First-class uncertainty layer on `flexplot()` (`uncertainty=`, `level=`, `bands=`); new `pyflexplot.uncertainty` module.
- **0.3.0** — `visualize()` accepts `NeuralNetFit` wrappers; formula parser validation hardened.

## [0.7.4] - 2026-08-31

### Added

- **`pyflexplot.meansplot()`** — port of R's `fifer::meansplot()`.
  Plots the mean of `y` per level of `x` with a configurable error bar:
  - `error="se"`: standard error of the mean (default).
  - `error="sd"`: sample standard deviation (ddof=1).
  - `error="ci"`: `level` confidence interval on the mean (t-distribution).
  - `error="range"`: min-max range.
  - `error="iqr"`: Q1-Q3 interquartile range.
  - `error="no"`: no error bar.
  - `connect=True` (default): draw a dashed gray line connecting the
    per-group means (useful for ordinal predictors).

  Implementation lives in `src/pyflexplot/descriptives.py` to keep the
  descriptive-stats surface separate from the formula-dispatch logic in
  `core.py`. Numeric predictors with few unique values are coerced to
  discrete levels so the line connects properly.

  Documented in [`docs/api/descriptives.md`](docs/api/descriptives.md).

### Tests

- 12 new tests in `tests/test_meansplot.py` covering: smoke / layer
  count, error-bar omission (`error="no"`), line omission
  (`connect=False`), validation (invalid `error`, color term, non-
  numeric y, `given` term), and numeric correctness (per-group means
  match manual groupby, SE bar width = `std/sqrt(n)`, SD bar width =
  sample std, numeric-x discretization).

## [0.7.3] - 2026-08-31

### Added

- **Real R² confidence interval in `estimates()`** — `_r_squared_ci()`
  now uses non-central-F inversion (Olkin & Finn, 1995) to compute the
  CI for the population R². Replaces the `None` placeholder documented
  since v0.6.3. Method:
  - Compute observed F statistic = (R²/k) / ((1-R²)/(n-k-1)).
  - Invert the noncentral F upper-tail survival function at α/2 and
    1-α/2 to find λ at each tail.
  - Recover ρ² via ρ² = λ / (n + λ).
  - scipy's `ncf.sf` returns a buggy negative value at `nc=0` on some
    versions; the implementation falls back to `f.sf` (central F) when
    nc=0.
  - Boundary cases: R² near 1 collapses the upper bound to 1; R² near
    0 collapses the lower bound to 0.

- **`eta_squared()` on the top-level package** — partial eta-squared
  (η²_p) for a fitted OLS model, with a per-model CI via the same
  non-central-F inversion. Returns a one-row DataFrame indexed by
  `"model"` (statsmodels' OLS exposes a single model-F, not per-term;
  for per-predictor semi-partial R², use `estimates()` instead). Port of
  R's `sjstats::eta_sq()` / `fifer::eta_squared()`.

- **`ghost_line="slope1"`** on `flexplot()` — adds a diagonal slope=1
  reference line via `geom_abline(intercept=0, slope=1)` for prediction-
  vs-observed overlays. The legacy `"dashed"` semantics (horizontal
  reference at y=0) are preserved.

### Tests

- 13 new tests in `tests/test_r_squared_ci.py` covering: validation
  (negative R², R² ≥ 1, invalid df, too few obs), output shape, the
  observed-R²-inside-CI property, CI narrows with larger n, CI widens
  with more predictors, parametric coverage tests across 3 seeds
  (population R² inside the CI), and integration with `estimates()`.
- 9 new tests in `tests/test_eta_squared.py` covering: shape, columns,
  unit-interval value, exact formula match, CI contains point estimate,
  input validation, and cross-validation with `_r_squared_ci`.
- 3 new tests in `tests/test_flexplot_extras.py` for `ghost_line=
  "slope1"`: layer adds `geom_abline`, doesn't add `geom_hline`, and
  invalid values raise `ValueError`.

### Documentation

- Docstring for `flexplot()` updates `ghost_line` accepted values.
- CHANGELOG: this entry.
- Coverage matrix (`docs/api/coverage.md`): nothing changes — these
  are v0.7.0 candidates we'd flagged earlier.

## [0.7.0] - 2026-08-31

### Added

- **`interaction_model: bool`** on `flexplot()` — when `True` AND the formula
  contains `*` or `:` syntax, fit a statsmodels OLS with the actual
  interaction term (e.g. `y ~ x * z`) and overlay non-parallel per-color-
  group regression lines. Default behavior is preserved: `interaction_model=
  False` keeps the legacy additive fit + `UserWarning`.
  Implementation: `_add_interaction_smooth()` fits `y ~ _x * _color` via
  `statsmodels.formula.api.ols`, then predicts on a per-color grid and
  draws one `geom_line` + optional `geom_ribbon` per level of the color
  group. Honors `uncertainty` ("ci" / "prediction"), `level`, and `bands`.
  Fallback conditions (re-routes through `_add_numeric_smooth()`):
  - No interaction term in the formula (e.g. `y ~ x + z`).
  - `color` group is `None` (e.g. `y ~ x:z` alone).
  - Only one level in the color group.
  When set, suppresses the additive-fit `UserWarning`.

### Notes

- v0.7.0 is a minor-version bump (semver) because the new feature
  introduces a visible behavioral change at the dispatch level (per-group
  fits vs additive) and a new boolean parameter.
- Modeling principle #1 (correct specification) is now honored for
  interaction formulas: when the user writes `y ~ x*z` and opts in via
  `interaction_model=True`, the fitted model includes the `x:z` term.

### Tests

- 11 new tests in `tests/test_interaction_model.py` covering: warning
  suppression, per-group lines, distinct slopes, fallback conditions
  (no interaction term, no color, single color level), uncertainty =
  None / "ci" / bands=[0.5, 0.95], and backward compat with the
  default additive path.

## [0.6.7] - 2026-08-31

### Added

- **`pyflexplot.ml.RFAdapter`** — thin wrapper so any scikit-learn
  estimator (random forests, gradient boosting, plain linear regression)
  can be used with `compare_fits()` and the rest of py-flexplot's
  visualization API. Carries the predictor-names metadata that
  `compare_fits()` needs to build evaluation DataFrames.
  - `RFAdapter(estimator, response_var, predictor_names)` — primary API.
  - `make_rf_adapter(estimator, data, response_var, predictor_names=None)` — convenience
    constructor that infers predictor names from a DataFrame.
  - Methods: `.predict(X)` (DataFrame or ndarray), `.predict_df(data)`
    (returns a single-column DataFrame named `pred_<response_var>`).
  - scikit-learn is **not** a declared dependency of py-flexplot; the
    adapter raises a clear `ImportError` if sklearn is missing. Tests
    `pytest.importorskip("sklearn")` so the test suite stays green without
    sklearn.
  - Documented in [`docs/api/ml.md`](docs/api/ml.md).

- **Documentation honesty pass.** The README and docs previously described
  py-flexplot as a "Python port of R's flexplot" without acknowledging
  partial coverage. The package is now described honestly:
  - **README** — new "What's covered (and what isn't)" section with
    high-level coverage matrix and links to `coverage.md`.
  - **docs/index.md** — new top-level banner pointing to
    [`coverage.md`](docs/api/coverage.md).
  - **docs/api/coverage.md** — new file: per-feature coverage matrix vs
    R-flexplot / fifer / flexplavaan, with explicit ✅ / ⚠️ / ❌ markers
    and notes on what's out-of-scope (mixed-effects via `lme4` /
    `statsmodels.MixedLM` mismatch; full SEM fitting; sklearn is an
    optional dep).
  - **docs/api/stats.md** — `model_comparison` signature now reflects
    the v0.6.3+ return shape (AIC / BIC / LogLik / R² / adj.R² /
    BayesFactor); new `estimates()` and `compare_fits()` sections
    document the v0.6.3 additions.
  - **docs/api/core.md** — extended parameter docs for v0.6.4–v0.6.6
    additions (`bins` / `labels` / `breaks`, `spread`,
    `sample` / `ghost_line` / `plot_type` / `return_data`,
    `ghost_reference` / `plot_string` / `related`); new `visualize`
    section with `plot='model' | 'residuals' | 'all'`; new examples
    block exercising the new params.

### Tests

- 12 new tests in `tests/test_ml_adapter.py` covering construction,
  DataFrame / ndarray prediction alignment, classifier support,
  integration with `compare_fits()`, and the
  `make_rf_adapter` convenience constructor.

## [0.6.6] - 2026-08-31

### Added

- **`ghost_reference: pd.DataFrame`** on `flexplot()` — overlays a reference
  dataset on the same axes. Two patterns detected by column shape:
  - `(x, y)` columns: draws a `geom_point` layer in gray, alpha=0.4
    (typical for "compare to a reference group").
  - `(x, "pred")` columns: draws a `geom_line` layer in red dashed
    (typical for prediction-vs-observed overlays).
  Validation: rejects non-DataFrame, missing x column, missing y/pred.
  R-flexplot parity.

- **`plot_string: dict`** on `flexplot()` — overrides the axis/legend
  labels derived from the formula. Accepts keys: `x`, `y`, `title`,
  `subtitle`, `caption`, `color`. Unknown keys are silently dropped
  (plotnine's `labs()` rejects them). Validation: rejects non-dict, non-
  string keys/values.

- **`related: bool`** on `flexplot()` — R-flexplot's panel-linking flag.
  Currently a no-op on the Python side because plotnine's facets share
  scales by default (`scales="fixed"`). Accepted for R-parity; future
  work could surface `scales="free_x"` / `"free_y"` / `"free"` as the
  actual user-facing control. Validation: rejects non-bool.

### Tests

- 9 new tests in `tests/test_ghost_reference.py` (validation + scatter
  pattern + pred-line pattern + interaction with ghost_line).
- 12 new tests in `tests/test_plot_string_related.py` (default behavior,
  override semantics, multi-key dicts, unknown-key silent drop,
  non-dict rejection, related bool validation).

## [0.6.5] - 2026-08-31

### Added

- **`sample: int`** on `flexplot()` — subsample N rows for the plotnine layers
  only; the smoother fits still see the full DataFrame. Deterministic via
  `np.random.default_rng(0)`. Useful for very large datasets where the
  scatter layer is the bottleneck but the fit should remain robust.
  Validation: rejects non-int (except bool), values < 1, and silently
  no-ops when `sample >= len(data)`.

- **`ghost_line: {"red", "dashed", None}`** on `flexplot()` — adds a
  reference `geom_hline` at y=0 after the main layers. `"red"` is a solid
  red reference (R-flexplot uses this for y=0 thresholds); `"dashed"` is
  a black dashed reference (R-flexplot uses this for prediction-vs-
  observed slope=1 references, though we only emit the horizontal at y=0
  for now; diagonal slope=1 is a v0.7.0 todo).

- **`plot_type: {"scatter", "line", "boxplot", "bar", None}`** override on
  `flexplot()` — bypasses the auto-dispatch and forces a specific geom.
  Useful when:
  - the auto-dispatch picks the wrong branch (e.g. 11 unique x values
    rather than the 10 cut-off so the discrete branch isn't taken);
  - the user knows they want a boxplot regardless of how x is shaped;
  - a publication requires a specific plot style.
  Implementation: short-circuits the dispatch chain via a `skip_dispatch`
  boolean. Validation rejects unknown values.

- **`return_data: bool`** on `flexplot()` — when `True`, returns
  `{"plot": ggplot, "data": DataFrame}` instead of just the ggplot.
  Useful for downstream tooling that wants both the rendered plot and the
  actual data that was plotted (especially when combined with `sample=`
  to know which rows were subsampled). Both return points (intercept-only
  and main path) honor this.

### Tests

- 20 new tests in `tests/test_flexplot_extras.py` covering all four new
  params, including validation, interaction (return_data + sample =
  subsampled dict), and backward compat (defaults preserve legacy
  behavior).

## [0.6.4] - 2026-08-31

### Added

- **`flexplot()` auto-bins numeric x via `bins=N` / `breaks=[...]` / `labels=[...]`** — closes the largest usability gap in the v0.6.2 R-audit (R's `flexplot()` auto-discretizes numeric predictors; the Python port required users to pre-cut their data). Three new parameters:
  - `bins: int` — number of equal-width bins (default `None` = no binning).
  - `breaks: list[float]` — explicit cut points; takes precedence over `bins` when both are given (with a `UserWarning`).
  - `labels: list[str]` — custom labels for the resulting discrete x levels; must have `len(breaks) - 1` or `len(bins)` entries depending on which path is taken.
  - Validation lives in `_validate_binning_params()` (rejects non-int `bins`, non-monotonic `breaks`, wrong-length `labels`).
  - Implementation uses `pd.cut()` with `include_lowest=True`; the resulting x is converted to string so plotnine treats it as discrete and the existing discrete-style summary layer applies.

- **`spread=` parameter on `flexplot()`** — controls the dispersion marker drawn alongside `geom_jitter` in the discrete-x branch. Mirrors R-flexplot's `spread`:
  - `None` / `"ci"` (default): bootstrap CI via `stat_summary(fun_data="mean_cl_boot")` — legacy behavior.
  - `"stdev"`: mean ± 1 SD as `geom_pointrange`.
  - `"range"`: min-max range.
  - `"iqr"`: Q1-Q3 IQR.
  - `"no"`: no summary layer.
  - Implemented via `_add_discrete_summary()` + `_make_spread_fn()` helper.

- **New `method` values: `"polynomial"`, `"cubic"`, `"logistic"`** — closes the parametric-smoother gap from the v0.6.2 audit. Routes through `_add_parametric_smooth()` which fits statsmodels directly and draws `geom_line` + `geom_ribbon` layers manually (plotnine's `geom_smooth(method="lm", ...)` doesn't accept `poly(x, k)` formulas cleanly).
  - `"polynomial"` / `"cubic"`: OLS with degree-3 polynomial in x. Cubic is an alias.
  - `"logistic"`: GLM with logit link on numeric binary y. Falls back to OLS with a `UserWarning` if y is not in `{0, 1}`.
  - When `method="logistic"` is explicit, the binary-y pre-check is bypassed so the parametric branch always fires (rather than the legacy binomial branch).
  - Nested `bands=[...]` works on all three new methods via `model.get_prediction().summary_frame(alpha)`.

### Tests

- 13 new tests in `tests/test_parametric_smooth.py` covering method registration, polynomial/cubic dispatch, logistic on binary + non-binary y, OLS fallback warning, nested bands, and backward compat with `method="auto"` / `"lm"`.
- 18 new tests in `tests/test_binning.py` covering param validation (non-int bins, short/non-monotonic breaks, wrong-length labels, both-set precedence), `_maybe_bin_numeric_x` (no-op, equal-width, explicit cuts, custom labels), and integration (route to discrete branch, no-op on already-discrete x, breaks-wins precedence).
- 12 new tests in `tests/test_spread.py` covering the full spread-value matrix, helper-fn shapes, and backward compat.

## [0.6.3] - 2026-08-30

### Added

- **`model_comparison()` now exposes Bayes factor + R² + adj.R².** Previously only an F-test was emitted; the new surface lets users compare models via Bayesian evidence (BF10 from the F-statistic approximation) and classical fit statistics side-by-side.

- **`compare_fits()` gains `return_preds: bool` and `pred_type: {"response", "link"}`** — when `return_preds=True`, the caller's plotnine overlay receives a DataFrame with one column per fit. `pred_type` controls whether predictions are on the response scale or the linear-predictor scale.

- **`estimates()` is now a real structured effect-size reporter** — closes the largest single R-parity gap in `pyflexplot.stats`. Previously a stub returning `model.summary()`; the new implementation returns a dict with:
  - `r.squared`, `adj.r.squared`, `sigma`, `n`
  - `r.squared.ci` (placeholder; non-central-F inversion is a v0.7.0 todo)
  - `coef`: DataFrame with `estimate`, `std.error`, `t`, `p.value`, `ci.lower`, `ci.upper`
  - `standardized`: Series of standardized betas per predictor
  - `semi.p.r2`: Series of semi-partial R² per predictor (computed via reduced-model fits)
  - `factors`, `numbers`, `formula`: predictor-type classification + the fitted formula string

- **`visualize(plot=)` switch** — R's `flexplot::visualize()` accepts `plot=c('all', 'residuals', 'model')`. The Python port now matches:
  - `plot='model'` (default): legacy predicted-vs-observed scatter with the fitted line. Unchanged.
  - `plot='residuals'`: returns `{'rvf': ggplot, 'hist': ggplot}` — residual-vs-predicted scatter + a residual histogram.
  - `plot='all'`: tries a cowplot-joined 2-column layout; falls back to a dict `{'model', 'rvf', 'hist'}` if cowplot isn't installed.

### Tests

- 30+ new tests covering the new surfaces, regression tests for invalid `plot=` values, and parity tests for the estimates() coef DataFrame shape.

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