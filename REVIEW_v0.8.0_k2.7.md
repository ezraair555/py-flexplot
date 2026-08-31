# Independent Code Review: py-flexplot v0.8.0 vs. R `flexplot`

**Reviewer:** `ollama/kimi-k2.7-code:cloud` (current OpenClaw model)  
**Date:** 2026-08-31  
**Scope:** Python source under `src/pyflexplot/`, tests under `tests/`, and the R reference at `r-flexplot/R/`. Read-only review; no source files modified.  
**Test suite status:** 482 passed, 4 skipped (keras/rpy2), 11 warnings in ~18 s.  

---

## 0. Executive summary

`py-flexplot` v0.8.0 is a substantial improvement over v0.2.1: the v0.2.1 critical bugs in `bluepill.mixed_model`, `flex_nn.permutation_importance`, categorical tuple handling, and the `estimates()` stub have all been fixed, test coverage has more than quadrupled, and the test suite is green.

However, **R parity is still partial**. The port currently supports the most common univariate / bivariate / multivariate-with-color paths, but several R behaviors either differ silently or are missing entirely. The highest-impact gaps are:

1. **`method="polynomial"` is cubic in Python but quadratic in R** — a silent semantic mismatch for anyone porting R code.
2. **`spread="sterr"` is not a standard error** in Python; it is aliased to a bootstrap CI.
3. **Default dispersion for categorical x is wrong**: Python defaults to a bootstrap CI on the mean; R defaults to median ± quartiles.
4. **`method` validation accepts `rlm` and `glm` but the plot layer silently falls back to OLS/LM**.
5. **`related=TRUE`, `third.eye`, formula functions (`log(x)`, `poly(x,2)`), and most univariate `plot.type`s are not implemented**.

These are not all bugs in isolation, but they are parity gaps that will surprise users expecting the R API to translate one-for-one.

---

## 1. Critical / high-severity parity issues

### BUG-1: `method="polynomial"` is degree-3, but R `polynomial` is degree-2

**Python:** `src/pyflexplot/core.py:1130` / `_add_parametric_smooth` lines ~1280-1290  
**R:** `r-flexplot/R/hidden_functions.R` `fit.function(..., method="polynomial")`  
**Severity:** High — silent wrong model when porting R code

In the Python implementation both `"polynomial"` and `"cubic"` build the same design matrix `[1, x, x², x³]`:

```python
if method in {"polynomial", "cubic"}:
    X = np.column_stack([np.ones_like(x_arr), x_arr, x_arr ** 2, x_arr ** 3])
```

In R, `"polynomial"` maps to `y ~ poly(x, 2, raw=TRUE)` (quadratic) and only `"cubic"` maps to degree 3. The Python docstring also incorrectly states that degree-3 is "R-flexplot's default."

**Fix:** Make `"polynomial"` quadratic (degree 2) and keep `"cubic"` degree 3. Update docstring and `_VALID_FLEXPLOT_METHODS` ordering.

---

### BUG-2: `spread="sterr"` is aliased to a bootstrap CI, not a standard error

**Python:** `src/pyflexplot/core.py:304-333` (`_add_discrete_summary`)  
**R:** `r-flexplot/R/hidden_functions.R` `fit.function(..., spread="sterr")`  
**Severity:** High — reports the wrong statistic

```python
elif spread == "sterr":
    spread = "ci"
```

R's `spread="sterr"` draws mean ± 1.96 × (SD / √n). Python silently replaces this with `stat_summary(fun_data="mean_cl_boot")`, which bootstraps the mean. The resulting intervals can be materially different, especially with small N or skewed residuals.

**Fix:** Implement the actual SE formula in `_add_discrete_summary` or reject `spread="sterr"` until it is implemented. Do not alias it to a bootstrap CI.

---

### BUG-3: `method="rlm"` and `method="glm"` are accepted but downgraded to LM

**Python:** `src/pyflexplot/core.py:394-397` (`_VALID_FLEXPLOT_METHODS`), `_add_numeric_smooth` lines ~1170-1190  
**Severity:** High — validation promises behavior the code does not deliver

`_VALID_FLEXPLOT_METHODS` includes `"rlm"` and `"glm"`. The public docstring does not advertise `"glm"`, but `"rlm"` is listed. In `_add_numeric_smooth` the only routed methods are `"loess"`, `"polynomial"`, `"cubic"`, and `"logistic"`. Any other value (including `"rlm"` and `"glm"`) falls through to `geom_smooth(method="lm", ...)`, so a user asking for robust regression gets ordinary OLS without any warning.

**Fix:** Either remove `"rlm"` / `"glm"` from the valid set until they are implemented, or raise `ValueError` when they are requested.

---

### BUG-4: Default `spread` for categorical-x plots differs from R

**Python:** `src/pyflexplot/core.py:304-333`  
**R:** `r-flexplot/R/flexplot.R` default `spread = "quartiles"`, `r-flexplot/R/hidden_functions.R` `fit.function(..., spread="quartiles")`  
**Severity:** Medium-High — visual and statistical default mismatch

Python maps `spread=None` to `"ci"` (bootstrap CI on the mean). R maps the categorical-x summary to `"quartiles"` by default, which draws median ± Q1/Q3. This means the *default* plot for, e.g., `flexplot("y ~ x", data=df)` with categorical `x` will show different central tendency and dispersion in Python vs. R.

**Fix:** Change the default for categorical-x summaries to `"quartiles"` (median ± IQR) to match R, while keeping the legacy `"ci"` available as an explicit opt-in.

---

## 2. Design / feature-parity gaps

### DESIGN-1: `related=TRUE` is a documented no-op

**Python:** `src/pyflexplot/core.py` lines ~990-1010  
**R:** `r-flexplot/R/flexplot.R` and `r-flexplot/R/uni.plot.R` / related logic  
**Severity:** Medium

The Python docstring says `related` is "currently a no-op on the Python side." In R, `related=TRUE` triggers a paired / related-samples analysis (difference scores and a related t-test display). Users passing `related=True` will get the same plot as `related=False`.

**Fix:** Implement the difference-score path or, at minimum, raise `NotImplementedError` so users are not misled.

---

### DESIGN-2: `third.eye` visualization is not implemented

**Python:** not present in `core.py` or `__init__.py`  
**R:** `r-flexplot/R/third.eye.R`  
**Severity:** Medium

R exposes `third.eye()` for three-way interaction visualization. The Python port has no equivalent.

---

### DESIGN-3: Univariate `plot.type` options are heavily truncated

**Python:** `src/pyflexplot/core.py:747-751` (intercept-only branch) and `plot_type` handling lines ~1050-1080  
**R:** `r-flexplot/R/flexplot_helper.R` `flexplot_histogram(..., plot.type=c("histogram","qq","density","boxplot"))`  
**Severity:** Medium

For an intercept-only formula, Python only draws a histogram with `bins=30`. R supports `plot.type="histogram"`, `"qq"`, `"density"`, and `"boxplot"` and uses `calculate_bins_for_histograms` for automatic bin selection.

The public `plot_type` parameter in Python only accepts `"scatter"`, `"line"`, `"boxplot"`, `"bar"`, and `None`. `"violin"`, `"histogram"`, `"qq"`, and `"density"` are missing.

**Fix:** Extend the intercept-only branch and the `plot_type` dispatch to cover the R variants.

---

### DESIGN-4: Formula transformations / functions are not supported

**Python:** formula parser in `src/pyflexplot/core.py`  
**R:** `r-flexplot/R/flexplot.R` / `flexplot_modify_data` / `formula_functions` handling  
**Severity:** Medium

R flexplot supports formulas such as `y ~ log(x)` or `y ~ x + I(x^2)`. The Python parser extracts variable names by regex / `all.vars`-like logic and will fail or silently mis-parse formulas containing transformations because it looks for column names that do not exist in the data frame.

**Fix:** Document the limitation or implement a formula-to-data transformation pipeline (e.g., `patsy` / `statsmodels.formula`) for the plotting path.

---

### DESIGN-5: Multivariate layout with 2+ numeric predictors is not supported

**Python:** `src/pyflexplot/core.py` formula parsing and dispatch  
**R:** `r-flexplot/R/hidden_functions.R` `make_flexplot_formula`, `variable_types`, and `flexplot_multivariate_aes`  
**Severity:** Medium

R automatically assigns up to four variables to slots (y, x, color, linetype/shape, panels) and, when a second numeric predictor is used, bins it and maps it to color/linetype/shape. Python supports exactly one x, one optional color, and up to two `given` panel variables. A formula like `y ~ x1 + x2` (both numeric) cannot be rendered the way R would.

**Fix:** Implement the R slotting algorithm or document that the port currently requires the user to bin the second numeric predictor manually.

---

### DESIGN-6: `alpha` default does not match R

**Python:** `src/pyflexplot/core.py` passes `alpha` through to `geom_jitterd` / `geom_jitter`; default is `None`  
**R:** `r-flexplot/R/flexplot_helper.R` `flexplot_alpha_default`  
**Severity:** Low-Medium

R defaults to `alpha=0.99977` and then chooses `0.5` for numeric x or `0.2` for categorical x. Python defaults to opaque points unless the user sets `alpha`. The visual density/overplotting behavior will differ.

---

### DESIGN-7: `jitter` semantics differ for categorical axes

**Python:** `src/pyflexplot/core.py` ~770-790 and jitter validation  
**R:** `r-flexplot/R/hidden_functions.R` `match_jitter_categorical`, `points.func`  
**Severity:** Low

R has a dedicated `geom_jitterd` / `position_jitterdodged` path for categorical x with optional color grouping. Python uses a single `geom_jitter` call with width/height. The dodge behavior for two categorical axes is likely different.

---

### DESIGN-8: `estimates()` lacks factor-level pairwise comparisons and categorical standardized betas

**Python:** `src/pyflexplot/stats.py:404`  
**R:** `r-flexplot/R/helper_estimates.R`, `r-flexplot/R/estimates.R`  
**Severity:** Low-Medium

The Python docstring explicitly notes that "Cohen's d / factor pairwise differences and standardized betas for categorical predictors are NOT yet implemented." R `estimates()` returns pairwise differences and semi-partial R² for factors. This is an acknowledged gap, but it matters for ANOVA-style reporting.

---

### DESIGN-9: `bluepill.mixed_model` explained-variance formula may be a port error

**Python:** `src/pyflexplot/bluepill.py` ~line 220  
**Severity:** Low (pending R reference check)

```python
explained = float(np.sum(np.asarray(fixed[1:]) ** 2) ** 2)
```

The code says this mirrors R's `sum(fixed^2)^2`. If the R source really squares the *sum of squares* again, then the port is correct; otherwise the residual variance calculation is off. I could not locate a direct R `bluepill` file in the cloned repo, so this is flagged for the maintainer to verify.

---

## 3. Test coverage gaps

### COVERAGE-1: R parity tests are skipped in CI

**File:** `tests/test_r_parity.py` lines 179, 191, 211  
**Status:** Skipped because `rpy2` is not installed in the current environment

These tests compare actual R output to Python output. Until they run regularly, parity regressions will not be caught automatically.

**Fix:** Add `rpy2` (and an R installation) to the test environment, or run the R parity suite in a separate CI job.

---

### COVERAGE-2: `spread="sterr"` behavior is not asserted against R

No test checks that `spread="sterr"` produces a standard-error interval. The existing `tests/test_spread.py` may pass because the alias happens to return *an* interval, but it does not verify the statistic.

---

### COVERAGE-3: `method="polynomial"` degree is not asserted

No test verifies that `method="polynomial"` yields a quadratic fit. The method is exercised in `tests/test_parametric_smooth.py` and `tests/test_v080_parity.py`, but the degree is not asserted.

---

### COVERAGE-4: `method="rlm"` / `method="glm"` acceptance is not tested

If a test had called `flexplot(..., method="rlm")`, the silent LM fallback would be visible. These methods are untested.

---

### COVERAGE-5: `plot.type` variants and `related=TRUE` paths are missing

Because the features are not implemented, there are no tests. As they are added, tests should cover histogram/qq/density/boxplot/violin and the related-samples difference path.

---

## 4. Documentation / API drift

### DOC-1: `method` docstring disagrees with the validator

The `flexplot()` docstring lists valid methods as:

```python
{"auto", "lm", "loess", "polynomial", "cubic", "logistic"}
```

but `_VALID_FLEXPLOT_METHODS` at `core.py:394` is:

```python
{"auto", "lm", "loess", "polynomial", "cubic", "logistic", "rlm", "glm"}
```

The extra two are not implemented, so either the docstring is incomplete or the validator is too permissive.

---

### DOC-2: `spread` docstring omits `"sterr"` and `"quartiles"`

The `flexplot()` docstring says `spread` accepts `{None, "ci", "stdev", "range", "iqr", "no"}` but `_VALID_SPREAD` at `core.py:299` includes `"sterr"` and `"quartiles"` as aliases. Users reading only the docstring will not know those values exist, and if they try them they will get behavior that differs from R.

---

### DOC-3: `polynomial` degree is documented incorrectly

The docstring states `"polynomial" / "cubic": degree-3 OLS in x (cubic is an alias)` and claims this matches "R-flexplot's default." R's default polynomial is degree 2.

---

## 5. Positive observations

- **v0.2.1 regressions are fixed.** `mixed_model` predictor column handling, `permutation_importance` metric dispatch, categorical tuple detection, and the `estimates()` stub are all corrected.
- **Test suite is healthy.** 482 tests pass; only optional backends (keras, rpy2) are skipped.
- **Formula validation is much better.** Missing columns, non-numeric y, too many `given` variables, and malformed `plot_string` are all rejected with clear messages.
- **Modern uncertainty API.** The `uncertainty`, `level`, `bands`, and `overlay` parameters are well-designed Python-native extensions.
- **Model comparison parity is close.** `model_comparison()` now handles non-nested models and returns `pred_difference`, matching R's behavior.
- **No `eval()` of user strings.** The Python port avoids R's string-then-eval pattern, which is a security and portability win.

---

## 6. Prioritized recommendation list

| Priority | Item | Effort |
|---|---|---|
| P0 | Fix `method="polynomial"` to be degree 2; make `"cubic"` degree 3 | Small |
| P0 | Implement or reject `spread="sterr"` (do not alias to bootstrap CI) | Small |
| P0 | Remove or implement `method="rlm"` and `method="glm"` | Small |
| P1 | Change categorical-x default `spread` from `"ci"` to `"quartiles"` to match R | Small |
| P1 | Implement `related=TRUE` or raise `NotImplementedError` | Medium |
| P1 | Add univariate `plot.type` options (`qq`, `density`, `boxplot`, `violin`) | Medium |
| P2 | Support formula transformations (`log(x)`, `poly(x,2)`, `I(...)`) | Medium |
| P2 | Implement R-style multivariate slotting for 2+ numeric predictors | Medium |
| P2 | Enable and run `rpy2`-based R parity tests in CI | Medium |
| P3 | Add factor pairwise differences / categorical standardized betas to `estimates()` | Medium |
| P3 | Implement `third.eye()` | Medium |

---

## 7. Verdict

`py-flexplot` v0.8.0 is **solid internally** but still **partial relative to R flexplot**. The most important next step is to eliminate the silent semantic mismatches (`polynomial` degree, `spread="sterr"`, `rlm`/`glm` fallback, and the default `spread`) because these will produce different plots/statistics without warning. Once those are fixed and the R parity tests run in CI, the remaining feature gaps (`related`, `third.eye`, formula functions, richer `plot.type`s) can be filled incrementally.
