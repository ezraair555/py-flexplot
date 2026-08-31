# Independent Code Review: py-flexplot v0.2.1 (`2d491cf`)

**Reviewer:** Automated subagent review  
**Date:** 2026-08-28  
**Scope:** All source under `src/pyflexplot/` and all tests under `tests/`  
**Test suite status:** 112 passed, 1 skipped (keras) in 9.03s  

---

## 1. Critical Bugs

### BUG-1: Off-by-one column offset in `mixed_model()` — last predictor is always constant

**File:** `src/pyflexplot/bluepill.py:225`  
**Severity:** Critical — silent data corruption  

The output loop uses `enumerate(predictor_names[1:], start=1)` with column index `j + 1`:

```python
for j, name in enumerate(predictor_names[1:], start=1):
    col = predictor_matrix[:, j + 1] if predictor_matrix.shape[1] > j + 1 else np.zeros(total_n)
```

`predictor_names[1]` (j=1) maps to `predictor_matrix[:, 2]`, skipping column 1 entirely. The last predictor (j = n_pred-1) maps to column `n_pred`, which is out of bounds, falling through to `np.zeros(total_n)`. After rescaling, zeros become the variable's declared mean with zero variance.

**Failure mode:** The README's own example produces `ses = 55.0` for every row. Any mixed_model call with 2+ predictors has the last predictor as a constant and all other predictors shifted by one column. This means the generated data does NOT match the declared `vars` specification. The y_std computation is correct (it uses the full matrix product), but the output columns for individual predictors are wrong.

**Fix:** Change `j + 1` to `j`:
```python
col = predictor_matrix[:, j] if predictor_matrix.shape[1] > j else np.zeros(total_n)
```

### BUG-2: `permutation_importance()` crashes on named metrics without explicit scorer dispatch

**File:** `src/pyflexplot/flex_nn.py:437-460`  
**Severity:** Critical — runtime crash on documented API  

`_DEFAULT_METRICS` declares `auc`, `precision`, `recall`, `f1`, and `loss` as known metrics, but the scorer dispatch only defines scorer functions for `mse`, `mae`, `rmse`, `r2`, and `accuracy`. For any of the five declared-but-undispatched metrics, `scorer` is never assigned, and the function hits `UnboundLocalError: cannot access local variable 'scorer'`.

**Confirmed:** `permutation_importance(fit, X, y, metric='auc')` → `UnboundLocalError`.

**Failure mode:** Any user passing a string metric other than the five with explicit scorer branches gets a crash. The `if direction is None:` fallback block (lines 448-456) that could define a scorer is unreachable for named metrics because `direction` is always set from `_DEFAULT_METRICS`.

**Fix:** Either remove the five metrics from `_DEFAULT_METRICS` that have no scorer implementation, or add scorer dispatch branches for `auc`, `precision`, `recall`, `f1`, and `loss`.

### BUG-3: Categorical specs passed as tuples are misidentified as continuous

**File:** `src/pyflexplot/bluepill.py:359-366` (`_apply_spec`)  
**Severity:** High — crash on valid input per type hints  

`_apply_spec` checks `isinstance(spec, tuple)` first and routes to `_rescale_continuous`. But `VarSpec = Union[Tuple[float, float, int], Sequence[str]]` — a tuple of strings is a valid categorical spec per the type hint. `_rescale_continuous` then tries `float("no")` and crashes with `ValueError`.

Inconsistency: `_check_errors` correctly distinguishes continuous vs categorical tuples using `len(spec) == 3 and all(isinstance(x, (int, float)) for x in spec)`, but `_apply_spec` does not use the same logic.

**Failure mode:** `mixed_model(..., vars={"y": (10,3,0), "x": ("no", "yes"), "cluster": [...]})` crashes with `ValueError: could not convert string to float: 'no'`.

**Fix:** Use the same numeric-checking logic as `_check_errors` in `_apply_spec`, or require categoricals to be lists (and update the type hint).

---

## 2. Design Issues

### DESIGN-1: `estimates()` is a stub that doesn't deliver its docstring promise

**File:** `src/pyflexplot/stats.py:39-44`  

The docstring says "Reports effect sizes (e.g., Cohen's d, Eta-squared)" but the function just returns `model.summary()`. No effect sizes are computed. This is misleading to anyone reading the API surface.

### DESIGN-2: `flexplot()` both-non-numeric path is dead code

**File:** `src/pyflexplot/core.py:160-161`  

`_validate_data_for_plot()` rejects non-numeric `y` with a `ValueError`, but `flexplot()` has a branch for `not is_y_numeric and not is_x_numeric` that adds `geom_jitter(width=0.2, height=0.2)`. This branch can never be reached. Either remove the dead branch or relax the validator to allow string `y` in the both-categorical case.

### DESIGN-3: `flexplot()` silently ignores 3+ given variables

**File:** `src/pyflexplot/core.py:155-158`  

The formula parser accepts `y ~ x | a + b + c`, but `flexplot()` only handles `len(given) == 1` (facet_wrap) and `len(given) >= 2` (facet_grid with `given[0]` and `given[1]`). Variables `given[2:]` are silently dropped with no warning.

### DESIGN-4: `flexplot()` silently ignores unknown `method` values

**File:** `src/pyflexplot/core.py:131-137`  

`method='auto'` and `method='lm'` add a linear smooth; `method='loess'` adds a loess smooth; any other string silently produces no smooth. No error or warning. A typo like `method='loes'` would be invisible.

### DESIGN-5: `_keras_predict()` permanently mutates `model.training` without restoring

**File:** `src/pyflexplot/flex_nn.py:298-300`  

The method sets `self.model.training = False` before prediction but never restores the original value. This is a side effect on a caller-owned object that persists after `predict()` returns.

### DESIGN-6: `polynomials` dict requires a `to` key that is ignored

**File:** `src/pyflexplot/bluepill.py:248-253, 387-396`  

The `_interaction_polynomial_checks` function requires all three keys `from`, `to`, `coef` for both interactions and polynomials. But `_add_polynomials` ignores `to` entirely. Users must provide a meaningless `to` value for polynomials.

### DESIGN-7: `visualize()` does not support `NeuralNetFit`

**File:** `src/pyflexplot/core.py:91-120`  

`visualize()` looks for `model.model.endog_names` and `model.model.data.orig_endog`, which are statsmodels-specific attributes. `NeuralNetFit` has a `.model` attribute (the underlying torch/keras module) but no `endog_names`. The README only claims `compare_fits` integration, but `visualize` being incompatible is an asymmetry a user would hit.

### DESIGN-8: `from plotnine import *` pollutes module namespaces

**Files:** `src/pyflexplot/core.py:3`, `src/pyflexplot/sem.py:2`  

`core` exports 245 names from plotnine. This is conventional in plotnine-using scripts but a lint smell in library code. Prefer explicit imports: `from plotnine import ggplot, aes, geom_point, ...`.

### DESIGN-9: Unused import: `patsy` in `core.py`

**File:** `src/pyflexplot/core.py:4`  

`import patsy` is at the top of the file but `patsy` is never referenced anywhere in the module.

---

## 3. Test Coverage Gaps

### GAP-1: No test verifies that `mixed_model` predictors have non-zero variance

**Risk:** Silent data corruption (BUG-1)  
The existing tests check that categorical columns use only declared levels and that continuous columns are "bounded by spec" (within 5 SD). But no test checks that each predictor column has non-zero variance, which would have caught the off-by-one bug. The `test_continuous_columns_are_bounded_by_spec` test passes because `stress` (which maps to the wrong column) still has the right mean and SD — it's just the wrong underlying data. `ses` being constant at 55.0 passes because the test only checks `mean ≈ 55, std ≈ 7 (abs=5)` and `std=0` is within `7 ± 5`.

### GAP-2: No test exercises `permutation_importance` with metrics other than `mse` and custom callables

**Risk:** Crash (BUG-2)  
The test suite only tests `metric='mse'` and `metric=lambda ...`. No test passes `metric='auc'`, `metric='mae'`, `metric='r2'`, or `metric='accuracy'`. BUG-2 would have been caught by a single test using `metric='auc'`.

### GAP-3: No test exercises categorical `vars` entries as tuples

**Risk:** Crash (BUG-3)  
All test code uses lists for categorical specs (`["no", "yes"]`). The type hint permits tuples for categoricals but no test verifies it.

### GAP-4: `permutation_importance` importance ranking is not validated for correctness

The tests check that results are sorted descending and that the DataFrame has the right columns, but no test verifies that the most important variable is actually the one with the strongest effect. The test builds `y = 2.0 * X[:, 0] + 0.1 * X[:, 1] + ...` but never asserts `result.iloc[0]["variable"] == "x1"`.

### GAP-5: `compare_fits` with NeuralNetFit is tested only in one direction (OLS first, NN second)

The reversed order is tested above in this review but not in the test suite. The test suite has a single integration test (`test_compare_fits_accepts_neural_net_fit`).

### GAP-6: No test for `flexplot()` with numeric color variable and smooth

When color is numeric, `geom_smooth` tries to fit one line per unique value, producing dozens of "Smoothing requires 2 or more points" warnings and likely wrong visualization. No test covers this case.

### GAP-7: No test for `added_plot` with >2 predictors

`added_plot("y ~ x + z + w", data=df)` is never tested. The residualization logic for multiple "other" variables is only tested with one other variable.

### GAP-8: Tests use random data without fixed seeds in some cases

Several tests in `test_core.py` use `np.random.normal(size=100)` without setting a seed. While the assertions only check types (isinstance ggplot), this makes tests non-deterministic and harder to reproduce if a failure is intermittent.

### GAP-9: `test_interactions_add_columns` tests that column structure is unchanged but doesn't verify interaction effects

The test correctly notes that interactions only affect latent y_std, not column layout. But it doesn't verify that the interaction actually changed the response variable's distribution. A regression test comparing `y` with and without interactions (same seed) would verify the interaction code path runs.

### GAP-10: No test for `estimates()` or `color_table()`

`test_stats.py` tests `model_comparison`, `p_format`, and `eliminated_columns` but never calls `estimates()` or `color_table()`. Since `estimates()` is just a pass-through to `model.summary()`, this is low-risk, but `color_table()` is completely untested.

---

## 4. Security / Safety

**No security issues found.**

- No `eval()`, `exec()`, `pickle`, `subprocess`, or `os.system` calls anywhere in the source.
- No filesystem operations beyond standard pandas/numpy.
- `prepare_torch_data` explicitly rejects NaN rather than silently imputing — good safety posture.
- `set_response_var` uses `setattr` on a model object — this is standard for attaching metadata to torch/keras models and not a security concern.
- `mixed_model` uses `np.random.default_rng(seed)` — no global RNG state mutation.
- The `_keras_predict` method mutates `model.training` (DESIGN-5) — not a security issue but a correctness concern covered above.

---

## 5. Style / Lint Smells

### STYLE-1: `from plotnine import *` in `core.py` and `sem.py`

Imports ~200 names into each module's namespace. Conventional for plotnine scripts but inappropriate for library code. Use explicit imports.

### STYLE-2: Unused import `patsy` in `core.py:4`

Dead import that suggests a planned feature that was never implemented or was removed.

### STYLE-3: `estimates()` has no type hints, no validation, and a misleading docstring

```python
def estimates(model):
    """Reports effect sizes (e.g., Cohen's d, Eta-squared) for statistical models."""
    summary = model.summary()
    return summary
```

No return type hint, no input validation, and the docstring promises something the code doesn't deliver.

### STYLE-4: Inconsistent type hint coverage across modules

- `core.py`: Has type hints on public functions but internal helpers like `_first_non_intercept_name` have none.
- `flex_nn.py`: Excellent type hint coverage (best in the project).
- `bluepill.py`: Good type hints on public functions, none on internal helpers.
- `sem.py`: Minimal type hints (only `var1: str, var2: str` on `disturbance_plot`).
- `stats.py`: No type hints at all.

### STYLE-5: Dead code block in `permutation_importance` (lines 448-456)

The `if direction is None:` block defines a scorer but `direction` is always set for both named and callable metrics before reaching it. This block can never execute.

### STYLE-6: `flexplot()` method parameter accepts any string without validation

`method='auto' | 'lm' | 'loess'` is documented in the signature but not enforced. Any string is accepted; unknown strings silently produce no smooth.

### STYLE-7: `_add_polynomials` ignores the `to` field but `_interaction_polynomial_checks` requires it

The validation function is shared between interactions and polynomials, but polynomials don't use `to`. Either split the validation or document that `to` is required but ignored for polynomials.

### STYLE-8: `model_comparison()` doesn't validate that models are nested

The docstring says "compares the fits of two nested statsmodels results" but doesn't check nesting — it only checks that `df_diff > 0` after the fact. A non-nested model pair with coincidentally positive df_diff would produce a meaningless p-value without warning.

### STYLE-9: Docstring coverage is good but inconsistent

`flex_nn.py` has excellent docstrings with Examples section. `bluepill.py` has good docstrings. `core.py` docstrings are one-liners with no parameter documentation. `sem.py` docstrings are minimal. `stats.py` is bare.

---

## 6. Suggested Improvements, Ranked

### 1. Fix the off-by-one column offset in `mixed_model()` (BUG-1)
**Why:** Silent data corruption in the headline feature of the bluepill module. The README example is broken. Every call with 2+ predictors produces wrong data.  
**Effort:** Small — one-line fix, but also add a test that checks each predictor has non-zero variance.

### 2. Fix `permutation_importance()` scorer dispatch for all declared metrics (BUG-2)
**Why:** Documented API crashes on 5 of the 11 declared metric names. Any user who passes `metric='auc'` gets an `UnboundLocalError`.  
**Effort:** Small — add scorer branches for `auc`, `precision`, `recall`, `f1`, `loss`, or remove them from `_DEFAULT_METRICS`.

### 3. Fix `_apply_spec` to handle tuple categoricals correctly (BUG-3)
**Why:** Valid input per type hints crashes. Inconsistency between validation logic and execution logic.  
**Effort:** Small — reuse the numeric-checking logic from `_check_errors` or require lists for categoricals.

### 4. Replace `estimates()` stub with a real implementation or mark it as experimental
**Why:** A function that claims to compute effect sizes but returns a summary table is misleading. Either implement it (Cohen's d, eta-squared from an OLS model is ~20 lines) or add a `# TODO: not yet implemented` note and update the docstring.  
**Effort:** Medium (implement) or Small (mark as stub).

### 5. Add tests that verify correctness, not just structure
**Why:** The existing tests verify that functions return the right types and don't crash, but rarely verify that the results are correct. BUG-1 and BUG-2 both pass the existing test suite. Add: (a) predictor variance check for `mixed_model`, (b) `permutation_importance` ranking correctness test asserting the strongest variable ranks first, (c) tests for all declared metric strings.  
**Effort:** Medium — ~10-15 new test cases across the two modules.

---

## File-by-File Summary

| File | Lines | Verdict |
|------|-------|---------|
| `__init__.py` | 33 | Clean. Well-organized exports. |
| `core.py` | 412 | Solid. Dead code in both-non-numeric path. Unused `patsy` import. `from plotnine import *` smell. |
| `flex_nn.py` | 525 | Best-written module. BUG-2 (scorer dispatch). Dead code block. `_keras_predict` side effect. |
| `bluepill.py` | 403 | BUG-1 (off-by-one, critical), BUG-3 (tuple categoricals). `explained` formula is an R-source quirk. Polynomials `to` key wasted. |
| `sem.py` | 183 | Clean. Good defensive handling of semopy API. Minimal docstrings. |
| `stats.py` | 91 | `estimates()` is a stub. No type hints. Otherwise functional. |
| `ebbr.py` | 149 | Clean. Good validation. Well-tested. No issues found. |
| `tests/` (9 files) | 1441 | Good breadth but test correctness, not implementation correctness. Key gaps: no variance check for bluepill, no metric dispatch tests for flex_nn. |

---

**Bottom line:** Three critical bugs (one silent data corruption, one crash on documented API, one crash on type-hint-valid input). All three are small fixes. The codebase is well-structured for a portfolio piece, but the test suite is testing that the code runs, not that it's right. The `flex_nn` module is the strongest; `bluepill` has the most issues; `core`/`sem`/`ebbr`/`stats` are solid.