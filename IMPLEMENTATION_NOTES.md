# v0.8.x Parity Implementation Notes

## What was implemented (post-review)

The following R-flexplot parity features were added to `src/pyflexplot/core.py` after the v0.8.0 review:

1. **Formula-function transformations** (`log(x)`, `sqrt(x)`, `poly(x, 2)`, `I(x**2)`, etc.)
   - Pre-processes the formula before parsing, applies a safe whitelisted evaluator (numpy/pandas/math functions plus `I()` and `poly()`), stores the transformed values in a column named after the inner variable (overwriting it, matching R), and rewrites the formula so downstream code sees the simpler variable name.
   - Splits additive terms in a parenthesis-aware way so `I(x ** 2 + 1)` is not broken on the internal `+`.
   - Given/panel variables after `|` are also transformed.

2. **Multivariate numeric slotting** (`y ~ x1 + x2`, `y ~ x1 + x2 | g`)
   - Numeric predictors in slot 2+ or in `given` with more than `bins` unique values are auto-binned into `<var>_binned` and used as the color/group aesthetic or facet variable.
   - Formulas with 3+ non-given predictors raise a clear error (R limits display complexity).

3. **R-parity defaults**
   - `alpha`: when not explicitly set, 0.2 for categorical x, 0.5 for numeric x (R's `.99977` sentinel is mapped to `None`).
   - `jitter`: `None` + categorical x → `(0.2, 0)`; `None` + numeric x → `(0, 0)`; bool/numeric values follow R's `match_jitter_categorical()`.
   - Low-cardinality numeric predictors (`< 5` unique values) are auto-converted to categorical before plotting, matching R's `convert_if_less_than_five()`.

4. **Univariate / related / spread / method fixes** (already landed in the v0.8.0 working tree before this pass)
   - Intercept-only plots: histogram/qq/density/boxplot/violin.
   - `related=True` paired-difference plots.
   - `spread="sterr"`, `spread="quartiles"` aliases, IQR default for discrete x.
   - `method="rlm"`, `"poisson"`, `"Gamma"`, polynomial/quadratic/cubic degrees.

## Known remaining deltas from R

- `third.eye` was explicitly excluded by the user.
- `poly(x, k)` stores only the highest-degree raw column (`x^k`) under the inner variable name, not the full raw polynomial matrix. This matches the observable R behavior for plotting but differs from a model-matrix construction.
- Explicit per-variable `breaks=list(var=c(...))` and `labels=list(var=...)` for binned color/given variables are not yet implemented; only a single flat `breaks`/`labels` pair (or auto-binning) is supported.
- R's exact bin-label string format (`"[a,b)"`) may differ slightly from pandas `pd.cut` string output.
- `I()` currently allows only single-column arithmetic expressions; multi-column expressions like `I(x + y)` are rejected.

## Test status

- `pytest -q`: 507 passed, 4 skipped (keras/rpy2 optional deps), 13 warnings.
- New tests live in `tests/test_formula_transforms.py`.
