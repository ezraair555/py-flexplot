# Coverage vs R-flexplot

This page tracks what `py-flexplot` covers vs the R packages it ports. It's
the honest answer to "is this a 1:1 port?" — **no, it's not**, and here's
exactly what's covered and what's not.

## Legend

| Symbol | Meaning |
|---|---|
| ✅ | Implemented in the Python port |
| ⚠️ | Partial: surface is present but with limitations or behavioral differences; see notes |
| ❌ | Not implemented (deferred or out of scope) |

---

## `flexplot::flexplot()` — formula-based visualization

| Feature | R arg | Status | Python arg | Notes |
|---|---|---|---|---|
| Formula syntax | `y ~ x + color \| given` | ✅ | `formula` | Same parser as R. |
| Numeric vs numeric (LM/loess) | implicit | ✅ | `method="auto"\|"lm"\|"loess"` | |
| Polynomial fit | `method="polynomial"` | ✅ | `method="polynomial"` | degree-3 OLS in x (matches R default). |
| Cubic fit | `method="cubic"` | ✅ | `method="cubic"` | Alias for `"polynomial"`. |
| Logistic fit | `method="logistic"` | ✅ | `method="logistic"` | GLM logit; explicit override routes through the parametric branch. |
| Binomial GLM on numeric binary y | implicit | ✅ | — | Auto-detected via binary pre-check (v0.6.1+). |
| Auto-bin numeric x | `bins=N` | ✅ | `bins=N` | Routes through `pd.cut`; v0.6.4. |
| Custom bin cuts | `breaks=[...]` | ✅ | `breaks=[...]` | Takes precedence over `bins` with `UserWarning`. |
| Custom bin labels | `labels=[...]` | ✅ | `labels=[...]` | Validated against `bins` / `breaks` length. |
| Dispersion marker | `spread=...` | ✅ | `spread={None,"ci","stdev","range","iqr","no"}` | Default `None` preserves bootstrap CI. |
| Subsample large data | `sample=N` | ✅ | `sample=N` | Subsamples plot only; fits use full data. Deterministic via `np.random.default_rng(0)`. |
| Overlay smoothers | `overlay=[...]` | ✅ | `overlay=[...]` | Per-overlay color / label / uncertainty / level. |
| Uncertainty: CI | implicit | ✅ | `uncertainty="ci"` | plotnine default. |
| Uncertainty: prediction | implicit | ✅ | `uncertainty="prediction"` | LM only. |
| Uncertainty: bootstrap | implicit | ✅ | `uncertainty="bootstrap"` | loess branch only; n=200. |
| Nested bands | `bands=[...]` | ✅ | `bands=[...]` | Multiple coverage levels. |
| Ghost reference line | `ghost.line="red"\|"dashed"` | ✅ | `ghost_line="red"\|"dashed"` | y=0 reference; diagonal slope=1 is v0.7.0. |
| Ghost reference data | `ghost.reference=df` | ✅ | `ghost_reference=df` | Auto-detects scatter vs prediction-line by column shape. |
| Plot label override | `plot.string={...}` | ✅ | `plot_string={...}` | Accepts x, y, title, subtitle, caption, color. |
| Force plot type | `plot.type="bar"` | ✅ | `plot_type="scatter"\|"line"\|"boxplot"\|"bar"` | Bypasses auto-dispatch. |
| Return data | `return.data=TRUE` | ✅ | `return_data=True` | Returns `{"plot", "data"}`. |
| Link related panels | `related=TRUE` | ⚠️ | `related=True` | No-op (plotnine already shares scales); accepted for R-parity. |
| R-style interaction syntax | `y ~ x*z` | ✅ | `formula` parser + `interaction_model=True` (v0.7.0+) | Parsed since v0.6.2; **default** fit is additive (parallel slopes per color group, `UserWarning` emitted). `interaction_model=True` fits the actual interaction term and overlays non-parallel per-color-group lines. |
| Mixed-effects models (`glmer`) | `method="glmer"` | ❌ | — | `statsmodels.MixedLM` is not a drop-in for `lme4`. **Deferred.** |
| Random forests | `method="rf"` | ✅ | `RFAdapter(estimator, ...)` | Use [`pyflexplot.ml.RFAdapter`](ml.md) to wrap a fitted sklearn estimator and pass it to `compare_fits()`. v0.6.7+. |
| Diagonal slope=1 reference | implicit | ❌ | — | v0.7.0 todo. |

---

## `flexplot::visualize()` — model visualization

| Feature | Status | Notes |
|---|---|---|
| Predicted-vs-observed scatter | ✅ | `plot='model'` (default). |
| Residual vs fitted | ✅ | `plot='residuals'` returns `{'rvf', 'hist'}`. |
| All-in-one layout | ✅ | `plot='all'` tries cowplot; falls back to dict if not. |
| Residual histogram | ✅ | Bundled in `plot='residuals'`. |

---

## `flexplot::compare_fits()` — overlay pre-fit models

| Feature | Status | Notes |
|---|---|---|
| Two-model overlay | ✅ | Legacy. |
| Return predictions | ✅ | `return_preds=True` (v0.6.3+). |
| Predict on response or link scale | ✅ | `pred_type="response"\|"link"` (v0.6.3+). |
| GLM support | ✅ | Predictions on the response scale (probability). |

---

## `fifer::model_comparison()` — fit statistics

| Feature | Status | Notes |
|---|---|---|
| AIC | ✅ | |
| BIC | ✅ | |
| Log-likelihood | ✅ | |
| Likelihood-ratio test p-value | ✅ | Returns `(DataFrame, p_value)`. |
| R² | ✅ | v0.6.3+. |
| Adjusted R² | ✅ | v0.6.3+. |
| Bayes factor (Kass & Raftery 1995) | ✅ | v0.6.3+, derived from BIC. |

The Python signature is `(DataFrame, p_value)` — a 2-tuple. R's
`flexplot::model.comparison()` returns a single `data.frame`. See
[`stats.md`](stats.md) for the precise return-shape contract.

---

## `fifer::estimates()` — effect-size reporter

| Feature | Status | Notes |
|---|---|---|
| R², adj.R², sigma, n | ✅ | v0.6.3+. |
| R² confidence interval | ⚠️ → ✅ | ✅ (v0.7.3+) | Olkin & Finn 1995 non-central-F inversion. Population R² CI is now a real tuple (was `None` placeholder since v0.6.3). |
| Coefficient table (estimate / SE / t / p / CI) | ✅ | v0.6.3+. |
| Standardized betas | ✅ | v0.6.3+. |
| Semi-partial R² | ✅ | Computed via reduced-model fits. |
| Factor vs numeric split | ✅ | v0.6.3+. |
| Formula echo | ✅ | v0.6.3+. |

---

## `flexplot::diagnose()` — data-quality diagnostics

| Feature | Status | Notes |
|---|---|---|
| Missingness summary | ✅ | v0.6.0+. |
| Cook's distance outlier flag | ✅ | Default `4/n`. |
| Ramsey RESET | ✅ | Functional-form misspecification. |
| Breusch-Pagan | ✅ | Heteroscedasticity. |

---

## Out-of-scope items (and why)

- **`lme4` / `glmer`** — statsmodels' `MixedLM` is *not* a drop-in. R's `lme4`
  uses ML/REML with explicit nested random-effects formulas (e.g.
  `(1|school) + (1|class)`); `statsmodels.MixedLM` only supports a single
  grouping variable per model and the parameterizations differ. Bridging
  cleanly would require either a thin wrapper that calls R via `rpy2`
  (heavy dep, defeats the purpose of a Python port) or a custom re-implementation
  using a different backend (e.g. `patsy` + `formulaic` for the random-effects
  syntax, plus a fitting backend). Neither fits within py-flexplot's scope.
- **`randomForest` / sklearn estimators** — covered via
  [`pyflexplot.ml.RFAdapter`](ml.md). Wraps any sklearn estimator with
  `.predict()` and exposes a uniform surface to `compare_fits()` and
  friends. scikit-learn is **not** a declared dependency of py-flexplot;
  install it separately to use this module.
- **`flexplavaan` for full SEM** — the SEM visualization surface in py-flexplot
  is intentionally minimal (`measurement_plot`, `hopper_plot`,
  `disturbance_plot`). Full SEM fitting is left to `semopy` or `lavaan` via
  `rpy2`; py-flexplot focuses on the visualization layer.

---

## Where to report gaps

Open a GitHub issue at <https://github.com/ezraair555/py-flexplot/issues>
with the R-flexplot feature you want ported and the closest existing Python
analog. For mixed-effects, please indicate whether you'd accept an `lme4`-via-
`rpy2` wrapper (heavy dep, but accurate) or a native statsmodels approach
(faster, but behavioral differences).