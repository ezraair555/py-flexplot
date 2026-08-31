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
| Polynomial fit | `method="polynomial"` | ✅ | `method="polynomial"` | degree-2 OLS in x (R parity). |
| Cubic fit | `method="cubic"` | ✅ | `method="cubic"` | degree-3 OLS in x. |
| Quadratic fit alias | `method="quadratic"` | ✅ | `method="quadratic"` | Explicit degree-2 alias. |
| Formula transformations | `log(x)`, `sqrt(x)`, `exp(x)`, `poly(x,2)`, `I(...)` | ✅ | `formula` | Implemented with a safe whitelisted evaluator (v0.8.1). |
| Logistic fit | `method="logistic"` | ✅ | `method="logistic"` | GLM logit; explicit override routes through the parametric branch. |
| Robust / count / gamma smoothers | `method="rlm"`, `"poisson"`, `"Gamma"` | ✅ | same | Implemented in parametric smoother branch. |
| Binomial GLM on numeric binary y | implicit | ✅ | — | Auto-detected via binary pre-check (v0.6.1+). |
| Auto-bin numeric x | `bins=N` | ✅ | `bins=N` | Routes through `pd.cut`; v0.6.4. |
| Custom bin cuts | `breaks=[...]` | ✅ | `breaks=[...]` | Takes precedence over `bins` with `UserWarning`. |
| Custom bin labels | `labels=[...]` | ✅ | `labels=[...]` | Validated against `bins` / `breaks` length. |
| Dispersion marker | `spread=...` | ✅ | `{None,"ci","stdev","range","iqr","no"}` + R aliases `"quartiles"`/`"sterr"` (v0.8.0+) | `None` now maps to quartiles/IQR for discrete x (R parity). |
| Point jitter / alpha / raw-data toggle | `jitter`, `alpha`, `raw.data` | ✅ | `jitter=` / `alpha=` / `raw_data=` (v0.8.0+) | R's `se=F` ≈ our `uncertainty=None`; `suppress_smooth` likewise. |
| Multivariate numeric slotting | implicit | ✅ | `formula` + auto-binning | Slot-2+ / `given` numeric predictors are auto-binned to `<var>_binned` for color/facets (v0.8.1). |
| Low-cardinality numeric auto-categorical | implicit | ✅ | internal | Numeric predictors with `<5` unique values are auto-treated as categorical (R parity). |
| Subsample large data | `sample=N` | ✅ | `sample=N` | Subsamples plot only; fits use full data. Deterministic via `np.random.default_rng(0)`. |
| Overlay smoothers | `overlay=[...]` | ✅ | `overlay=[...]` | Per-overlay color / label / uncertainty / level. |
| Uncertainty: CI | implicit | ✅ | `uncertainty="ci"` | plotnine default. |
| Uncertainty: prediction | implicit | ✅ | `uncertainty="prediction"` | LM only. |
| Uncertainty: bootstrap | implicit | ✅ | `uncertainty="bootstrap"` | loess branch only; n=200. |
| Nested bands | `bands=[...]` | ✅ | `bands=[...]` | Multiple coverage levels. |
| Ghost line (panel-repeated, R-parity) | `ghost.line=<color>` | ✅ with facets | `ghost_line=<color>` + `ghost_reference={var: level}` (v0.8.0+) | Without facets, legacy Python-only y=0 / `"slope1"` references remain (`"slope1"` added v0.7.3). |
| Ghost reference data | `ghost.reference=df` | ✅ | DataFrame (overlay) or dict (panel selector, v0.8.0+) | Auto-detects scatter vs prediction-line by column shape. |
| Plot label override | `plot.string={...}` | ✅ | `plot_string={...}` | Accepts x, y, title, subtitle, caption, color. |
| Force plot type | `plot.type=...` | ✅ | `plot_type="scatter"\|"line"\|"boxplot"\|"bar"\|"histogram"\|"qq"\|"density"\|"violin"` | Includes univariate histogram/qq/density/boxplot/violin (v0.8.x). |
| Return data | `return.data=TRUE` | ✅ | `return_data=True` | Returns `{"plot", "data"}`. |
| Related-samples view | `related=TRUE` | ✅ | `related=True` | Implemented as paired-difference plot (2-level predictor, equal group sizes). |
| R-style interaction syntax | `y ~ x*z` | ✅ | `formula` parser + `interaction_model=True` (v0.7.0+) | Parsed since v0.6.2; **default** fit is additive (parallel slopes per color group, `UserWarning` emitted). `interaction_model=True` fits the actual interaction term and overlays non-parallel per-color-group lines. |
| Mixed-effects models (linear) | `method="lmer"` / `method="mixedlm"` | ✅ | `method="lmer"` / `"mixedlm"` + `random_effects=` | Random-intercept/slope (limited to `x`) via statsmodels MixedLM. |
| Mixed-effects models (binomial) | `method="glmer"` | ⚠️ | `method="glmer"` + `random_effects=` | Implemented with `statsmodels.BinomialBayesMixedGLM` random-intercept path; not a full `lme4::glmer` drop-in. |
| Random forests | `method="rf"` | ✅ | `RFAdapter(estimator, ...)` | Use [`pyflexplot.ml.RFAdapter`](ml.md) to wrap a fitted sklearn estimator and pass it to `compare_fits()`. v0.6.7+. |
| Diagonal slope=1 reference | implicit | ✅ | `ghost_line="slope1"` | Added in v0.7.3. |

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
| R-style extra args accepted | ✅ (partial behavior) | `report_se`, `re`, `num_points`, `clusters` accepted (v0.8.1 parity stub). |

---

## `flexplot::third.eye()` — three-way interaction helper

| Feature | Status | Notes |
|---|---|---|
| Public API endpoint exists | ✅ | `pyflexplot.third_eye()` exported (v0.8.1). |
| Full `third.eye` behavior | ❌ | Placeholder currently raises `NotImplementedError`; use `flexplot(..., interaction_model=True)` as current alternative. |

---

## `fifer::model_comparison()` — fit statistics

| Feature | Status | Notes |
|---|---|---|
| AIC | ✅ | |
| BIC | ✅ | |
| Log-likelihood | ✅ | |
| Likelihood-ratio test p-value | ✅ | Returns `(DataFrame, p_value)`; `p_value=None` for non-nested pairs (v0.8.0+, R parity). |
| pred.difference (prediction-difference quantiles) | ✅ | `model_comparison(..., return_pred_difference=True)` (v0.8.0+). |
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
| Factor-level estimates (per-level CIs) | ✅ | `estimates()["factor_estimates"]` (v0.8.0+). |
| Mean differences (pairwise contrasts, Cohen's d) | ✅ | `estimates()["mean_differences"]` (v0.8.0+). |
| mc= parameter | ✅ | `estimates(model, mc=False)` gates comparison outputs (v0.8.0+). |
| Factor vs numeric split | ✅ | v0.6.3+; deduplicated v0.8.0+. |
| Formula echo | ✅ | v0.6.3+. |
| Per-term partial η²_p (type-III SS) | ✅ | `eta_squared(model, typ=3)` (v0.7.5+). |
| Per-term CI on η²_p | ✅ | v0.7.5+ via the same non-central-F inversion. |

---

## `fifer::meansplot()` — descriptive visualizations

| Feature | Status | Python equivalent | Notes |
|---|---|---|---|
| Mean + SE / SD error bars | ✅ | `pyflexplot.meansplot(error="se"\|"sd")` | v0.7.4+. |
| CI on the mean (t-distribution) | ✅ | `pyflexplot.meansplot(error="ci")` | |
| Range / IQR error bars | ✅ | `pyflexplot.meansplot(error="range"\|"iqr")` | |
| Connecting line between means | ✅ | `pyflexplot.meansplot(connect=True)` | Always gray dashed in the Python port. |
| Color term | ❌ | — | Rejected explicitly; not in scope. |
| `\| given` facets | ❌ | — | Rejected explicitly; not in scope. |

---

## `flexplot::scatter3D()` — 3D scatter

| Feature | Status | Python equivalent | Notes |
|---|---|---|---|
| True 3D rendering (rgl) | ❌ | — | rgl is R-only; no Python port. |
| 2D scatter projection (x, z) colored by y | ✅ | `pyflexplot.scatter3D(type="points")` | v0.7.5+. |
| 2D heatmap projection (binned y) | ✅ | `pyflexplot.scatter3D(type="tile")` | v0.7.5+. |

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

- **`lme4` / `glmer` parity caveat** — py-flexplot now supports mixed models
  directly (`method="mixedlm"|"lmer"|"glmer"` + `random_effects=`), but this
  is still not a complete `lme4` clone. Current limits:
  - one grouping factor per fit;
  - random formula support is intentionally narrow (`(1|g)` / `(1 + x|g)`);
  - `glmer` is implemented through `BinomialBayesMixedGLM` (Bayesian VB fit),
    so estimates may differ from `lme4::glmer` MLE/REML behavior.
  If you need stricter `lme4`-matching behavior, use a bridge backend such as
  `pymer4` (R `lme4` under the hood via `rpy2`).
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
