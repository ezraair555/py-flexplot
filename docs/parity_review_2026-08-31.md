# Port-Parity Review: R-flexplot → py-flexplot

**Date:** 2026-08-31
**Reviewer:** glm-3.5-flash serving (current session), at John's request.
**R reference:** `dustinfife/flexplot` @ master — NAMESPACE + roxygen man pages
(`flexplot.Rd`, `estimates.Rd`, `model.comparison.Rd`, `added.plot.Rd`,
`visualize.Rd`, `compare_fits.Rd`), fetched 2026-08-31.
**Python baseline:** py-flexplot v0.7.5 (`e385df3`), 452 tests passing.

Two prior internal audits (v0.6.2-era + coverage.md) existed. This review
was done fresh against the R package's own NAMESPACE/man pages rather than
against our memory of it. It found **real gaps the earlier audits missed.**

---

## 1. Primary-function verdict (the six exported "primary" functions)

| R function | Python | Verdict |
|---|---|---|
| `flexplot()` | `flexplot()` | ⚠️ Surface mostly ported; several parameter gaps below |
| `added.plot()` | `added_plot()` | ⚠️ Present but 3 semantic differences (below) |
| `visualize()` | `visualize()` | ✅ Full parity (`plot=` model/residuals/all; `formula` param present) |
| `compare.fits()` | `compare_fits()` | ⚠️ 2 params missing (`report.se`, `num_points`) |
| `estimates()` | `estimates()` | ⚠️ Missing 3 output components (below) |
| `model.comparison()` | `model_comparison()` | ⚠️ 2 gaps (non-nested models, `pred.difference`) |

---

## 2. flexplot() parameter-level parity

### Ported ✅ (semantics verified against R docs)
- `formula` (y ~ x + color | given, interaction `*`/`:`)
- `bins` / `labels` / `breaks`, `sample`, `related` (name only — see drift),
  `return_data`, `ghost.reference` (form differs, see drift).

### Token/semantic drift ⚠️
- **`spread` tokens differ.** R: `c("quartiles", "stdev", "sterr")`, default
  **quartiles**. Ours: `{None,"ci","stdev","range","iqr","no"}`, default
  None (bootstrap CI). `quartiles` ≈ our `iqr`; `sterr` ≈ our `ci`. Should
  accept R's tokens as aliases and re-point the default (or document).
- **`ghost.line` semantics differ (largest single gap).** R's ghost line
  *repeats the fit line from one panel into the others* to aid cross-panel
  comparison; color is any ggplot color string (e.g. `"black"`, `"red"`).
  Ours draws a y=0 / slope=1 reference *within* the plot. Different feature
  sharing a name. Requires faceting support (re-fit a reference group and
  overlay its line into every panel).
- **`ghost.reference` form differs.** R: `list("health"=31, "income"=90000)`
  (variable-level references). Ours: a DataFrame overlay. Same idea
  (reference data), different interface.
- **`plot.string` collides.** R's `plot.string` (logical) *returns the
  generated ggplot code*. Ours (`plot_string`, dict) overrides axis labels.
  Intentional repurpose, but worth a docstring warning.
- **`related` semantics differ.** R: plots *difference scores* for paired
  designs. Ours: documented no-op (scale sharing). Genuine gap.
- **`method` default/members differ.** R default is **"loess"**; ours
  "auto"→lm (documented choice). R also accepts `"rlm"` and `"glm"` as
  primary methods; ours only offers them in `overlay=`.
- **`se=FALSE`** maps to ours `uncertainty=None`; `suppress_smooth=TRUE`
  likewise. Different names, same effect — acceptable but worth a
  cross-reference table in the docs.

### Missing parameters ❌
- `jitter` (bool or length-2 vector) — we hardcode `geom_jitter(width=0.2)`.
- `raw.data` (show/hide raw points over summaries) — we always draw points.
- `alpha` (point transparency) — we hardcode 0.5/0.3/0.4.
- `silent` (suppress messages) — minor; N/A-ish.
- `bins` as a **list per variable** (R: `list(income=c(95000,...))`) — ours
  is one int for the primary x only. Multi-binned `y ~ x + z` plots are
  R-flexplot's "2N/3N" feature and our binning applies to x only.
- `plot.type` options: R has `histogram/qq/density/boxplot/violin/line`;
  ours has `scatter/line/boxplot/bar`. Missing `qq`, `density`, `violin`
  (the univariate histogram exists via intercept-only dispatch).

---

## 3. model_comparison() gaps ❌

R's `model.comparison()` works for **non-nested** models too (AIC/BIC/BF
need no nesting); ours raises `ValueError` on non-positive df diff because
the LRT p-value requires nesting. Also, R returns `list(statistics=...,
pred.difference=...)` — the `pred.difference` component (quantiles of the
two models' prediction differences) has **no Python equivalent at all**.

Recommended fix split:
1. `pred.difference` — new second/third element of the return tuple.
2. Non-nested support: keep LRT p-value `None` when not nested instead of
   raising; still emit AIC/BIC/BF/R².

## 4. estimates() output gaps

R's `estimates()` (per README output) has four components; we have two:
- ✅ Model R² (with CI — ours now real, v0.7.3+)
- ✅ Semi-partial R² per predictor
- ❌ **Estimates for Factors** — per-level intercept estimates with
  lower/upper per factor level
- ❌ **Mean Differences** — pairwise contrast table with `cohens.d`
- ❌ `mc=` parameter ("perform model comparisons?") — related to the above

Cohen's d for pairwise factor contrasts is the most standard piece of
these; factor-level estimates need contrast coding choices.

## 5. added_plot() gaps

- R plots the **last** variable against y-residualized-on-the-others by
  default ("y~x + z → residualize y~x, plot z"). Ours plots the **first**.
  At minimum this belongs as a documented difference; better, adopt R's
  direction (or add `x=` to select).
- ❌ `lm_formula=` — custom conditioning model for multivariate AVPs.
- ❌ `x=` — choose which variable to residualize/plot.
- R adds the mean of y back onto residuals ("to maintain interpretation");
  ours plots raw residuals. Cosmetic-ish but changes axis interpretation.

## 6. compare_fits() gaps
- ❌ `report.se=` (SEs alongside prediction table)
- ❌ `num_points=` (grid resolution; ours hardcodes 200 in manual-smooth
  paths, plotnine defaults elsewhere)
- `re=` / `clusters=` — mixed-model only; consistent with lme4 deferral.

## 7. NAMESPACE-level: functions not ported at all (39 exports total)

Intentionally out-of-scope (documented in coverage.md):
`lme4`-backed (`mixed.mod.visual`, `icc`, `cluster_adjusted_scatter`,
`compare_fits(re=)`), JASP internals (`*_jasp`, `flexplot_jasp2`), R-geom
internals (`geom_jitterd`, `position_jitterd*`), internal plumbing
(`get_fitted`, `get_terms`, `prepare_data_for_compare_fits`,
`post_prediction_process_cf`), and stubs (`third.eye` — R docs say "will
be implemented shortly").

**Real gaps (user-facing, small-to-medium):**
- `anchor.predictions` — anchored prediction plots (~1 day)
- `marginal_plot` — marginal effects plots (~1 day)
- `magnet_plot` — model-choice visualization (~1 day)
- `third.eye` — comprehensive diagnostic panel (multi-day; R also marks
  it unimplemented, so parity pressure is low)
- `univariate_list` — histogram grid for all variables (~half day)
- `rsq_change` — standalone (we compute it inside estimates()); trivial
- `standardized.beta` — standalone accessor; trivial
- `bf.bic` — standalone Bayes-factor-from-BIC; trivial (we compute it
  inside model_comparison)
- `sensitivity.table`, `floor_ceiling`, `rescale`, `flip_data`,
  `make.formula`, `modify_points`, `partial_residual_plot`,
  `logistic_*` suite (5 fns), `mediate_plot` — scattered utils and
  logistic-diagnostic suite; mostly small, some (partial_residual_plot)
  close to existing `added_plot`.

---

## 8. Bottom line

- **Coverage of the 6 primary functions:** 2 full (visualize), 4 with
  parameter/output-level gaps.
- **Highest-leverage remaining work, ranked:**
  1. `model_comparison` non-nested support + `pred.difference` (~1 day)
  2. `estimates()` factor-level estimates + mean-differences w/ Cohen's d (~1–2 days)
  3. `added_plot()` alignment with R (direction, `x=`, `lm_formula=`, mean-offset) (~1 day)
  4. R-token aliases for `spread` (`quartiles`, `sterr`) + `jitter`/`alpha`/`raw.data` params (~half day)
  5. True ghost.line (panel-to-panel repeated reference line) (~1 day, facet-aware)
  6. Small standalone accessors: `standardized.beta`, `rsq_change`, `bf.bic` (~half day combined)
- **Honesty note:** coverage.md previously over-stated parity for the six
  primary functions. This review supersedes those rows.

Source URLs (fetched 2026-08-31):
- https://raw.githubusercontent.com/dustinfife/flexplot/master/NAMESPACE
- .../man/flexplot.Rd, estimates.Rd, model.comparison.Rd, added.plot.Rd,
  visualize.Rd, compare_fits.Rd