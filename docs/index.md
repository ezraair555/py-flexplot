# py-flexplot: Documentation

Welcome to the Python port of Dustin Fife's R statistical ecosystem. This package brings the power of `flexplot`, `fifer`, `flexplavaan`, and `ebbr` to Python, optimized for the "grammar of graphics" via `plotnine`.

> **Note on coverage:** This is a *partial* port, not a 1:1 translation. The
> R-flexplot surface that translates cleanly onto `plotnine` + `statsmodels`
> is implemented; some R-only features are deferred or unsupported. See
> **[Coverage vs R-flexplot](api/coverage.md)** for the full matrix.

## Modules

### 1. [Core Visualization (flexplot)](api/core.md)
Intelligent plotting that automatically chooses geoms based on variable types in your formula.
- `flexplot(formula, data)`
- `added_plot(formula, data)`
- `compare_fits(formula, data, model1, model2)`
- [`diagnose(formula, data)` (v0.6.0+)](api/quality.md) — auto data-quality diagnostics

### 1a. [Uncertainty Module (v0.4.0+)](api/uncertainty.md)
Helpers for confidence / prediction / bootstrap intervals. Used internally by `flexplot()` but also usable directly.

### 2. [Biostatistical Utilities (fifer)](api/stats.md)
Toolbox for data cleanup, formatting, and standard statistical reporting.
- `model_comparison(model1, model2)`
- `p_format(p_value)`
- `eliminated_columns(df, threshold)`
- `color_table(df)`

### 3. [Empirical Bayes (ebbr)](api/ebbr.md)
Shrinkage estimation for binomial data.
- `fit_beta_prior(successes, totals)`
- `add_ebb_estimate(df, success_col, total_col)`

### 4. [SEM Visualization (flexplavaan)](api/sem.md)
Visualizing latent variable models.
- `measurement_plot(model, latent_var, data)`

### 5. [Neural-Network Integration (flex_nn)](api/flex_nn.md)
Wrappers so fitted `torch` (or keras) networks can be used in
`compare_fits()` alongside statsmodels fits.
- `NeuralNetFit(model, response_var, predictor_names)`
- `permutation_importance(fit, X, y)`
- `prepare_torch_data(data, categorical_vars)`

### 6. [Synthetic Data Generation (bluepill)](api/bluepill.md)
Generate clustered mixed-model data with fixed + random effects,
interactions, and polynomial terms.  Ideal for teaching and demos.
- `mixed_model(fixed, random, sigma, clusters, n_per, vars)`
- `estimate_sd(mean, min, max, num_sds=3)`

### 7. [Machine-Learning Adapters (ml)](api/ml.md)
Thin adapters so scikit-learn estimators (`RandomForestRegressor`,
`RandomForestClassifier`, and any estimator with `.predict()`) can be
used with `compare_fits()` and the rest of the visualization surface.
- `RFAdapter(estimator, response_var, predictor_names)`
- `make_rf_adapter(estimator, data, response_var, predictor_names=None)`

> **Optional dependency:** scikit-learn is **not** declared in py-flexplot's
> `pyproject.toml` — install it separately when you want to use this
> module.

### 8. [Descriptive Visualizations (descriptives)](api/descriptives.md)
Port of R's `fifer::meansplot()` and a 2D projection of `flexplot::scatter3D()`.
- `meansplot(formula, data, error="se", level=0.95, connect=True)`
- `scatter3D(formula, data, type="points"|"tile", bins=20)`

---

## Examples & Case Studies

- **[Titanic Survival Analysis](examples/titanic.md)** (Visual Walkthrough with Plots)
- **[Diagnostic + Visualization Workflow](examples/diagnostics_workflow.md)** (v0.6.x: `diagnose()` + uncertainty bands + overlay)
- **[General Linear Model Examples](examples/statistical_wiki_glm.md)** (textbook datasets and GLM workflows)
- **[Quickstart Notebook](../notebooks/quickstart.ipynb)** (Interactive Exploration)
- **[Biostatistics & Empirical Bayes](../examples/notebooks/stats_and_eb.ipynb)** (Data Cleanup and Shrinkage)
- **[flex_nn + bluepill Walk-through](../examples/notebooks/flex_nn_example.ipynb)** (Neural-network wrappers and synthetic data generation)
