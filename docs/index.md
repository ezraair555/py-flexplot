# py-flexplot: Documentation

Welcome to the unified Python port of Dustin Fife's R statistical ecosystem. This package brings the power of `flexplot`, `fifer`, `flexplavaan`, and `ebbr` to Python, optimized for the "grammar of graphics" via `plotnine`.

## Modules

### 1. [Core Visualization (flexplot)](api/core.md)
Intelligent plotting that automatically chooses geoms based on variable types in your formula.
- `flexplot(formula, data)`
- `added_plot(formula, data)`
- `compare_fits(formula, data, model1, model2)`

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

---

## Examples & Case Studies

- **[Titanic Survival Analysis](examples/titanic.md)** (Visual Walkthrough with Plots)
- **[Diagnostic + Visualization Workflow](examples/diagnostics_workflow.md)** (v0.6.x: `diagnose()` + uncertainty bands + overlay)
- **[Quickstart Notebook](../notebooks/quickstart.ipynb)** (Interactive Exploration)
- **[Biostatistics & Empirical Bayes](../examples/notebooks/stats_and_eb.ipynb)** (Data Cleanup and Shrinkage)
- **[flex_nn + bluepill Walk-through](../examples/notebooks/flex_nn_example.ipynb)** (Neural-network wrappers and synthetic data generation)
