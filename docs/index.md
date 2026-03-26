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

---

## Examples & Case Studies

- **[Titanic Survival Analysis](examples/titanic.md)** (Visual Walkthrough with Plots)
- **[Quickstart Notebook](../notebooks/quickstart.ipynb)** (Interactive Exploration)
- **[Biostatistics & Empirical Bayes](../examples/notebooks/stats_and_eb.ipynb)** (Data Cleanup and Shrinkage)
