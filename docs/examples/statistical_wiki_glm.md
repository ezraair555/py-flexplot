# General Linear Model Examples

`py-flexplot` includes the three public example datasets used by the
[Simplistics General Linear Model chapter](https://simplistics.net/stats_modeling/the-general-linear-model.html#glm-approach):

- `load_avengers()` - 812 simulated fighters and battle outcomes
- `load_diet()` - 78 paired pre-treatment and six-week weight observations
- `load_exercise_data()` - 200 simulated exercise/therapy observations

The data are redistributed from Dustin Fife's
[`flexplot` R package](https://github.com/dustinfife/flexplot), which declares
GPL-2 licensing. The attribution and license are shipped beside the CSV files.
The Python package code remains under its own project license.

The loaders preserve the upstream R column names, including names such as
`pre.weight` and `therapy.type`. Rename columns when a Python-friendly name is
more convenient.

```python
from pyflexplot.datasets import load_avengers, load_diet, load_exercise_data

avengers = load_avengers()
diet = load_diet()
exercise = load_exercise_data()
```

## One-sample test as a GLM

The chapter tests whether the average IQ differs from 100. In Python,
`scipy.stats` provides the direct t-test, while `statsmodels` expresses the
same test as an intercept-only model.

```python
import statsmodels.formula.api as smf
from scipy import stats

avengers = load_avengers()
print(stats.ttest_1samp(avengers["iq"], 100))

avengers["iq_minus_100"] = avengers["iq"] - 100
one_sample = smf.ols("iq_minus_100 ~ 1", data=avengers).fit()
print(one_sample.summary())
```

## Independent and paired comparisons

A two-group comparison is an ordinary linear model with a categorical
predictor. Dots in R column names can be quoted with Patsy's `Q()` helper.

```python
from patsy import Q

independent = smf.ols('Q("ptsd") ~ C(Q("north_south"))', data=avengers).fit()
print(independent.summary())

# Paired test expressed as a one-sample model on within-person change.
diet = load_diet()
diet["change"] = diet["weight6weeks"] - diet["pre.weight"]
paired_as_glm = smf.ols("change ~ 1", data=diet).fit()
print(paired_as_glm.summary())
```

## ANOVA is a GLM

The one-way ANOVA is the same linear-model family with a categorical
predictor. Fit with `statsmodels`, then use `py-flexplot` to inspect the
conditional distribution and fitted relationship.

```python
exercise = load_exercise_data()
exercise = exercise.rename(columns={
    "therapy.type": "therapy_type",
    "weight.loss": "weight_loss",
})

anova = smf.ols("weight_loss ~ C(therapy_type)", data=exercise).fit()
print(anova.summary())

import pyflexplot

pyflexplot.flexplot("weight_loss ~ therapy_type", data=exercise)
print(pyflexplot.estimates(anova))
```

For planned pairwise follow-ups, use `statsmodels.stats.multicomp` or
`scipy.stats` with a multiplicity correction. The model and the visualization
should be treated as one analysis, not as separate descriptive and inferential
steps.

## Regression and nonlinear terms

The chapter's linear and quadratic examples map directly to Patsy formulas.

```python
avengers = load_avengers()
linear = smf.ols("agility ~ speed", data=avengers).fit()
quadratic = smf.ols("agility ~ speed + I(speed ** 2)", data=avengers).fit()

pyflexplot.flexplot("agility ~ speed", data=avengers)
pyflexplot.visualize(quadratic)
```

The current source dataset reproduces the chapter's regression slope and
R-squared closely (`speed` about 43.09 and R-squared about 0.239).

## Binary outcomes

Convert a binary outcome explicitly when using `statsmodels` logistic
regression. `py-flexplot` can then visualize the fitted relationship.

```python
avengers = load_avengers()
avengers["died_binary"] = avengers["died"].eq("yes").astype(int)
logistic = smf.logit("died_binary ~ agility", data=avengers).fit(disp=False)
print(logistic.summary())
pyflexplot.flexplot("died_binary ~ agility", data=avengers, method="logistic")
```

## Source-data caveat

The wiki page contains printed numerical output generated at an earlier point
in the R package's data history. The current upstream `avengers.csv` shipped
by `flexplot` is the reproducible source used here. Most examples retain the
same analysis structure, but a printed coefficient or p-value should not be
assumed identical when the upstream data snapshot has changed.
