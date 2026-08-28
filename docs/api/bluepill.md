# Synthetic Data Generation (bluepill Port)

`pyflexplot.bluepill` is a Python port of Dustin Fife's
[`bluepill`](https://github.com/dustinfife/bluepill) R package ("An R
package for creating simulated dataset").  Two functions are exposed:

## `estimate_sd`

```python
from pyflexplot.bluepill import estimate_sd

# Recover a standard deviation from a known mean and a known min/max range.
# mean=10, range [5, 15], 3 SD wide -> sd = 5/3.
estimate_sd(10, 5, 15, num_sds=3)
# 1.6666...
```

Useful when designing a simulation and you have a target shape but no
variance handy.  The R default is `num_sds=3` (the range covers +/-3 SD
around the mean).

## `mixed_model`

```python
from pyflexplot.bluepill import mixed_model

df = mixed_model(
    fixed=[0.0, 0.2, 0.5, 0.3, 0.2],
    random=[0.1, 0.1, 0.0, 0.2, 0.1],
    sigma=0.3,
    clusters=15,
    n_per=[11, 3],
    vars={
        "depression": (10.0, 3.0, 0),
        "stress": (22.0, 7.0, 0),
        "life_events": ["no", "yes"],
        "parental_depression": ["no", "mild", "moderate", "severe"],
        "ses": (55.0, 15.0, 0),
        "therapist": [f"Dr. {chr(65 + i)}" for i in range(15)],
    },
    seed=42,
)
print(df.head())
```

Generates a synthetic data frame with the structural properties of a
mixed-effects model:

- *fixed* and *random* are length-`n+1` vectors where `fixed[0]` is the
  intercept coefficient and `random[0]` is its random-effect SD.
- Each cluster draws one coefficient vector from `N(fixed, random)` and
  replicates it across that cluster's rows.  `random[j] == 0` freezes a
  coefficient across clusters.
- Cluster sizes are drawn from a normal truncated at 1.
- *vars* maps each variable name to either `(mean, sd, digits)` for a
  continuous variable, or a sequence of category labels for a
  categorical variable.  The **last** entry is the cluster id and must
  be categorical with `clusters` unique levels.
- *interactions* and *polynomials* are optional dicts with keys `"from"`,
  `"to"`, `"coef"`.  Indices are 1-based over the predictor slots (the
  intercept does not count).

The first variable in *vars* is treated as the response and is computed
from the standardized `sum(coef_matrix * predictor_matrix)` plus
residual noise.

## Example: teaching a mixed-model class

```python
import pandas as pd
import statsmodels.formula.api as smf

df = mixed_model(
    fixed=[0.0, 0.2, 0.5, 0.3, 0.2],
    random=[0.1, 0.1, 0.0, 0.2, 0.1],
    sigma=0.3,
    clusters=15,
    n_per=[11, 3],
    vars={
        "depression": (10.0, 3.0, 0),
        "stress": (22.0, 7.0, 0),
        "life_events": ["no", "yes"],
        "parental_depression": ["no", "mild", "moderate", "severe"],
        "ses": (55.0, 15.0, 0),
        "therapist": [f"Dr. {chr(65 + i)}" for i in range(15)],
    },
    seed=42,
)

model = smf.mixedlm(
    "depression ~ stress + C(life_events) + C(parental_depression) + ses",
    data=df,
    groups=df["therapist"],
).fit()
print(model.summary())
```

The ground truth is known (the `fixed` vector), so this is a great
vehicle for teaching and for sanity-checking py-flexplot's visualization
paths.