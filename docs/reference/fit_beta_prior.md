# Empirical Bayes Beta Prior • fit_beta_prior

## Description

`fit_beta_prior()` fits a beta prior distribution to binomial data using maximum likelihood estimation (MLE) or method of moments.

## Usage

```python
fit_beta_prior(successes, totals, method="mle")
```

## Arguments

| Argument | Description |
|----------|-------------|
| `successes` | Array-like of success counts |
| `totals` | Array-like of total trial counts |
| `method` | Estimation method: `"mle"` (default) or `"moments"` |

## Details

The beta-binomial model assumes that success rates follow a beta distribution with parameters α (alpha) and β (beta). This function estimates these parameters from observed data.

**Methods:**
- **MLE**: Maximizes the beta-binomial likelihood (more accurate)
- **Moments**: Uses method of moments (faster, good for initialization)

## Returns

A `BetaPrior` dataclass with attributes:
- `alpha`: Estimated alpha parameter
- `beta`: Estimated beta parameter
- `n_obs`: Number of observations
- `method`: Estimation method used
- `mean`: Prior mean (alpha / (alpha + beta))

## Examples

### Fit Beta Prior to Baseball Data

```python
import pandas as pd
from pyflexplot import fit_beta_prior

# Baseball batting data
df = pd.DataFrame({
    'player': ['A', 'B', 'C', 'D'],
    'hits': [45, 30, 15, 60],
    'at_bats': [150, 100, 50, 200]
})

# Fit beta prior
prior = fit_beta_prior(df['hits'], df['at_bats'])

print(f"Alpha: {prior.alpha:.3f}")
print(f"Beta: {prior.beta:.3f}")
print(f"Prior mean: {prior.mean:.3f}")
```

### Compare Estimation Methods

```python
# MLE estimation (default)
prior_mle = fit_beta_prior(df['hits'], df['at_bats'], method="mle")

# Method of moments
prior_mom = fit_beta_prior(df['hits'], df['at_bats'], method="moments")

print(f"MLE: alpha={prior_mle.alpha:.3f}, beta={prior_mle.beta:.3f}")
print(f"Moments: alpha={prior_mom.alpha:.3f}, beta={prior_mom.beta:.3f}")
```

## See Also

- [`add_ebb_estimate()`](add_ebb_estimate.html) - Add empirical Bayes estimates to data
- [py-ebbr](https://github.com/ezraair555/py-ebbr) - Full empirical Bayes package

## References

Ported from David Robinson's R `ebbr` package: https://cran.r-project.org/package=ebbr
