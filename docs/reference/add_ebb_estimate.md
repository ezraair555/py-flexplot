# Add Empirical Bayes Estimates • add_ebb_estimate

## Description

`add_ebb_estimate()` adds empirical Bayes estimates to a dataframe containing binomial data (successes and totals).

## Usage

```python
add_ebb_estimate(df, success_col, total_col, prior=None)
```

## Arguments

| Argument | Description |
|----------|-------------|
| `df` | Pandas DataFrame with binomial data |
| `success_col` | Column name for success counts |
| `total_col` | Column name for total trial counts |
| `prior` | Optional `BetaPrior` object (fitted automatically if not provided) |

## Details

Empirical Bayes estimation "shrinks" individual estimates toward the overall mean, providing more stable estimates especially for observations with small sample sizes.

The function adds three columns to the dataframe:
- `ebb_fitted`: Empirical Bayes estimate (posterior mean)
- `ebb_low`: Lower bound of 95% credible interval
- `ebb_high`: Upper bound of 95% credible interval

## Returns

A copy of the input DataFrame with added columns for empirical Bayes estimates.

## Examples

### Add EB Estimates to Baseball Data

```python
import pandas as pd
from pyflexplot import add_ebb_estimate

# Baseball batting data
df = pd.DataFrame({
    'player': ['A', 'B', 'C', 'D', 'E'],
    'hits': [45, 30, 15, 60, 5],
    'at_bats': [150, 100, 50, 200, 20]
})

# Add empirical Bayes estimates
df_with_eb = add_ebb_estimate(df, 'hits', 'at_bats')

print(df_with_eb)
```

### Compare Raw vs. EB Estimates

```python
df_with_eb = add_ebb_estimate(df, 'hits', 'at_bats')

# Calculate raw batting average
df_with_eb['raw_avg'] = df_with_eb['hits'] / df_with_eb['at_bats']

# Compare
print(df_with_eb[['player', 'raw_avg', 'ebb_fitted']])

# Notice how small samples are shrunk toward the mean
```

### Use Custom Prior

```python
from pyflexplot import fit_beta_prior, add_ebb_estimate

# Fit prior separately
prior = fit_beta_prior(df['hits'], df['at_bats'])

# Add estimates with custom prior
df_with_eb = add_ebb_estimate(df, 'hits', 'at_bats', prior=prior)
```

## See Also

- [`fit_beta_prior()`](fit_beta_prior.html) - Fit beta prior to data
- [py-ebbr](https://github.com/ezraair555/py-ebbr) - Full empirical Bayes package

## References

Ported from David Robinson's R `ebbr` package: https://cran.r-project.org/package=ebbr
