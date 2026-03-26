# Empirical Bayes API (ebbr Port)

The `ebbr` module provides tools for Empirical Bayes shrinkage, especially useful for analyzing batting averages, click-through rates, or any success/total count data.

## `fit_beta_prior`

Fits a Beta distribution (α, β) to a set of observations using Maximum Likelihood Estimation (MLE).

```python
from pyflexplot import fit_beta_prior
prior = fit_beta_prior(successes=df["hits"], totals=df["at_bats"])
print(prior.alpha, prior.beta)
```

### Returns
- A `BetaPrior` dataclass containing the fitted parameters and convenience methods like `.mean`.

---

## `add_ebb_estimate`

Applies the fitted prior to individual observations to calculate posterior means (shrunken estimates) and 95% credible intervals.

```python
from pyflexplot import add_ebb_estimate
shrunken_df = add_ebb_estimate(df, success_col="hits", total_col="at_bats")
```

### New Columns
- `ebb_fitted`: The shrunken posterior mean.
- `ebb_low`: Lower bound of the 95% credible interval.
- `ebb_high`: Upper bound of the 95% credible interval.

---

## Example: Batting Averages

When a player has only 10 at-bats and 4 hits, their raw average is 0.400. `add_ebb_estimate` will shrink this toward the league average (determined by the prior) to provide a more stable estimate of their true talent.
