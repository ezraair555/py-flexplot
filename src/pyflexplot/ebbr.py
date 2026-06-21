"""Empirical Bayes binomial estimation (ported from ebbr)."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln
from scipy.stats import beta as beta_dist
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List


@dataclass
class BetaPrior:
    alpha: float
    beta: float
    n_obs: int
    method: str = "mle"

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


def _beta_binomial_loglik(params, successes, totals):
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return np.inf
    log_coeff = gammaln(totals + 1) - gammaln(successes + 1) - gammaln(
        totals - successes + 1
    )
    ll = (
        log_coeff
        + betaln(successes + alpha, totals - successes + beta)
        - betaln(alpha, beta)
    )
    return -np.sum(ll)


def fit_beta_prior(successes, totals, method="mle"):
    """
    Fit a Beta prior to observed binomial counts via MLE or method of moments.

    Validates that ``0 <= successes <= totals`` and ``totals > 0``. Guards
    against zero variance in the observed rates and verifies that the
    optimizer converged to finite parameters.
    """
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)

    if successes.ndim != 1 or totals.ndim != 1:
        raise ValueError("successes and totals must be one-dimensional arrays/Series")
    if len(successes) != len(totals):
        raise ValueError(
            f"successes and totals must have the same length: "
            f"{len(successes)} vs {len(totals)}"
        )
    if len(successes) == 0:
        raise ValueError("successes and totals must not be empty")

    if np.any((successes < 0) | (successes > totals)):
        bad = np.where((successes < 0) | (successes > totals))[0]
        raise ValueError(
            f"successes must satisfy 0 <= successes <= totals. Bad indices: {bad[:10].tolist()}"
        )
    if np.any(totals <= 0):
        bad = np.where(totals <= 0)[0]
        raise ValueError(f"totals must be positive. Bad indices: {bad[:10].tolist()}")
    if np.any(np.isnan(successes)) or np.any(np.isnan(totals)):
        raise ValueError("successes and totals must not contain NaN")

    rates = successes / totals
    m = rates.mean()
    v = rates.var()

    # Guard zero variance: all rates identical => no information to estimate a
    # beta prior. Return a weakly informative prior rather than NaN/Inf seeds.
    if v == 0:
        if method == "moments":
            return BetaPrior(1.0, 1.0, len(successes), "moments")
        return BetaPrior(1.0, 1.0, len(successes), "mle")

    common = m * (1 - m) / v - 1
    alpha0 = m * common
    beta0 = (1 - m) * common

    # Clip method-of-moments seeds to avoid negative or extreme initial values.
    alpha0 = max(1e-6, min(alpha0, 1e6))
    beta0 = max(1e-6, min(beta0, 1e6))

    if method == "moments":
        return BetaPrior(alpha0, beta0, len(successes), "moments")

    res = minimize(
        _beta_binomial_loglik,
        x0=[alpha0, beta0],
        args=(successes, totals),
        bounds=((1e-6, None), (1e-6, None)),
    )

    if not res.success:
        raise RuntimeError(
            f"Beta prior optimization did not converge: {res.message}"
        )

    alpha_hat, beta_hat = res.x
    if not (np.isfinite(alpha_hat) and np.isfinite(beta_hat) and alpha_hat > 0 and beta_hat > 0):
        raise RuntimeError(
            f"Beta prior optimization returned invalid parameters: alpha={alpha_hat}, beta={beta_hat}"
        )

    return BetaPrior(alpha_hat, beta_hat, len(successes), "mle")


def add_ebb_estimate(df, success_col, total_col, prior=None):
    """
    Add empirical-Bayes beta-binomial shrinkage estimates to a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if success_col not in df.columns:
        raise ValueError(f"success_col {success_col!r} not found in DataFrame")
    if total_col not in df.columns:
        raise ValueError(f"total_col {total_col!r} not found in DataFrame")

    successes = pd.to_numeric(df[success_col], errors="coerce")
    totals = pd.to_numeric(df[total_col], errors="coerce")

    if successes.isna().any() or totals.isna().any():
        raise ValueError(
            f"{success_col!r} and {total_col!r} must be numeric and non-missing"
        )

    if prior is None:
        prior = fit_beta_prior(successes.values, totals.values)

    # Use scalar arithmetic on the underlying arrays to avoid index-alignment
    # surprises when df has a non-default index.
    alpha1 = prior.alpha + successes.values
    beta1 = prior.beta + totals.values - successes.values

    fitted = alpha1 / (alpha1 + beta1)
    low = beta_dist.ppf(0.025, alpha1, beta1)
    high = beta_dist.ppf(0.975, alpha1, beta1)

    out = df.copy()
    out["ebb_fitted"] = fitted
    out["ebb_low"] = low
    out["ebb_high"] = high
    return out
