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
    log_coeff = gammaln(totals + 1) - gammaln(successes + 1) - gammaln(totals - successes + 1)
    ll = log_coeff + betaln(successes + alpha, totals - successes + beta) - betaln(alpha, beta)
    return -np.sum(ll)

def fit_beta_prior(successes, totals, method="mle"):
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    
    # Method of moments for initial values
    rates = successes / totals
    m = rates.mean()
    v = rates.var()
    common = m * (1 - m) / v - 1
    alpha0 = m * common
    beta0 = (1 - m) * common
    
    if method == "moments":
        return BetaPrior(alpha0, beta0, len(successes), "moments")
        
    res = minimize(_beta_binomial_loglik, x0=[alpha0, beta0], args=(successes, totals), bounds=((1e-6, None), (1e-6, None)))
    return BetaPrior(res.x[0], res.x[1], len(successes), "mle")

def add_ebb_estimate(df, success_col, total_col, prior=None):
    if prior is None:
        prior = fit_beta_prior(df[success_col], df[total_col])
    
    alpha1 = prior.alpha + df[success_col]
    beta1 = prior.beta + df[total_col] - df[success_col]
    
    out = df.copy()
    out["ebb_fitted"] = alpha1 / (alpha1 + beta1)
    out["ebb_low"] = beta_dist.ppf(0.025, alpha1, beta1)
    out["ebb_high"] = beta_dist.ppf(0.975, alpha1, beta1)
    return out
