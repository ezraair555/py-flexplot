import pandas as pd
import numpy as np
import pytest

from pyflexplot import fit_beta_prior, add_ebb_estimate


def test_fit_beta_prior_basic():
    successes = np.array([10, 20, 30, 40])
    totals = np.array([50, 60, 70, 80])
    prior = fit_beta_prior(successes, totals)
    assert prior.alpha > 0
    assert prior.beta > 0
    assert np.isfinite(prior.alpha)
    assert np.isfinite(prior.beta)


def test_fit_beta_prior_moments():
    successes = np.array([10, 20, 30, 40])
    totals = np.array([50, 60, 70, 80])
    prior = fit_beta_prior(successes, totals, method="moments")
    assert prior.alpha > 0
    assert prior.beta > 0


def test_fit_beta_prior_zero_variance():
    # All successes are 50% of totals: zero variance in observed rates.
    successes = np.array([5, 5, 5])
    totals = np.array([10, 10, 10])
    prior = fit_beta_prior(successes, totals)
    assert np.isfinite(prior.alpha) and prior.alpha > 0
    assert np.isfinite(prior.beta) and prior.beta > 0


def test_fit_beta_prior_rejects_successes_gt_totals():
    with pytest.raises(ValueError, match="successes"):
        fit_beta_prior(np.array([10]), np.array([5]))


def test_fit_beta_prior_rejects_negative_successes():
    with pytest.raises(ValueError, match="successes"):
        fit_beta_prior(np.array([-1]), np.array([5]))


def test_fit_beta_prior_rejects_zero_totals():
    with pytest.raises(ValueError, match="totals"):
        fit_beta_prior(np.array([0]), np.array([0]))


def test_add_ebb_estimate_basic():
    df = pd.DataFrame({
        "successes": [10, 20, 30],
        "totals": [50, 60, 70],
    })
    out = add_ebb_estimate(df, "successes", "totals")
    assert "ebb_fitted" in out.columns
    assert "ebb_low" in out.columns
    assert "ebb_high" in out.columns
    assert np.all(out["ebb_fitted"] >= out["ebb_low"])
    assert np.all(out["ebb_fitted"] <= out["ebb_high"])


def test_add_ebb_estimate_non_default_index():
    df = pd.DataFrame(
        {"successes": [10, 20, 30], "totals": [50, 60, 70]},
        index=["a", "b", "c"],
    )
    out = add_ebb_estimate(df, "successes", "totals")
    assert len(out) == 3
    assert not out["ebb_fitted"].isna().any()
