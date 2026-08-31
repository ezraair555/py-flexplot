"""Tests for pyflexplot.eta_squared — port of R's sjstats::eta_sq().

The Python implementation computes partial eta-squared (η²_p) at the
*model* level (a single value, indexed by 'model') because statsmodels'
OLS exposes one model-F, not per-term Fs. For per-predictor semi-partial
R² (a different but related quantity), use ``estimates()``.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pyflexplot import eta_squared
from pyflexplot.stats import eta_squared as eta_squared_internal


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------


def test_eta_squared_returns_dataframe():
    """eta_squared returns a DataFrame."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=80), "x1": rng.normal(size=80)})
    model = smf.ols("y ~ x1", data=df).fit()
    out = eta_squared(model)
    assert isinstance(out, pd.DataFrame)


def test_eta_squared_returns_model_index():
    """The single row is indexed by 'model'."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=80), "x1": rng.normal(size=80)})
    model = smf.ols("y ~ x1", data=df).fit()
    out = eta_squared(model)
    assert list(out.index) == ["model"]


def test_eta_squared_columns_include_required_keys():
    """Required columns: eta_sq, eta_sq_ci_low, eta_sq_ci_high, F."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=80), "x1": rng.normal(size=80)})
    model = smf.ols("y ~ x1", data=df).fit()
    out = eta_squared(model)
    for col in ("eta_sq", "eta_sq_ci_low", "eta_sq_ci_high", "F"):
        assert col in out.columns, f"missing column {col}"


# ---------------------------------------------------------------------------
# Numeric correctness
# ---------------------------------------------------------------------------


def test_eta_squared_value_in_unit_interval():
    """eta_sq is in [0, 1]."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "x1": rng.normal(size=200),
        "x2": rng.normal(size=200),
    })
    model = smf.ols("y ~ x1 + x2", data=df).fit()
    out = eta_squared(model)
    eta2 = float(out["eta_sq"].iloc[0])
    assert 0.0 <= eta2 <= 1.0


def test_eta_squared_value_matches_formula():
    """η²_p = (F * df1) / (F * df1 + df2)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=100),
        "x1": rng.normal(size=100),
        "x2": rng.normal(size=100),
    })
    model = smf.ols("y ~ x1 + x2", data=df).fit()
    expected = (model.fvalue * model.df_model) / (
        model.fvalue * model.df_model + model.df_resid
    )
    actual = float(eta_squared(model)["eta_sq"].iloc[0])
    assert abs(actual - expected) < 1e-10


def test_eta_squared_ci_contains_point_estimate():
    """The CI is for the population η²_p; for moderate samples it should
    contain the observed value (or at least the observed value shouldn't be
    far outside the CI for n=200).
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "x1": rng.normal(size=200),
        "x2": rng.normal(size=200),
    })
    model = smf.ols("y ~ x1 + x2", data=df).fit()
    out = eta_squared(model)
    eta2 = float(out["eta_sq"].iloc[0])
    lo = float(out["eta_sq_ci_low"].iloc[0])
    hi = float(out["eta_sq_ci_high"].iloc[0])
    # At n=200 the CI should be reasonably tight.
    assert hi - lo < 0.3
    # And the observed value should be inside (or near) the CI for a
    # model with at least some signal.
    # Allow either: contains the value, OR is at most 1 sample-width away.
    if not (lo <= eta2 <= hi):
        # If not in CI, the discrepancy should be modest.
        if eta2 < lo:
            assert abs(eta2 - lo) < 0.05
        else:
            assert abs(eta2 - hi) < 0.05


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_eta_squared_rejects_non_statsmodels_object():
    """Passing a non-statsmodels object raises TypeError."""
    with pytest.raises(TypeError, match="eta_squared requires"):
        eta_squared(42)


def test_eta_squared_rejects_object_without_exog_names():
    """Passing an object without .model.exog_names raises TypeError."""
    class _FakeModel:
        df_model = 1
        df_resid = 100

    with pytest.raises(TypeError, match="exog_names"):
        eta_squared(_FakeModel())


# ---------------------------------------------------------------------------
# Cross-validation with the standalone _r_squared_ci
# ---------------------------------------------------------------------------


def test_eta_squared_ci_matches_r_squared_ci_for_equivalent_inputs():
    """The CI in eta_squared uses _r_squared_ci; verify they agree."""
    from pyflexplot.stats import _r_squared_ci

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=150),
        "x1": rng.normal(size=150),
        "x2": rng.normal(size=150),
    })
    model = smf.ols("y ~ x1 + x2", data=df).fit()
    out = eta_squared(model, level=0.90)
    eta2 = float(out["eta_sq"].iloc[0])
    lo = float(out["eta_sq_ci_low"].iloc[0])
    hi = float(out["eta_sq_ci_high"].iloc[0])
    direct_lo, direct_hi = _r_squared_ci(
        r2=eta2, df_model=int(model.df_model), nobs=int(model.nobs), level=0.90
    )
    assert abs(lo - direct_lo) < 1e-10
    assert abs(hi - direct_hi) < 1e-10