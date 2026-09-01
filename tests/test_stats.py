import pandas as pd
import numpy as np
import pytest
import statsmodels.formula.api as smf

from pyflexplot import model_comparison, estimates, p_format, eliminated_columns


def test_model_comparison_basic():
    np.random.seed(0)
    df = pd.DataFrame({
        "y": np.random.normal(size=100),
        "x": np.random.normal(size=100),
        "z": np.random.normal(size=100),
    })
    small = smf.ols("y ~ x", data=df).fit()
    large = smf.ols("y ~ x + z", data=df).fit()

    res, p = model_comparison(small, large)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 2
    assert 0 <= p <= 1
    # Bayes factor column attached to the better model (lowest BIC).
    assert "BayesFactor" in res.columns
    bf_values = res["BayesFactor"].tolist()
    bics = res["BIC"].tolist()
    better_idx = int(np.argmin(bics))
    assert bf_values[better_idx] >= 1.0
    assert bf_values[1 - better_idx] == pytest.approx(
        1.0 / bf_values[better_idx]
    )
    # R-squared columns present for OLS fits.
    assert "R.squared" in res.columns
    assert "Adj.R.squared" in res.columns
    for r in res["R.squared"]:
        assert 0.0 <= r <= 1.0


def test_model_comparison_wrong_order_still_valid():
    np.random.seed(0)
    df = pd.DataFrame({
        "y": np.random.normal(size=100),
        "x": np.random.normal(size=100),
        "z": np.random.normal(size=100),
    })
    small = smf.ols("y ~ x", data=df).fit()
    large = smf.ols("y ~ x + z", data=df).fit()

    # Either order should return a valid p-value because we reorder by LLF.
    res1, p1 = model_comparison(small, large)
    res2, p2 = model_comparison(large, small)
    assert p1 == pytest.approx(p2)
    # Bayes factor table should be the same regardless of order (BIC
    # symmetry via inversion).
    assert res1["BayesFactor"].tolist() == pytest.approx(
        res2["BayesFactor"].tolist()
    )


def test_model_comparison_bayes_factor_matches_bic():
    """BF for model 1 over model 2 = exp((BIC_2 - BIC_1) / 2)."""
    np.random.seed(0)
    df = pd.DataFrame({
        "y": np.random.normal(size=100),
        "x": np.random.normal(size=100),
    })
    m1 = smf.ols("y ~ x", data=df).fit()
    m2 = smf.ols("y ~ 1", data=df).fit()
    res, _ = model_comparison(m1, m2)
    bf_raw = float(np.exp((m2.bic - m1.bic) / 2.0))
    bics = res["BIC"].tolist()
    better_idx = int(np.argmin(bics))
    assert res["BayesFactor"].iloc[better_idx] == pytest.approx(bf_raw)


def test_model_comparison_missing_attributes():
    class Dummy:
        pass
    with pytest.raises(ValueError, match="missing required attributes"):
        model_comparison(Dummy(), Dummy())


def test_p_format():
    assert p_format(0.05) == ".050"
    assert p_format(0.0005) == "<.001"


def test_estimates_returns_real_effect_size_report():
    """estimates() returns a structured dict (not model.summary()) with
    R², standardized betas, semi-partial R², and factor/numeric split.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=100),
        "x1": rng.normal(size=100),
        "x2": rng.normal(size=100),
    })
    fit = smf.ols("y ~ x1 + x2", data=df).fit()
    est = estimates(fit)
    # Structure: dict with the expected keys
    for key in (
        "r.squared",
        "adj.r.squared",
        "sigma",
        "n",
        "r.squared.ci",
        "coef",
        "standardized",
        "semi.p.r2",
        "factors",
        "numbers",
        "formula",
    ):
        assert key in est, f"Missing key: {key}"
    # R² and adj-R² are in [0, 1] (adj-R² can be negative; that is correct)
    assert 0.0 <= est["r.squared"] <= 1.0
    assert est["adj.r.squared"] <= 1.0
    # n is positive
    assert est["n"] == 100
    # coef DataFrame has the right shape (intercept + 2 predictors)
    assert len(est["coef"]) == 3
    # standardized excludes the intercept
    assert "Intercept" not in est["standardized"].index
    assert "x1" in est["standardized"].index
    assert "x2" in est["standardized"].index
    # semi.p.r2 is per predictor
    assert "x1" in est["semi.p.r2"].index
    assert "x2" in est["semi.p.r2"].index
    # factors/numbers split
    assert est["factors"] == []
    assert sorted(est["numbers"]) == ["x1", "x2"]


def test_estimates_standardized_betas_consistent_with_coefs():
    """Standardized beta = b_j * sd(x_j) / sd(y) must match the manual
    computation on the same data.
    """
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "x": rng.normal(size=200),
    })
    fit = smf.ols("y ~ x", data=df).fit()
    est = estimates(fit)
    expected = fit.params["x"] * df["x"].std(ddof=1) / df["y"].std(ddof=1)
    assert est["standardized"]["x"] == pytest.approx(float(expected), rel=1e-9)


def test_estimates_semi_p_r2_with_three_predictors():
    """Semi-partial R² for each predictor equals R²(full) - R²(reduced
    without that predictor). Verify on a model with three predictors.
    """
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame({
        "y": rng.normal(size=n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
    })
    fit = smf.ols("y ~ x1 + x2 + x3", data=df).fit()
    est = estimates(fit)
    for pred in ("x1", "x2", "x3"):
        other = " + ".join(p for p in ("x1", "x2", "x3") if p != pred)
        reduced = smf.ols(f"y ~ {other}", data=df).fit()
        expected = fit.rsquared - reduced.rsquared
        assert est["semi.p.r2"][pred] == pytest.approx(
            float(expected), abs=1e-9
        ), f"Semi-p R² for {pred}: expected {expected}, got {est['semi.p.r2'][pred]}"


def test_estimates_factor_detection():
    """Object / categorical columns are routed to factors; numeric to numbers.

    For ``C(group)`` in the formula, the underlying frame column is
    ``group``; the factor routing happens via the frame's column dtype.
    """
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "y": rng.normal(size=100),
        "x1": rng.normal(size=100),
        "group": rng.choice(["A", "B"], size=100),
    })
    # Convert "group" to a C() formula term to get a single coefficient
    fit = smf.ols("y ~ x1 + C(group)", data=df).fit()
    est = estimates(fit)
    # The factor maps to the underlying column name "group".
    assert "group" in est["factors"]
    assert "x1" in est["numbers"]


def test_estimates_rejects_non_ols():
    """estimates() raises TypeError for non-OLS-like models."""
    with pytest.raises(TypeError, match="rsquared"):
        estimates(object())


def test_eliminated_columns():
    df = pd.DataFrame({
        "keep": [1, 2, 3],
        "drop": [np.nan, np.nan, 3.0],
    })
    out = eliminated_columns(df, threshold=0.5)
    assert "keep" in out.columns
    assert "drop" not in out.columns
