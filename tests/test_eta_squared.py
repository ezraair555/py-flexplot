"""Tests for pyflexplot.eta_squared — port of R's sjstats::eta_sq().

v0.7.5: per-term partial eta-squared via type-III SS (statsmodels'
``anova_lm(..., typ=3)``). Returns one row per non-intercept term with
η²_p, CI, F, p-value, and df.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pyflexplot import eta_squared


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


def test_eta_squared_indexed_by_predictor_names():
    """Each row is indexed by a predictor name (excluding intercept)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=100),
        "x1": rng.normal(size=100),
        "x2": rng.normal(size=100),
    })
    model = smf.ols("y ~ x1 + x2", data=df).fit()
    out = eta_squared(model)
    assert sorted(out.index.tolist()) == ["x1", "x2"]


def test_eta_squared_one_row_per_predictor():
    """N predictors => N rows (excluding intercept)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=120),
        "x1": rng.normal(size=120),
        "x2": rng.normal(size=120),
        "x3": rng.normal(size=120),
    })
    model = smf.ols("y ~ x1 + x2 + x3", data=df).fit()
    out = eta_squared(model)
    assert len(out) == 3


def test_eta_squared_columns_include_required_keys():
    """Required columns: eta_sq, eta_sq_ci_low, eta_sq_ci_high, F, p_value, df."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=80), "x1": rng.normal(size=80)})
    model = smf.ols("y ~ x1", data=df).fit()
    out = eta_squared(model)
    for col in ("eta_sq", "eta_sq_ci_low", "eta_sq_ci_high", "F", "p_value", "df"):
        assert col in out.columns, f"missing column {col}"


# ---------------------------------------------------------------------------
# Categorical predictors
# ---------------------------------------------------------------------------


def test_eta_squared_categorical_has_df_greater_than_one():
    """A 3-level categorical predictor has df=2."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "g": rng.choice(["A", "B", "C"], size=200),
    })
    model = smf.ols("y ~ C(g)", data=df).fit()
    out = eta_squared(model)
    assert int(out.loc["C(g)", "df"]) == 2


def test_eta_squared_mixed_predictors_returns_one_row_per_term():
    """Mixed numeric + categorical: one row per term."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "x1": rng.normal(size=200),
        "g": rng.choice(["A", "B", "C"], size=200),
    })
    model = smf.ols("y ~ x1 + C(g)", data=df).fit()
    out = eta_squared(model)
    assert sorted(out.index.tolist()) == ["C(g)", "x1"]


# ---------------------------------------------------------------------------
# Numeric correctness
# ---------------------------------------------------------------------------


def test_eta_squared_values_in_unit_interval():
    """All per-term eta_sq values are in [0, 1]."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "x1": rng.normal(size=200),
        "x2": rng.normal(size=200),
    })
    model = smf.ols("y ~ x1 + x2", data=df).fit()
    out = eta_squared(model)
    for eta2 in out["eta_sq"]:
        assert 0.0 <= eta2 <= 1.0


def test_eta_squared_value_matches_anova_table_formula():
    """η²_p = (F * df_term) / (F * df_term + df_resid) for each term."""
    from statsmodels.stats.anova import anova_lm
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "x1": rng.normal(size=200),
        "x2": rng.normal(size=200),
    })
    model = smf.ols("y ~ x1 + x2", data=df).fit()
    anova_tbl = anova_lm(model, typ=3)
    df_resid = float(anova_tbl.loc["Residual", "df"])
    out = eta_squared(model)
    for term in out.index:
        F_term = float(anova_tbl.loc[term, "F"])
        df_term = float(anova_tbl.loc[term, "df"])
        expected = (F_term * df_term) / (F_term * df_term + df_resid)
        actual = float(out.loc[term, "eta_sq"])
        assert abs(actual - expected) < 1e-10


def test_eta_squared_p_values_match_anova_table():
    """Per-term p_values match the type-III ANOVA table."""
    from statsmodels.stats.anova import anova_lm
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "x1": rng.normal(size=200),
        "x2": rng.normal(size=200),
    })
    model = smf.ols("y ~ x1 + x2", data=df).fit()
    anova_tbl = anova_lm(model, typ=3)
    out = eta_squared(model)
    for term in out.index:
        expected_p = float(anova_tbl.loc[term, "PR(>F)"])
        actual_p = float(out.loc[term, "p_value"])
        assert abs(actual_p - expected_p) < 1e-10


def test_eta_squared_strong_predictor_has_high_eta_sq():
    """A predictor engineered to explain y should have high η²_p."""
    rng = np.random.default_rng(0)
    n = 500
    x_strong = rng.normal(size=n)
    x_weak = rng.normal(size=n)
    y = 1.5 * x_strong + 0.1 * x_weak + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "x_strong": x_strong, "x_weak": x_weak})
    model = smf.ols("y ~ x_strong + x_weak", data=df).fit()
    out = eta_squared(model)
    assert out.loc["x_strong", "eta_sq"] > out.loc["x_weak", "eta_sq"]


# ---------------------------------------------------------------------------
# CI
# ---------------------------------------------------------------------------


def test_eta_squared_ci_columns_are_tuples_in_unit_interval():
    """Per-term CI bounds are in [0, 1]."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "x1": rng.normal(size=200),
        "x2": rng.normal(size=200),
    })
    model = smf.ols("y ~ x1 + x2", data=df).fit()
    out = eta_squared(model)
    for term in out.index:
        lo = float(out.loc[term, "eta_sq_ci_low"])
        hi = float(out.loc[term, "eta_sq_ci_high"])
        assert 0.0 <= lo <= hi <= 1.0


def test_eta_squared_ci_uses_r_squared_ci():
    """Per-term CI uses _r_squared_ci for the corresponding r2 / df / n."""
    from pyflexplot.stats import _r_squared_ci
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=150),
        "x1": rng.normal(size=150),
        "x2": rng.normal(size=150),
    })
    model = smf.ols("y ~ x1 + x2", data=df).fit()
    out = eta_squared(model, level=0.90)
    for term in out.index:
        eta2 = float(out.loc[term, "eta_sq"])
        lo = float(out.loc[term, "eta_sq_ci_low"])
        hi = float(out.loc[term, "eta_sq_ci_high"])
        df_term = int(out.loc[term, "df"])
        direct_lo, direct_hi = _r_squared_ci(
            r2=eta2, df_model=df_term, nobs=int(model.nobs), level=0.90
        )
        assert abs(lo - direct_lo) < 1e-10
        assert abs(hi - direct_hi) < 1e-10


def test_eta_squared_custom_level_changes_ci_width():
    """A larger level widens the CI."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "x1": rng.normal(size=200),
    })
    model = smf.ols("y ~ x1", data=df).fit()
    out_90 = eta_squared(model, level=0.90)
    out_99 = eta_squared(model, level=0.99)
    width_90 = float(out_90.iloc[0]["eta_sq_ci_high"] - out_90.iloc[0]["eta_sq_ci_low"])
    width_99 = float(out_99.iloc[0]["eta_sq_ci_high"] - out_99.iloc[0]["eta_sq_ci_low"])
    assert width_99 >= width_90


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
# typ parameter
# ---------------------------------------------------------------------------


def test_eta_squared_typ_2_matches_anova_typ_2():
    """typ=2 produces SS values consistent with anova_lm(typ=2)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "x1": rng.normal(size=200),
        "x2": rng.normal(size=200),
    })
    model = smf.ols("y ~ x1 + x2", data=df).fit()
    out2 = eta_squared(model, typ=2)
    # The exact value should match what anova_lm(typ=2) gives.
    from statsmodels.stats.anova import anova_lm
    anova_tbl = anova_lm(model, typ=2)
    df_resid = float(anova_tbl.loc["Residual", "df"])
    for term in out2.index:
        F_term = float(anova_tbl.loc[term, "F"])
        df_term = float(anova_tbl.loc[term, "df"])
        expected = (F_term * df_term) / (F_term * df_term + df_resid)
        actual = float(out2.loc[term, "eta_sq"])
        assert abs(actual - expected) < 1e-10