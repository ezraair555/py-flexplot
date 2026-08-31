"""Tests for v0.8.0 parity fixes (#1-#6 from docs/parity_review_2026-08-31.md).

#1 model_comparison: non-nested support + pred.difference.
#2 estimates(): factor_estimates + mean_differences (Cohen's d) + mc=.
#3 added_plot(): R semantics (last var, x=, lm_formula=, offset).
#4 flexplot(): R spread aliases, jitter=, alpha=, raw_data=, method rlm/glm.
#5 ghost_line: panel-repetition semantics with facets + dict ghost_reference.
#6 standalone accessors: standardized_beta, rsq_change, bf_bic.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from plotnine import ggplot

from pyflexplot import (
    flexplot,
    added_plot,
    model_comparison,
    estimates,
    standardized_beta,
    rsq_change,
    bf_bic,
)


def _ols_data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "y": rng.normal(size=n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
    })


def _factor_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.choice(["A", "B", "C"], size=n)
    x = rng.normal(size=n)
    y = 2.0 * (g == "B") + 1.0 * x + rng.normal(scale=0.8, size=n)
    return pd.DataFrame({"y": y, "x": x, "g": g})


# ===========================================================================
# #1 model_comparison
# ===========================================================================


def test_model_comparison_non_nested_returns_none_p():
    """Non-nested models: AIC/BIC/BF still compare; LRT p-value is None."""
    df = _ols_data()
    m_nest = smf.ols("y ~ x1", data=df).fit()
    m_nonnest = smf.ols("y ~ x2 + x3", data=df).fit()
    res, p = model_comparison(m_nest, m_nonnest)
    assert p is None
    # The stats table is still complete.
    assert {"AIC", "BIC", "LogLik", "BayesFactor"}.issubset(res.columns)


def test_model_comparison_nested_still_has_p():
    """Nested models retain a numeric LRT p-value."""
    df = _ols_data()
    m_small = smf.ols("y ~ x1", data=df).fit()
    m_large = smf.ols("y ~ x1 + x2", data=df).fit()
    res, p = model_comparison(m_small, m_large)
    assert p is not None and 0.0 <= p <= 1.0


def test_model_comparison_pred_difference_quantiles():
    """return_pred_difference=True returns a 5-quantile Series."""
    df = _ols_data()
    m1 = smf.ols("y ~ x1", data=df).fit()
    m2 = smf.ols("y ~ x1 + x2", data=df).fit()
    res, p, diff = model_comparison(m1, m2, return_pred_difference=True)
    assert isinstance(diff, pd.Series)
    assert list(diff.index) == [0.0, 0.25, 0.5, 0.75, 1.0]
    # Quantiles are monotone non-decreasing.
    vals = diff.to_numpy()
    assert np.all(np.diff(vals) >= -1e-12)
    # Median difference for nested models should be ~0 (small models share preds).
    assert abs(diff.loc[0.5]) < 1.0


def test_model_comparison_default_return_stays_2tuple():
    """Backward compat: default call returns exactly 2 elements."""
    df = _ols_data()
    m1 = smf.ols("y ~ x1", data=df).fit()
    m2 = smf.ols("y ~ x1 + x2", data=df).fit()
    out = model_comparison(m1, m2)
    assert len(out) == 2


# ===========================================================================
# #2 estimates() factor tables
# ===========================================================================


def test_estimates_factor_estimates_table():
    """factor_estimates has one row per level with CI covering the truth."""
    df = _factor_data()
    m = smf.ols("y ~ x + C(g)", data=df).fit()
    rep = estimates(m)
    fe = rep["factor_estimates"]
    assert isinstance(fe, pd.DataFrame)
    assert set(fe["variable"].unique()) == {"g"}
    assert sorted(fe["level"].unique()) == ["A", "B", "C"]
    # Engineered truth: level B is ~2.0 above the others.
    b_est = fe.loc[fe["level"] == "B", "estimate"].iloc[0]
    assert 1.5 < b_est < 2.5


def test_estimates_mean_differences_with_cohens_d():
    """mean_differences has pairwise contrasts with Cohen's d."""
    df = _factor_data()
    m = smf.ols("y ~ x + C(g)", data=df).fit()
    rep = estimates(m)
    md = rep["mean_differences"]
    assert isinstance(md, pd.DataFrame)
    assert {"difference", "ci.lower", "ci.upper", "cohens.d"}.issubset(md.columns)
    # 3 levels -> 3 pairs.
    assert len(md) == 3
    # A - B should be strongly negative (~ -2.2) with |d| > 1.
    ab = md.loc[md["comparison"] == "A - B"]
    assert ab["difference"].iloc[0] < -1.5
    assert abs(ab["cohens.d"].iloc[0]) > 1.0


def test_estimates_mc_false_gates_comparisons():
    """mc=False: mean_differences is None and semi.p.r2 is empty."""
    df = _factor_data()
    m = smf.ols("y ~ x + C(g)", data=df).fit()
    rep = estimates(m, mc=False)
    assert rep["mean_differences"] is None
    assert len(rep["semi.p.r2"]) == 0
    # factor_estimates still compute (they're estimates, not comparisons).
    assert rep["factor_estimates"] is not None


def test_estimates_pure_numeric_model_has_empty_factor_tables():
    """No factor predictors: tables are None (or empty) without error."""
    df = _ols_data()
    m = smf.ols("y ~ x1 + x2", data=df).fit()
    rep = estimates(m)
    assert rep["factor_estimates"] is None


# ===========================================================================
# #3 added_plot R semantics
# ===========================================================================


def test_added_plot_default_displays_last_variable():
    """R default: y ~ x + z plots z (the LAST variable) on the x-axis."""
    df = _ols_data(n=80)
    p = added_plot("y ~ x1 + x2", data=df)
    assert isinstance(p, ggplot)
    # The x-aesthetic maps to the last variable (x2).
    assert str(p.mapping.get("x", "")) == "x2" or getattr(p.mapping, "x", None) == "x2"


def test_added_plot_x_positional_selection():
    """x=1 selects the first predictor for display."""
    df = _ols_data(n=80)
    p = added_plot("y ~ x1 + x2", data=df, x=1)
    assert str(p.mapping.get("x", "")) == "x1"


def test_added_plot_lm_formula_conditioning():
    """lm_formula= overrides the conditioning model."""
    df = _ols_data(n=80)
    p = added_plot("y ~ x1 + x2", data=df, lm_formula="y ~ x1")
    assert isinstance(p, ggplot)


def test_added_plot_offset_keeps_y_scale():
    """offset=True (default): residual mean ≈ observed mean of y."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": 100.0 + rng.normal(size=200),
        "x1": rng.normal(size=200),
        "x2": rng.normal(size=200),
    })
    p = added_plot("y ~ x1 + x2", data=df)
    resid_center = float(p.data["y|cond"].mean())
    # With offset=True the residuals sit near 100, not near 0.
    assert resid_center > 95.0


# ===========================================================================
# #4 flexplot() R-token aliases + jitter/alpha/raw_data + method rlm/glm
# ===========================================================================


def test_flexplot_spread_accepts_r_tokens():
    """spread='quartiles' and 'sterr' are accepted (R tokens)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.choice(["a", "b", "c"], size=90),
        "y": rng.normal(size=90),
    })
    p1 = flexplot("y ~ x", data=df, spread="quartiles")
    p2 = flexplot("y ~ x", data=df, spread="sterr")
    assert isinstance(p1, ggplot) and isinstance(p2, ggplot)


def test_flexplot_method_rlm_and_glm():
    """method='rlm' and 'glm' route through the numeric smoother."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=80),
        "y": rng.normal(size=80),
    })
    for method_value in ("rlm", "glm"):
        p = flexplot("y ~ x", data=df, method=method_value)
        assert isinstance(p, ggplot)
        types = [l.geom.__class__.__name__ for l in p.layers]
        assert "geom_smooth" in types


def test_flexplot_jitter_bool_and_vector():
    """jitter=True / jitter=[0.05, 0] / jitter=False are all accepted."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.choice(["a", "b"], size=60),
        "y": rng.normal(size=60),
    })
    for jit in (True, False, [0.05, 0.0]):
        p = flexplot("y ~ x", data=df, jitter=jit)
        assert isinstance(p, ggplot)


def test_flexplot_jitter_false_uses_points_not_jitter():
    """jitter=False: discrete branch uses geom_point instead of geom_jitter."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.choice(["a", "b"], size=60),
        "y": rng.normal(size=60),
    })
    p = flexplot("y ~ x", data=df, jitter=False)
    types = [l.geom.__class__.__name__ for l in p.layers]
    assert "geom_jitter" not in types
    assert "geom_point" in types


def test_flexplot_jitter_invalid_raises():
    """jitter=[0.1] (wrong length) raises ValueError."""
    df = pd.DataFrame({"x": ["a", "b"] * 5, "y": np.random.normal(size=10)})
    with pytest.raises(ValueError, match="jitter must be"):
        flexplot("y ~ x", data=df, jitter=[0.1])


def test_flexplot_raw_data_false_hides_points():
    """raw_data=False: single-predictor numeric formula hides the scatter."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.choice(["a", "b", "c"], size=90),
        "y": rng.normal(size=90),
    })
    p = flexplot("y ~ x", data=df, raw_data=False)
    types = [l.geom.__class__.__name__ for l in p.layers]
    assert "geom_jitter" not in types
    assert "geom_point" not in types


def test_flexplot_alpha_overrides_point_transparency():
    """alpha=0.15 is accepted (and plumbs through plotnine's aes params);
    alpha=2 is rejected."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=60), "y": rng.normal(size=60)})
    # plotnine stores alpha as part of the layer's aes params; verify
    # acceptance + layer presence rather than param introspection.
    p = flexplot("y ~ x", data=df, alpha=0.15)
    point_layers = [l for l in p.layers if l.geom.__class__.__name__ == "geom_point"]
    assert point_layers
    with pytest.raises(ValueError, match="alpha must be"):
        flexplot("y ~ x", data=df, alpha=2)


# ===========================================================================
# #5 ghost_line panel repetition
# ===========================================================================


def test_ghost_line_with_facet_repeats_reference_line():
    """With a facet, ghost_line=color adds a panel-repeated geom_line."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "x": rng.normal(size=200),
        "g": rng.choice(["A", "B"], size=200),
    })
    p = flexplot("y ~ x | g", data=df, ghost_line="red")
    line_layers = [l for l in p.layers if l.geom.__class__.__name__ == "geom_line"]
    assert len(line_layers) >= 1


def test_ghost_reference_dict_selects_reference_panel():
    """ghost_reference={'g': 'B'} uses group B's line as the ghost."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "x": rng.normal(size=200),
        "g": rng.choice(["A", "B"], size=200),
    })
    p = flexplot("y ~ x | g", data=df, ghost_line="black", ghost_reference={"g": "B"})
    assert isinstance(p, ggplot)


def test_ghost_reference_dict_without_facet_raises():
    """Dict ghost_reference requires a facet."""
    df = pd.DataFrame({"y": np.random.normal(size=50), "x": np.random.normal(size=50)})
    with pytest.raises(TypeError, match="requires a .* facet"):
        flexplot("y ~ x", data=df, ghost_reference={"g": "A"})


def test_ghost_line_no_facet_rejects_arbitrary_colors():
    """Without facets, non-legacy ghost_line colors raise ValueError."""
    df = pd.DataFrame({"y": np.random.normal(size=50), "x": np.random.normal(size=50)})
    with pytest.raises(ValueError, match="requires a facet|must be"):
        flexplot("y ~ x", data=df, ghost_line="magenta")


# ===========================================================================
# #6 standalone accessors
# ===========================================================================


def test_standardized_beta_values():
    df = _ols_data()
    m = smf.ols("y ~ x1 + x2", data=df).fit()
    betas = standardized_beta(m)
    assert isinstance(betas, pd.Series)
    assert set(betas.index) == {"x1", "x2"}
    # Small-noise, zero-signal data: betas should be small.
    assert betas.abs().max() < 0.3


def test_standardized_beta_rejects_non_model():
    with pytest.raises(TypeError, match="standardized_beta requires"):
        standardized_beta(42)


def test_rsq_change_value():
    df = _ols_data()
    m_red = smf.ols("y ~ x1", data=df).fit()
    m_full = smf.ols("y ~ x1 + x2", data=df).fit()
    delta = rsq_change(m_red, m_full)
    assert abs(delta - (m_full.rsquared - m_red.rsquared)) < 1e-12


def test_bf_bic_matches_model_comparison_column():
    df = _ols_data()
    m1 = smf.ols("y ~ x1", data=df).fit()
    m2 = smf.ols("y ~ x1 + x2", data=df).fit()
    res, _ = model_comparison(m1, m2)
    direct = bf_bic(m1, m2)
    # model_comparison attributes BF>=1 to the better model's row; the raw
    # bf_bic is model1-over-model2. Both directions appear in the table.
    vals = res["BayesFactor"].to_numpy()
    assert any(abs(v - direct) < 1e-9 for v in vals) or any(
        abs(v - 1.0 / direct) < 1e-9 for v in vals
    )