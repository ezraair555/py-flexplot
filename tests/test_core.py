import pandas as pd
import numpy as np
import pytest
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
from plotnine import ggplot

from pyflexplot import flexplot, added_plot, visualize, compare_fits
from pyflexplot.core import parse_flexplot_formula


def test_basic_plot():
    df = pd.DataFrame({
        "y": np.random.normal(size=100),
        "x": np.random.normal(size=100),
        "z": np.random.choice(["A", "B"], size=100)
    })

    # 1. Basic Scatter
    p1 = flexplot("y ~ x", data=df)
    assert isinstance(p1, ggplot)

    # 2. Faceted plot
    p2 = flexplot("y ~ x | z", data=df)
    assert isinstance(p2, ggplot)

    # 3. Added Variable Plot
    p3 = added_plot("y ~ x + z", data=df)
    assert isinstance(p3, ggplot)


def test_parse_flexplot_formula_happy():
    out = parse_flexplot_formula("y ~ x")
    assert out == {
        "y": "y",
        "x": "x",
        "color": None,
        "given": [],
        "all_x": ["x"],
        "intercept_only": False,
        "has_interaction": False,
    }

    out = parse_flexplot_formula("y ~ x + z | a + b")
    assert out["y"] == "y"
    assert out["x"] == "x"
    assert out["color"] == "z"
    assert out["given"] == ["a", "b"]
    assert out["all_x"] == ["x", "z"]
    assert out["has_interaction"] is False


def test_parse_flexplot_formula_strips_tokens():
    out = parse_flexplot_formula("  y  ~  x  +  z  |  a  +  b  ")
    assert out["y"] == "y"
    assert out["x"] == "x"
    assert out["color"] == "z"
    assert out["given"] == ["a", "b"]


def test_parse_flexplot_formula_rejects_missing_tilde():
    with pytest.raises(ValueError, match="exactly one '~'"):
        parse_flexplot_formula("y x")


def test_parse_flexplot_formula_rejects_multiple_tilde():
    with pytest.raises(ValueError, match="exactly one '~'"):
        parse_flexplot_formula("y ~ x ~ z")


def test_parse_flexplot_formula_rejects_multiple_pipe():
    with pytest.raises(ValueError, match="at most one '|'"):
        parse_flexplot_formula("y ~ x | a | b")


def test_parse_flexplot_formula_accepts_single_pipe():
    out = parse_flexplot_formula("y ~ x | a")
    assert out["given"] == ["a"]


def test_parse_flexplot_formula_empty_outcome():
    with pytest.raises(ValueError, match="non-empty outcome"):
        parse_flexplot_formula(" ~ x")


def test_parse_flexplot_formula_empty_predictor():
    with pytest.raises(ValueError, match="predictors after '~'"):
        parse_flexplot_formula("y ~")


def test_parse_flexplot_formula_intercept_only():
    out = parse_flexplot_formula("y ~ 1")
    assert out["intercept_only"] is True
    assert out["x"] is None
    assert out["all_x"] == []


def test_flexplot_intercept_only():
    df = pd.DataFrame({"y": np.random.normal(size=50)})
    p = flexplot("y ~ 1", data=df)
    assert isinstance(p, ggplot)


def test_flexplot_empty_data():
    df = pd.DataFrame({"y": [], "x": []})
    with pytest.raises(ValueError, match="non-empty"):
        flexplot("y ~ x", data=df)


def test_flexplot_missing_column():
    df = pd.DataFrame({"y": [1, 2, 3], "x": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing columns"):
        flexplot("y ~ x | z", data=df)


def test_flexplot_non_numeric_columns():
    df = pd.DataFrame({"y": ["a", "b", "c"], "x": [1, 2, 3]})
    with pytest.raises(ValueError, match="numeric"):
        flexplot("y ~ x", data=df)


def test_flexplot_color_aesthetic_applied():
    df = pd.DataFrame({
        "y": np.random.normal(size=50),
        "x": np.random.normal(size=50),
        "z": np.random.choice(["A", "B"], size=50),
    })
    p = flexplot("y ~ x + z", data=df)
    assert isinstance(p, ggplot)
    # The plot should render without error.
    p.draw()


def test_flexplot_categorical_y_path():
    df = pd.DataFrame({
        "y": [0, 1] * 25,
        "x": np.random.normal(size=50),
    })
    p = flexplot("y ~ x", data=df)
    assert isinstance(p, ggplot)


def test_flexplot_binary_y_routes_to_binomial_branch():
    """Numeric [0, 1] y must draw the binomial GLM smoother, not LM.

    Regression test for v0.6.1: before the fix, ``is_numeric_dtype([0, 1])``
    returned True and the LM branch was always taken. This test guards the
    new pre-check that routes binary y to the binomial branch.
    """
    from plotnine import geom_smooth

    df = pd.DataFrame({
        "y": [0, 1] * 30,
        "x": list(range(60)),
    })
    p = flexplot("y ~ x", data=df)
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    assert len(smooth_layers) == 1
    assert smooth_layers[0].stat.params.get("method") == "glm"
    assert smooth_layers[0].stat.params.get("method_args") == {"family": "binomial"}


def test_flexplot_binary_y_as_float_also_routes_to_binomial():
    """Float [0.0, 1.0] y must also draw the binomial GLM smoother."""
    from plotnine import geom_smooth

    df = pd.DataFrame({
        "y": [0.0, 1.0] * 30,
        "x": list(range(60)),
    })
    p = flexplot("y ~ x", data=df)
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    assert len(smooth_layers) == 1
    assert smooth_layers[0].stat.params.get("method") == "glm"


def test_flexplot_non_binary_numeric_y_still_uses_lm():
    """Numeric y with >2 unique values (e.g., 0/1/2) must NOT hit binomial."""
    from plotnine import geom_smooth

    df = pd.DataFrame({
        "y": [0, 1, 2] * 20,
        "x": list(range(60)),
    })
    p = flexplot("y ~ x", data=df)
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    assert len(smooth_layers) == 1
    # Should be LM (or loess depending on default method), not glm/binomial.
    assert smooth_layers[0].stat.params.get("method") != "glm"


def test_flexplot_categorical_y_rejects_non_numeric_string():
    df = pd.DataFrame({
        "y": ["low", "medium", "high"] * 20,
        "x": np.random.normal(size=60),
    })
    with pytest.raises(ValueError, match="numeric"):
        flexplot("y ~ x", data=df)


def test_visualize_statsmodels():
    df = pd.DataFrame({
        "y": np.random.normal(size=50),
        "x": np.random.normal(size=50),
    })
    model = smf.ols("y ~ x", data=df).fit()
    p = visualize(model, data=df)
    assert isinstance(p, ggplot)


def test_visualize_intercept_only_raises():
    df = pd.DataFrame({
        "y": np.random.normal(size=50),
    })
    model = smf.ols("y ~ 1", data=df).fit()
    with pytest.raises(ValueError, match="non-intercept"):
        visualize(model, data=df)


def test_visualize_no_data_raises():
    class Dummy:
        pass
    with pytest.raises(ValueError, match="No data provided"):
        visualize(Dummy())


def test_visualize_no_predict_raises():
    class Dummy:
        pass
    df = pd.DataFrame({"y": [1, 2, 3]})
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        visualize(Dummy(), data=df)


def test_compare_fits_statsmodels():
    df = pd.DataFrame({
        "y": np.random.normal(size=50),
        "x": np.random.normal(size=50),
    })
    m1 = smf.ols("y ~ x", data=df).fit()
    m2 = smf.ols("y ~ x", data=df).fit()
    p = compare_fits("y ~ x", data=df, model1=m1, model2=m2)
    assert isinstance(p, ggplot)


def test_compare_fits_missing_predict():
    df = pd.DataFrame({"y": [1, 2, 3], "x": [1, 2, 3]})
    with pytest.raises(ValueError, match="predict method"):
        compare_fits("y ~ x", data=df, model1=None, model2=None)


def test_compare_fits_return_preds_returns_dataframe():
    """return_preds=True returns a DataFrame, not a plot."""
    df = pd.DataFrame({
        "y": np.random.normal(size=50),
        "x": np.random.normal(size=50),
    })
    m1 = smf.ols("y ~ x", data=df).fit()
    m2 = smf.ols("y ~ x", data=df).fit()
    out = compare_fits(
        "y ~ x", data=df, model1=m1, model2=m2,
        return_preds=True,
    )
    assert isinstance(out, pd.DataFrame)
    assert "__m1" in out.columns and "__m2" in out.columns
    assert len(out) == len(df)


def test_compare_fits_pred_type_link_for_glm():
    """pred_type='link' returns linear-predictor scale for GLM models."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.choice([0, 1], size=80),
        "x": rng.normal(size=80),
    })
    glm = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()
    # Response scale: probabilities in [0, 1].
    p_resp = glm.predict(df)
    # Link scale: log-odds, unbounded.
    p_link = glm.predict(df, linear=True)
    assert p_resp.min() >= 0 and p_resp.max() <= 1
    # Link scale values should differ from response scale values.
    out_resp = compare_fits(
        "y ~ x", data=df, model1=glm, model2=glm,
        return_preds=True, pred_type="response",
    )
    out_link = compare_fits(
        "y ~ x", data=df, model1=glm, model2=glm,
        return_preds=True, pred_type="link",
    )
    assert not np.allclose(out_resp["__m1"].values, out_link["__m1"].values)


def test_added_plot_residual_alignment():
    df = pd.DataFrame({
        "y": np.random.normal(size=50),
        "x": np.random.normal(size=50),
        "z": np.random.choice(["A", "B"], size=50),
    })
    p = added_plot("y ~ x + z", data=df)
    assert isinstance(p, ggplot)


def test_added_plot_residual_alignment_with_missing():
    df = pd.DataFrame({
        "y": [1.0, 2.0, np.nan, 4.0],
        "x": [1.0, np.nan, 3.0, 4.0],
        "z": ["A", "B", "A", "B"],
    })
    with pytest.raises(ValueError, match="Residual lengths"):
        added_plot("y ~ x + z", data=df)


# --- Interaction-syntax tests (v0.6.2) --------------------------------------


def test_parse_flexplot_formula_accepts_star_syntax():
    """`y ~ x*z` parses without error and expands to `x + z + x:z`."""
    out = parse_flexplot_formula("y ~ x*z")
    assert out["x"] == "x"
    assert out["color"] == "z"
    assert out["all_x"] == ["x", "z", "x:z"]
    assert out["has_interaction"] is True


def test_parse_flexplot_formula_accepts_colon_syntax():
    """`y ~ x:z` parses; first atom of `x:z` is used for x_name."""
    out = parse_flexplot_formula("y ~ x:z")
    assert out["x"] == "x"
    assert out["color"] is None
    assert out["all_x"] == ["x:z"]
    assert out["has_interaction"] is True


def test_parse_flexplot_formula_mixed_interaction_and_main():
    """`y ~ x*z + w` expands and is detected as interaction-bearing."""
    out = parse_flexplot_formula("y ~ x*z + w")
    assert out["x"] == "x"
    assert out["color"] == "z"
    assert out["all_x"] == ["x", "z", "x:z", "w"]
    assert out["has_interaction"] is True


def test_flexplot_with_star_syntax_emits_userwarning():
    """`y ~ x*z` triggers a UserWarning about additive fit (v0.6.x)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=40),
        "x": rng.normal(size=40),
        "z": rng.choice(["A", "B"], size=40),
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = flexplot("y ~ x*z", data=df)
    user_warnings = [
        w for w in caught if issubclass(w.category, UserWarning)
    ]
    assert len(user_warnings) >= 1
    assert "additive" in str(user_warnings[0].message).lower()


def test_flexplot_without_interaction_emits_no_userwarning():
    """`y ~ x + z` (no `*` or `:`) emits no interaction UserWarning."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=40),
        "x": rng.normal(size=40),
        "z": rng.choice(["A", "B"], size=40),
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = flexplot("y ~ x + z", data=df)
    interaction_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "interaction syntax" in str(w.message).lower()
    ]
    assert interaction_warnings == []


def test_flexplot_with_interaction_does_not_error_on_columns():
    """`y ~ x*z` must look up columns `x` and `z`, not `x:z`."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=40),
        "x": rng.normal(size=40),
        "z": rng.choice(["A", "B"], size=40),
    })
    # Should NOT raise "missing column x:z".
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        p = flexplot("y ~ x*z", data=df)
    assert isinstance(p, ggplot)
