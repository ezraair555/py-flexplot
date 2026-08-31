import pandas as pd
import numpy as np
import pytest
import statsmodels.formula.api as smf
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
    assert out == {"y": "y", "x": "x", "color": None, "given": [], "all_x": ["x"], "intercept_only": False}

    out = parse_flexplot_formula("y ~ x + z | a + b")
    assert out["y"] == "y"
    assert out["x"] == "x"
    assert out["color"] == "z"
    assert out["given"] == ["a", "b"]
    assert out["all_x"] == ["x", "z"]


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
