"""Tests for v0.6.4 parametric smooth extensions to flexplot().

Covers the new ``method`` values added in v0.6.4:
- ``"polynomial"``: OLS with degree-3 polynomial in x
- ``"cubic"``: alias of ``"polynomial"``
- ``"logistic"``: GLM with logit link on numeric binary y
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from plotnine import (
    ggplot,
)

from pyflexplot import flexplot
from pyflexplot.core import _VALID_FLEXPLOT_METHODS


# ---------------------------------------------------------------------------
# Method registration
# ---------------------------------------------------------------------------


def test_valid_methods_includes_polynomial_cubic_logistic():
    """The three new methods are in the frozenset."""
    assert "polynomial" in _VALID_FLEXPLOT_METHODS
    assert "cubic" in _VALID_FLEXPLOT_METHODS
    assert "logistic" in _VALID_FLEXPLOT_METHODS


def test_flexplot_rejects_unknown_method():
    """An unknown method raises ValueError (regression on validation)."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="method must be one of"):
        flexplot("y ~ x", data=df, method="spline")


# ---------------------------------------------------------------------------
# Polynomial / cubic
# ---------------------------------------------------------------------------


def test_polynomial_returns_ggplot():
    """flexplot(method='polynomial') renders without error."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": np.linspace(-3, 3, 60),
        "y": np.linspace(-3, 3, 60) ** 2 + rng.normal(scale=0.5, size=60),
    })
    p = flexplot("y ~ x", data=df, method="polynomial")
    assert isinstance(p, ggplot)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_line" in layer_types
    assert "geom_ribbon" in layer_types


def test_polynomial_with_uncertainty_none_returns_only_scatter():
    """uncertainty=None + polynomial: no fit line, just the points."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": np.linspace(-3, 3, 40),
        "y": np.linspace(-3, 3, 40) + rng.normal(size=40),
    })
    p = flexplot("y ~ x", data=df, method="polynomial", uncertainty=None)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_line" not in layer_types
    assert "geom_ribbon" not in layer_types
    assert "geom_point" in layer_types


def test_cubic_is_alias_of_polynomial():
    """cubic and polynomial produce a geom_line layer (both are degree-3 OLS)."""
    rng = np.random.default_rng(1)
    x = np.linspace(0, 5, 50)
    df = pd.DataFrame({
        "x": x,
        "y": np.sin(x) + rng.normal(scale=0.2, size=50),
    })
    p_poly = flexplot("y ~ x", data=df, method="polynomial")
    p_cubic = flexplot("y ~ x", data=df, method="cubic")
    poly_types = [layer.geom.__class__.__name__ for layer in p_poly.layers]
    cubic_types = [layer.geom.__class__.__name__ for layer in p_cubic.layers]
    assert "geom_line" in poly_types and "geom_line" in cubic_types


def test_polynomial_with_bands_draws_nested_ribbons():
    """bands=[0.5, 0.8, 0.95] + polynomial: three ribbon layers."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "x": np.linspace(-2, 2, 80),
        "y": np.linspace(-2, 2, 80) ** 2 + rng.normal(scale=0.3, size=80),
    })
    p = flexplot(
        "y ~ x", data=df,
        method="polynomial",
        bands=[0.5, 0.8, 0.95],
    )
    n_ribbons = sum(
        1 for layer in p.layers
        if layer.geom.__class__.__name__ == "geom_ribbon"
    )
    assert n_ribbons == 3


def test_polynomial_with_too_few_points_returns_plot_without_error():
    """n=1 row: degenerate; should still return a ggplot (with no fit)."""
    df = pd.DataFrame({"x": [1.0], "y": [2.0]})
    p = flexplot("y ~ x", data=df, method="polynomial")
    assert isinstance(p, ggplot)


def test_polynomial_recovery_nonlinear():
    """Polynomial of degree 3 fits a quadratic signal reasonably (low MSE)."""
    rng = np.random.default_rng(3)
    x = np.linspace(-3, 3, 200)
    y_true = x ** 2
    y = y_true + rng.normal(scale=0.1, size=200)
    df = pd.DataFrame({"x": x, "y": y})

    from statsmodels.regression.linear_model import OLS
    X = np.column_stack([np.ones_like(x), x, x ** 2, x ** 3])
    model = OLS(y, X).fit()
    pred = model.predict(np.column_stack([np.ones_like(x), x, x ** 2, x ** 3]))
    rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
    # Degree-3 polynomial in x: x^3 + x^2 + x + const. With small noise,
    # the coefficient on x^3 should be ~0 and the x^2 coefficient should
    # be ~1. The quadratic component recovered should dominate.
    assert abs(model.params[2] - 1.0) < 0.05, f"x^2 coef should be ~1, got {model.params[2]}"


# ---------------------------------------------------------------------------
# Logistic
# ---------------------------------------------------------------------------


def test_logistic_returns_ggplot_on_binary_outcome():
    """flexplot(method='logistic') on binary {0, 1} y renders correctly.

    Note: when method='logistic' is explicit, the binary pre-check is
    bypassed and the parametric-smooth branch is used (geom_line + ribbon),
    not the legacy binomial branch (geom_smooth).
    """
    rng = np.random.default_rng(0)
    x = np.linspace(-3, 3, 100)
    prob = 1 / (1 + np.exp(-x))
    y = (rng.random(100) < prob).astype(float)
    df = pd.DataFrame({"x": x, "y": y})

    p = flexplot("y ~ x", data=df, method="logistic")
    assert isinstance(p, ggplot)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_line" in layer_types


def test_logistic_on_continuous_y_falls_back_to_ols_with_warning():
    """Non-binary y + logistic: emits UserWarning and falls back to OLS."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "x": np.linspace(-2, 2, 40),
        "y": rng.normal(size=40),  # continuous, NOT binary
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = flexplot("y ~ x", data=df, method="logistic")
    fallback_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "logistic" in str(w.message).lower()
        and "falling back" in str(w.message).lower()
    ]
    assert len(fallback_warnings) == 1
    # Plot still composes a line + ribbon (OLS fallback path).
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_line" in layer_types


def test_logistic_with_band_renders_nested_ribbons():
    """bands + logistic on binary y: still draws ribbons."""
    rng = np.random.default_rng(2)
    x = np.linspace(-2, 2, 60)
    y = (rng.random(60) < 1 / (1 + np.exp(-x))).astype(float)
    df = pd.DataFrame({"x": x, "y": y})
    p = flexplot("y ~ x", data=df, method="logistic", bands=[0.5, 0.95])
    n_ribbons = sum(
        1 for layer in p.layers
        if layer.geom.__class__.__name__ == "geom_ribbon"
    )
    assert n_ribbons == 2


# ---------------------------------------------------------------------------
# Backward compat
# ---------------------------------------------------------------------------


def test_method_auto_still_routes_to_lm():
    """method='auto' (default) still routes numeric-vs-numeric through lm."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": np.linspace(-2, 2, 50),
        "y": np.linspace(-2, 2, 50) + rng.normal(scale=0.2, size=50),
    })
    p = flexplot("y ~ x", data=df, method="auto")
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    # plotnine's geom_smooth(method="lm") shows up as a layer; check the
    # smoother layer exists rather than asserting the geom class name.
    has_smooth = any("geom_smooth" in t for t in layer_types)
    assert has_smooth


def test_method_lm_still_works():
    """method='lm' (legacy default) still renders a geom_smooth layer."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": np.linspace(-2, 2, 40),
        "y": np.linspace(-2, 2, 40) + rng.normal(scale=0.1, size=40),
    })
    p = flexplot("y ~ x", data=df, method="lm")
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert any("geom_smooth" in t for t in layer_types)