"""Integration tests for v0.4.0\u2013v0.6.2 features.

These tests exercise combinations of the new features:
  - Uncertainty bands (uncertainty=..., level=, bands=)
  - Overlay smoothers (overlay=...)
  - Auto diagnostics (diagnose())
  - Interaction syntax (y ~ x*z, y ~ x:z)
  - Binomial GLM routing for numeric [0, 1] y (v0.6.1)

The aim is to make sure the features compose cleanly and don't
interfere with each other in the realistic workflow.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from plotnine import geom_ribbon, geom_smooth

from pyflexplot import diagnose, flexplot


# --- Helpers --------------------------------------------------------------


def _make_df(n: int = 200, seed: int = 0, *, binary_y: bool = False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    if binary_y:
        y = (rng.normal(size=n) > 0).astype(int)
    else:
        y = 2.0 * x + rng.normal(scale=0.5, size=n)
    z = rng.choice(["A", "B"], size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


# --- Uncertainty + Overlay combined --------------------------------------


def test_flexplot_bands_and_overlay_compose():
    """Nested bands on the primary + overlay smoothers in one chart."""
    df = _make_df()
    p = flexplot(
        "y ~ x + z", data=df,
        bands=[0.5, 0.95],
        overlay=[{"method": "loess", "label": "LOESS"}],
    )
    # 2 band layers on primary LM + 1 overlay loess = 3 smooth layers.
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    assert len(smooth_layers) == 3


def test_flexplot_overlay_with_bootstrap_uncertainty():
    """Overlay entry can specify its own uncertainty type."""
    df = _make_df()
    p = flexplot(
        "y ~ x + z", data=df,
        overlay=[{"method": "loess", "uncertainty": "bootstrap"}],
    )
    # Construction should succeed without raising; we don't render to
    # avoid the scikit-misc dependency for actual loess.
    assert p is not None


def test_flexplot_no_uncertainty_no_overlay_just_data():
    """uncertainty=None + overlay=[] yields one scatter only."""
    df = _make_df()
    p = flexplot(
        "y ~ x + z", data=df,
        uncertainty=None,
        overlay=[],
    )
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    assert len(smooth_layers) == 0


# --- diagnose() + flexplot() workflow --------------------------------------


def test_diagnose_then_flexplot_binary_y_workflow():
    """Typical diagnostic-then-plot workflow with a binary outcome."""
    df = _make_df(n=300, binary_y=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        diag = diagnose("y ~ x + z", data=df, verbose=False)
    assert diag["n_complete"] == 300
    # Now plot the binary y with default ci \u2014 must go to binomial branch.
    p = flexplot("y ~ x + z", data=df)
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    assert len(smooth_layers) == 1
    assert smooth_layers[0].stat.params.get("method") == "glm"


def test_diagnose_then_flexplot_continuous_y_workflow():
    """Diagnostic-then-plot with a continuous outcome."""
    df = _make_df(n=300, binary_y=False)
    diag = diagnose("y ~ x + z", data=df, verbose=False)
    assert "linearity" in diag
    p = flexplot(
        "y ~ x + z", data=df,
        bands=[0.5, 0.95],
        overlay=[{"method": "loess", "label": "LOESS"}],
    )
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    assert len(smooth_layers) == 3


# --- Interaction syntax + other features ---------------------------------


def test_flexplot_interaction_syntax_with_overlay():
    """Interaction syntax parses and combines with overlay entries."""
    df = _make_df()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        p = flexplot(
            "y ~ x*z", data=df,
            overlay=[{"method": "loess", "label": "LOESS"}],
        )
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    # Primary LM + 1 overlay loess = 2 smooth layers.
    assert len(smooth_layers) == 2


def test_flexplot_interaction_syntax_with_facet():
    """Interaction syntax combined with `| given` facet variables."""
    df = _make_df(n=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        p = flexplot("y ~ x*z | z", data=df)
    # Construction succeeds; primary smooth layer present.
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    assert len(smooth_layers) >= 1


# --- Edge cases -----------------------------------------------------------


def test_flexplot_bool_y_routes_to_binomial():
    """Boolean y ([True,, False]) routes to binomial (binary pre-check)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.choice([True, False], size=100),
        "x": rng.normal(size=100),
    })
    p = flexplot("y ~ x", data=df)
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    assert smooth_layers[0].stat.params.get("method") == "glm"


def test_flexplot_y_with_three_values_does_not_route_to_binomial():
    """Numeric y with 3 unique values still goes to LM, not binomial."""
    df = pd.DataFrame({
        "y": [0, 1, 2] * 30,
        "x": list(range(90)),
    })
    p = flexplot("y ~ x", data=df)
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    assert smooth_layers[0].stat.params.get("method") != "glm"


def test_flexplot_y_with_missing_among_binary_still_binomial():
    """y=[0,1] with some NaN still routes to binomial (pre-check skips NaN)."""
    df = pd.DataFrame({
        "y": [0, 1] * 25 + [np.nan] * 10,
        "x": list(range(60)),
    })
    p = flexplot("y ~ x", data=df)
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    assert smooth_layers[0].stat.params.get("method") == "glm"


def test_overlay_color_cycle_wraps_after_five():
    """Sixth overlay entry should re-use the first cycle color."""
    from pyflexplot.core import _OVERLAY_COLOR_CYCLE

    df = _make_df()
    overlay = [{"method": "loess"} for _ in range(6)]
    # All-loess overlays; six entries should wrap the cycle.
    p = flexplot("y ~ x + z", data=df, overlay=overlay)
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    # Primary + 6 overlays = 7 smooth layers.
    assert len(smooth_layers) == 7
    colors = [
        layer.geom.aes_params.get("color")
        for layer in smooth_layers[1:]  # skip primary
    ]
    # The 6 overlay colors should be the cycle in order, wrapping at index 5.
    expected = [_OVERLAY_COLOR_CYCLE[i % len(_OVERLAY_COLOR_CYCLE)] for i in range(6)]
    assert colors == expected


def test_bands_with_single_level_behaves_like_level():
    """bands=[0.90] should produce a single band, same as level=0.90."""
    df = _make_df()
    p_a = flexplot("y ~ x + z", data=df, bands=[0.90])
    p_b = flexplot("y ~ x + z", data=df, level=0.90)
    smooth_a = [l for l in p_a.layers if isinstance(l.geom, geom_smooth)]
    smooth_b = [l for l in p_b.layers if isinstance(l.geom, geom_smooth)]
    assert len(smooth_a) == len(smooth_b) == 1
    assert smooth_a[0].stat.params.get("level") == smooth_b[0].stat.params.get("level")


def test_diagnose_with_categorical_x_does_not_crash():
    """diagnose() may not handle categorical x gracefully (non-numeric)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=60),
        "x": rng.choice(["A", "B"], size=60),
    })
    # Should raise ValueError because the design matrix has no numeric
    # predictors \u2014 not silently produce wrong results.
    with pytest.raises(ValueError):
        diagnose("y ~ x", data=df, verbose=False)


# --- Docstring example sanity (lightweight) -------------------------------


def test_flexplot_docstring_examples_construct():
    """Verify the docstring Examples section produces working plot objects.

    Not full doctest; just confirms the snippets construct without error.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50)})

    # From the flexplot() docstring Examples block.
    p1 = flexplot("y ~ x", data=df)
    assert p1 is not None

    p2 = flexplot("y ~ x", data=df, bands=[0.5, 0.8, 0.95])
    assert p2 is not None

    p3 = flexplot(
        "y ~ x", data=df,
        overlay=[{"method": "loess", "label": "LOESS smoother"}],
    )
    assert p3 is not None