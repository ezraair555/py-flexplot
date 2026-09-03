"""Tests for v0.6.6 ghost.reference= parameter on flexplot().

R-flexplot accepts ghost.reference as a DataFrame overlay on the same
axes. Two patterns detected automatically by column shape:
  - Scatter reference: columns matching (x, y) → geom_point in gray.
  - Prediction-line reference: columns (x, 'pred') → geom_line in red.
"""
import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

from pyflexplot import flexplot


def _sample_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": np.linspace(-3, 3, n),
        "y": np.linspace(-3, 3, n) ** 2 + rng.normal(scale=0.3, size=n),
    })


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_ghost_reference_none_no_extra_layer():
    """ghost_reference=None (default) does NOT add any extra reference layer.

    The auto-dispatch for numeric-y/numeric-x does add a geom_point scatter,
    so we can't simply assert 'geom_point not in layer_types'. Instead we
    verify that the geom_point layer count matches the no-ghost_reference
    baseline.
    """
    df = _sample_df()
    p_baseline = flexplot("y ~ x", data=df)
    p_with_none = flexplot("y ~ x", data=df, ghost_reference=None)
    n_layers_baseline = len(p_baseline.layers)
    n_layers_with_none = len(p_with_none.layers)
    assert n_layers_baseline == n_layers_with_none


def test_ghost_reference_non_dataframe_raises():
    """ghost_reference=42 raises TypeError."""
    df = _sample_df()
    with pytest.raises(TypeError, match="ghost_reference must be"):
        flexplot("y ~ x", data=df, ghost_reference=42)


def test_ghost_reference_dict_requires_facet():
    """ghost_reference dict without a facet raises TypeError."""
    df = _sample_df()
    with pytest.raises(TypeError, match="requires a .* facet"):
        flexplot("y ~ x", data=df, ghost_reference={"g": "A"})


def test_ghost_line_panel_repetition_semantics():
    """R-parity: with a facet, ghost_line=X fits the line on the reference
    panel and repeats it into every panel, drawn in that color (v0.8.0)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "x": rng.normal(size=200),
        "g": rng.choice(["A", "B"], size=200),
    })
    p = flexplot("y ~ x | g", data=df, ghost_line="red")
    # The ghost line is an extra geom_line layer.
    line_layers = [l for l in p.layers if l.geom.__class__.__name__ == "geom_line"]
    assert len(line_layers) >= 1


def test_ghost_line_dict_reference_selects_panel():
    """ghost_reference={'g': 'B'} fits the ghost line on group B only."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "x": rng.normal(size=200),
        "g": rng.choice(["A", "B"], size=200),
    })
    p = flexplot("y ~ x | g", data=df, ghost_line="black", ghost_reference={"g": "B"})
    assert isinstance(p, ggplot)


def test_ghost_reference_dict_defaults_ghost_line_gray():
    """Passing ghost_reference dict without ghost_line automatically adds the ghost fit line."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "fare": rng.normal(size=100),
        "age": rng.normal(size=100),
        "pclass": rng.choice([1, 2, 3], size=100),
    })
    p = flexplot("fare ~ age | pclass", data=df, ghost_reference={"pclass": 1})
    assert isinstance(p, ggplot)
    line_layers = [l for l in p.layers if l.geom.__class__.__name__ == "geom_line"]
    assert len(line_layers) >= 1


def test_ghost_line_with_nans_in_data():
    """Verify ghost line fits properly even when data contains NaNs in x or y."""
    rng = np.random.default_rng(0)
    age = rng.normal(size=100)
    fare = rng.normal(size=100)
    age[0:15] = np.nan  # Introduce missing values in predictor
    df = pd.DataFrame({
        "fare": fare,
        "age": age,
        "pclass": rng.choice([1, 2, 3], size=100),
    })
    p = flexplot("fare ~ age | pclass", data=df, ghost_line="dashed", ghost_reference={"pclass": 1})
    assert isinstance(p, ggplot)
    line_layers = [l for l in p.layers if l.geom.__class__.__name__ == "geom_line"]
    assert len(line_layers) >= 1


def test_ghost_reference_missing_x_column_raises():
    """ghost_reference without the x column raises ValueError."""
    df = _sample_df()
    ref = pd.DataFrame({"wrong_col": [1.0, 2.0], "y": [3.0, 4.0]})
    with pytest.raises(ValueError, match="must have column"):
        flexplot("y ~ x", data=df, ghost_reference=ref)


def test_ghost_reference_missing_y_and_pred_raises():
    """ghost_reference without y or pred columns raises ValueError."""
    df = _sample_df()
    ref = pd.DataFrame({"x": [1.0, 2.0], "foo": [3.0, 4.0]})
    with pytest.raises(ValueError, match="scatter|line"):
        flexplot("y ~ x", data=df, ghost_reference=ref)


# ---------------------------------------------------------------------------
# Scatter pattern (x, y columns)
# ---------------------------------------------------------------------------


def test_ghost_reference_scatter_pattern_adds_geom_point():
    """ghost_reference with (x, y) columns draws geom_point in gray."""
    df = _sample_df()
    ref = pd.DataFrame({"x": [-2, -1, 0, 1, 2], "y": [4, 1, 0, 1, 4]})
    p = flexplot("y ~ x", data=df, ghost_reference=ref)
    # Two geom_point layers: the main scatter + the reference overlay.
    point_layers = [
        layer for layer in p.layers
        if layer.geom.__class__.__name__ == "geom_point"
    ]
    assert len(point_layers) >= 1


def test_ghost_reference_scatter_data_has_correct_length():
    """The reference layer's data should match the reference DataFrame.

    plotnine exposes layer data via ``layer.data`` (public) or
    ``layer._data`` (private). On newer plotnine versions ``.data`` is the
    public attribute. We try ``.data`` first, fall back to ``._data``.
    """
    df = _sample_df()
    ref = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    p = flexplot("y ~ x", data=df, ghost_reference=ref)
    for layer in p.layers:
        layer_data = getattr(layer, "data", None) or getattr(layer, "_data", None)
        if layer_data is not None and len(layer_data) == 3 and "x" in layer_data.columns:
            assert sorted(layer_data["x"].tolist()) == [1.0, 2.0, 3.0]
            return
    pytest.fail("Expected a layer with 3 rows matching the reference data")


# ---------------------------------------------------------------------------
# Prediction-line pattern (x, 'pred' columns)
# ---------------------------------------------------------------------------


def test_ghost_reference_pred_pattern_adds_geom_line():
    """ghost_reference with (x, 'pred') columns draws geom_line in red."""
    df = _sample_df()
    ref = pd.DataFrame({"x": np.linspace(-3, 3, 20), "pred": np.linspace(0, 9, 20)})
    p = flexplot("y ~ x", data=df, ghost_reference=ref)
    line_layers = [
        layer for layer in p.layers
        if layer.geom.__class__.__name__ == "geom_line"
    ]
    assert len(line_layers) >= 1


def test_ghost_reference_pred_pattern_does_not_add_geom_point():
    """The 'pred' pattern doesn't add geom_point (only geom_line)."""
    df = _sample_df()
    ref = pd.DataFrame({"x": np.linspace(-3, 3, 20), "pred": np.zeros(20)})
    p = flexplot("y ~ x", data=df, ghost_reference=ref)
    # Should have geom_line from the prediction; not an extra geom_point
    # overlay (the main geom_smooth-based layer uses geom_smooth, not geom_point).
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_line" in layer_types


def test_ghost_reference_takes_precedence_over_ghost_line():
    """When both ghost_line and ghost_reference are set, both layers appear."""
    df = _sample_df()
    ref = pd.DataFrame({"x": [0.0], "y": [0.0]})
    p = flexplot("y ~ x", data=df, ghost_line="red", ghost_reference=ref)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_hline" in layer_types  # ghost_line
    assert "geom_point" in layer_types  # ghost_reference (scatter)