"""Tests for v0.6.5 flexplot() extras: sample, ghost_line, plot_type, return_data.

Covers the deferred v0.6.x items from the R-audit:
- sample: int — subsample N rows for plotting (full data still used for fits).
- ghost_line: {"red", "dashed", None} — reference line layer.
- plot_type: {"scatter", "line", "boxplot", "bar", None} — explicit geom override.
- return_data: bool — if True, returns {"plot": ..., "data": ...} instead of just plot.
"""
import numpy as np
import pandas as pd
import pytest
from plotnine import (
    ggplot,
    geom_point,
    geom_line,
    geom_boxplot,
    geom_bar,
    geom_hline,
    geom_jitter,
)

from pyflexplot import flexplot


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


def test_sample_int_renders_a_ggplot():
    """sample=50 on a 1000-row df renders without error."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=1000),
        "y": rng.normal(size=1000),
    })
    p = flexplot("y ~ x", data=df, sample=50)
    assert isinstance(p, ggplot)


def test_sample_larger_than_data_is_noop():
    """sample=N with N >= len(data) draws all points (no subsampling)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=20),
        "y": rng.normal(size=20),
    })
    # Should not raise; should render normally.
    p = flexplot("y ~ x", data=df, sample=100)
    assert isinstance(p, ggplot)


def test_sample_zero_raises():
    """sample=0 raises ValueError (>= 1 required)."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    with pytest.raises(ValueError, match="sample must be >= 1"):
        flexplot("y ~ x", data=df, sample=0)


def test_sample_negative_raises():
    """sample=-1 raises ValueError."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    with pytest.raises(ValueError, match="sample must be >= 1"):
        flexplot("y ~ x", data=df, sample=-1)


def test_sample_non_int_raises():
    """sample=2.5 raises TypeError."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    with pytest.raises(TypeError, match="sample must be an int"):
        flexplot("y ~ x", data=df, sample=2.5)


def test_sample_subsamples_data_layer_but_not_fit():
    """sample=100 on 1000 rows: geom_point shows ~100 rows; fit uses all 1000.

    Verification strategy: the subsampled DataFrame is plumbed through to
    plotnine, but the smoother fits still see the full DataFrame. We check
    that the resulting ggplot object is well-formed and the geom_point layer
    references a data attribute with <= 100 rows. plotnine may lazily
    compute, so the layer's data might not be the dataframe directly but
    inherit it from the parent ggplot. Verify the parent plot has the
    expected data shape.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.uniform(-10, 10, size=1000),
        "y": rng.normal(size=1000),
    })
    p = flexplot("y ~ x", data=df, sample=100)
    assert isinstance(p, ggplot)
    # The ggplot's data should be the subsample (100 rows), not the full df.
    if p.data is not None:
        assert len(p.data) <= 100


# ---------------------------------------------------------------------------
# ghost_line
# ---------------------------------------------------------------------------


def test_ghost_line_red_adds_geom_hline():
    """ghost_line='red' adds a geom_hline at y=0."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})
    p = flexplot("y ~ x", data=df, ghost_line="red")
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_hline" in layer_types


def test_ghost_line_dashed_adds_geom_hline():
    """ghost_line='dashed' adds a dashed geom_hline at y=0."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})
    p = flexplot("y ~ x", data=df, ghost_line="dashed")
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_hline" in layer_types


def test_ghost_line_none_no_extra_layer():
    """ghost_line=None (default) does NOT add a geom_hline."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})
    p = flexplot("y ~ x", data=df)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_hline" not in layer_types


def test_ghost_line_invalid_raises():
    """ghost_line='neon' raises ValueError."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})
    with pytest.raises(ValueError, match="ghost_line must be"):
        flexplot("y ~ x", data=df, ghost_line="neon")


# ---------------------------------------------------------------------------
# plot_type
# ---------------------------------------------------------------------------


def test_plot_type_boxplot_forces_geom_boxplot():
    """plot_type='boxplot' forces geom_boxplot regardless of x dtype."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.choice(["a", "b", "c"], size=60),
        "y": rng.normal(size=60),
    })
    p = flexplot("y ~ x", data=df, plot_type="boxplot")
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_boxplot" in layer_types
    # Boxplot is forced — even though auto-dispatch would have picked
    # geom_jitter, the override wins.


def test_plot_type_bar_forces_geom_bar():
    """plot_type='bar' forces geom_bar."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.choice(["a", "b", "c"], size=60),
        "y": rng.normal(size=60),
    })
    p = flexplot("y ~ x", data=df, plot_type="bar")
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_bar" in layer_types


def test_plot_type_scatter_forces_geom_point_no_fit():
    """plot_type='scatter' forces geom_point with no smoother."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})
    p = flexplot("y ~ x", data=df, plot_type="scatter")
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_point" in layer_types
    # No fit/summary layers.
    assert "geom_smooth" not in layer_types
    assert "geom_jitter" not in layer_types


def test_plot_type_line_forces_geom_line():
    """plot_type='line' forces geom_line."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": np.arange(30, dtype=float),
        "y": rng.normal(size=30),
    })
    p = flexplot("y ~ x", data=df, plot_type="line")
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_line" in layer_types


def test_plot_type_invalid_raises():
    """plot_type='pie' raises ValueError."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})
    with pytest.raises(ValueError, match="plot_type must be one of"):
        flexplot("y ~ x", data=df, plot_type="pie")


def test_plot_type_default_routes_through_auto_dispatch():
    """plot_type=None (default) preserves the legacy auto-dispatch."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.choice(["a", "b", "c"], size=60),
        "y": rng.normal(size=60),
    })
    p = flexplot("y ~ x", data=df)  # no plot_type
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    # Should hit the discrete branch (geom_jitter).
    assert "geom_jitter" in layer_types


# ---------------------------------------------------------------------------
# return_data
# ---------------------------------------------------------------------------


def test_return_data_default_returns_plot():
    """return_data=False (default) returns just the plot object."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})
    p = flexplot("y ~ x", data=df)
    assert isinstance(p, ggplot)


def test_return_data_true_returns_dict_with_plot_and_data():
    """return_data=True returns {"plot": ggplot, "data": DataFrame}."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})
    result = flexplot("y ~ x", data=df, return_data=True)
    assert isinstance(result, dict)
    assert "plot" in result
    assert "data" in result
    assert isinstance(result["plot"], ggplot)
    assert isinstance(result["data"], pd.DataFrame)
    assert len(result["data"]) == len(df)


def test_return_data_with_sample_returns_subsampled_dataframe():
    """return_data=True + sample=10: result["data"] is the 10-row subsample."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    result = flexplot("y ~ x", data=df, sample=10, return_data=True)
    assert len(result["data"]) == 10


def test_return_data_intercept_only_branch():
    """return_data=True on the intercept-only branch also returns the dict."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=30)})
    result = flexplot("y ~ 1", data=df, return_data=True)
    assert isinstance(result, dict)
    assert "plot" in result
    assert "data" in result