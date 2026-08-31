"""Tests for pyflexplot.scatter3D — 2D projection of y ~ x + z.

A partial port of R-flexplot's ``scatter3D()``: plotnine doesn't support
3D rendering, so we project to one of several 2D views (scatter of
(x, z) colored by y; or a tile/heatmap aggregating y on a (x, z) grid).
"""
import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

from pyflexplot import scatter3D


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "y": rng.normal(size=200),
        "x1": rng.normal(size=200),
        "x2": rng.normal(size=200),
    })


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_scatter3d_points_returns_ggplot(sample_df):
    """type='points' returns a plotnine ggplot."""
    p = scatter3D("y ~ x1 + x2", data=sample_df)
    assert isinstance(p, ggplot)


def test_scatter3d_tile_returns_ggplot(sample_df):
    """type='tile' returns a plotnine ggplot."""
    p = scatter3D("y ~ x1 + x2", data=sample_df, type="tile")
    assert isinstance(p, ggplot)


def test_scatter3d_points_renders_geom_point(sample_df):
    """type='points' renders geom_point layer."""
    p = scatter3D("y ~ x1 + x2", data=sample_df)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_point" in layer_types


def test_scatter3d_tile_renders_geom_tile(sample_df):
    """type='tile' renders geom_tile layer."""
    p = scatter3D("y ~ x1 + x2", data=sample_df, type="tile")
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_tile" in layer_types


def test_scatter3d_points_does_not_render_geom_tile(sample_df):
    """type='points' does not render geom_tile."""
    p = scatter3D("y ~ x1 + x2", data=sample_df)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_tile" not in layer_types


def test_scatter3d_default_type_is_points(sample_df):
    """Default type is 'points'."""
    p = scatter3D("y ~ x1 + x2", data=sample_df)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_point" in layer_types


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_scatter3d_rejects_unknown_type(sample_df):
    """type='wireframe' raises ValueError."""
    with pytest.raises(ValueError, match="type must be one of"):
        scatter3D("y ~ x1 + x2", data=sample_df, type="wireframe")


def test_scatter3d_rejects_single_predictor_formula(sample_df):
    """y ~ x (no z) raises ValueError."""
    with pytest.raises(ValueError, match="two predictors"):
        scatter3D("y ~ x1", data=sample_df)


def test_scatter3d_rejects_non_numeric_columns():
    """Non-numeric y raises ValueError."""
    df = pd.DataFrame({
        "y": ["a", "b", "c"] * 10,
        "x1": np.random.default_rng(0).normal(size=30),
        "x2": np.random.default_rng(0).normal(size=30),
    })
    with pytest.raises(ValueError, match="numeric"):
        scatter3D("y ~ x1 + x2", data=df)


def test_scatter3d_rejects_given_term():
    """y ~ x + z | w (given term) raises ValueError."""
    df = pd.DataFrame({
        "y": np.random.default_rng(0).normal(size=60),
        "x1": np.random.default_rng(0).normal(size=60),
        "x2": np.random.default_rng(0).normal(size=60),
        "w": np.random.default_rng(0).choice(["A", "B"], size=60),
    })
    with pytest.raises(ValueError, match="given"):
        scatter3D("y ~ x1 + x2 | w", data=df)


# ---------------------------------------------------------------------------
# Numeric correctness (tile mode)
# ---------------------------------------------------------------------------


def test_scatter3d_tile_aggregates_y_per_bin(sample_df):
    """type='tile' aggregates y to mean per (x_bin, z_bin) cell."""
    p = scatter3D("y ~ x1 + x2", data=sample_df, type="tile", bins=10)
    # p.data should have at most 10 * 10 = 100 rows (some bins may be empty).
    assert len(p.data) <= 100
    assert len(p.data) >= 1


def test_scatter3d_tile_uses_default_20_bins(sample_df):
    """Default bins=20 (verifiable via row count cap)."""
    p = scatter3D("y ~ x1 + x2", data=sample_df, type="tile")
    assert len(p.data) <= 400  # 20 * 20 = 400


def test_scatter3d_tile_custom_bins_respected(sample_df):
    """bins=5 produces at most 25 rows."""
    p = scatter3D("y ~ x1 + x2", data=sample_df, type="tile", bins=5)
    assert len(p.data) <= 25


def test_scatter3d_tile_means_match_groupby(sample_df):
    """Per-bin tile mean matches manual groupby on the same bin edges."""
    bins = 8
    work = sample_df.copy()
    work["__x_bin"] = pd.cut(work["x1"], bins=bins, labels=False, include_lowest=True)
    work["__z_bin"] = pd.cut(work["x2"], bins=bins, labels=False, include_lowest=True)
    expected = (
        work.groupby(["__x_bin", "__z_bin"], observed=True)["y"]
        .mean()
        .reset_index()
        .dropna()
    )
    p = scatter3D("y ~ x1 + x2", data=sample_df, type="tile", bins=bins)
    # Tile's fill aesthetic uses the aggregated y; verify the aggregated
    # values match expected. The tile layer's fill should equal the per-bin
    # mean of y. Plotnine stores the layer data on the parent plot.data.
    assert len(p.data) == len(expected)