"""Tests for v0.6.6 plot.string and related parameters on flexplot().

plot.string: dict override for axis/legend labels.
related: bool — link related panels via shared scales (no-op on the
  Python side because plotnine facets share scales by default; the flag
  is accepted for R-parity but doesn't change behavior).
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
        "y": np.linspace(-3, 3, n) + rng.normal(scale=0.3, size=n),
    })


# ---------------------------------------------------------------------------
# plot.string
# ---------------------------------------------------------------------------


def test_plot_string_none_no_label_override():
    """plot_string=None (default) preserves default labels.

    plotnine's default behavior is to leave labels as None (plotnine then
    uses the column names from the data). We verify that with no
    plot_string, p.labels.x and p.labels.y are None.
    """
    df = _sample_df()
    p = flexplot("y ~ x", data=df)
    assert isinstance(p, ggplot)
    # Default labels are None (plotnine substitutes the column name).
    assert p.labels.x is None
    assert p.labels.y is None


def test_plot_string_dict_overrides_x_label():
    """plot_string={'x': 'Custom X'} overrides the x label."""
    df = _sample_df()
    p = flexplot("y ~ x", data=df, plot_string={"x": "Custom X"})
    assert p.labels.x == "Custom X"


def test_plot_string_dict_overrides_y_label():
    """plot_string={'y': 'Custom Y'} overrides the y label."""
    df = _sample_df()
    p = flexplot("y ~ x", data=df, plot_string={"y": "Custom Y"})
    assert p.labels.y == "Custom Y"


def test_plot_string_dict_overrides_title():
    """plot_string={'title': 'My Title'} overrides the title."""
    df = _sample_df()
    p = flexplot("y ~ x", data=df, plot_string={"title": "My Title"})
    assert p.labels.title == "My Title"


def test_plot_string_dict_with_multiple_keys():
    """plot_string with multiple keys applies all overrides."""
    df = _sample_df()
    p = flexplot(
        "y ~ x", data=df,
        plot_string={
            "x": "Predictor",
            "y": "Response",
            "title": "Experiment 1",
        },
    )
    assert p.labels.x == "Predictor"
    assert p.labels.y == "Response"
    assert p.labels.title == "Experiment 1"


def test_plot_string_ignores_unknown_keys_silently():
    """plot_string keys outside {x, y, title, subtitle, caption, color} are
    silently dropped (plotnine's labs() rejects unknown keys)."""
    df = _sample_df()
    # Should not raise.
    p = flexplot(
        "y ~ x", data=df,
        plot_string={"x": "Custom", "foo": "ignored", "bar": "also ignored"},
    )
    assert p.labels.x == "Custom"


def test_plot_string_non_dict_raises():
    """plot_string="not a dict" raises TypeError."""
    df = _sample_df()
    with pytest.raises(TypeError, match="must be a dict"):
        flexplot("y ~ x", data=df, plot_string="x = Custom")


def test_plot_string_non_string_value_raises():
    """plot_string with a non-string value raises TypeError."""
    df = _sample_df()
    with pytest.raises(TypeError, match="must all be strings"):
        flexplot("y ~ x", data=df, plot_string={"x": 42})


# ---------------------------------------------------------------------------
# related
# ---------------------------------------------------------------------------


def _paired_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    g1 = rng.normal(size=n)
    g2 = g1 + rng.normal(scale=0.5, size=n) + 0.2
    return pd.DataFrame({
        "subject": np.tile(np.arange(n), 2),
        "group": ["a"] * n + ["b"] * n,
        "y": np.concatenate([g1, g2]),
    })


def test_related_default_false():
    """related=False (default) renders normally without error."""
    df = _sample_df()
    p = flexplot("y ~ x", data=df)
    assert isinstance(p, ggplot)


def test_related_true_renders_difference_plot():
    """related=True with a two-level predictor produces a paired difference plot."""
    df = _paired_df()
    p = flexplot("y ~ group", data=df, related=True)
    assert isinstance(p, ggplot)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_hline" in layer_types
    assert "geom_pointrange" in layer_types
    p.draw()


def test_related_true_returns_difference_data():
    """related=True with return_data=True exposes the paired differences."""
    df = _paired_df(n=10)
    out = flexplot("y ~ group", data=df, related=True, return_data=True)
    assert "plot" in out and "data" in out
    assert out["data"].shape[0] == 10
    assert out["data"].columns[0].startswith("Difference")


def test_related_with_numeric_two_level_predictor():
    """related=True works when the two-level predictor is coded 0/1 numeric."""
    df = _paired_df()
    df["group"] = df["group"].map({"a": 0, "b": 1})
    p = flexplot("y ~ group", data=df, related=True)
    assert isinstance(p, ggplot)
    p.draw()


def test_related_with_facets_raises():
    """related=True with a faceted formula raises a clear error."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.choice(["a", "b"], size=120),
        "y": rng.normal(size=120),
        "g": rng.choice(["A", "B", "C"], size=120),
    })
    with pytest.raises(ValueError, match="related=True is only supported"):
        flexplot("y ~ x | g", data=df, related=True)


def test_related_non_two_level_predictor_raises():
    """related=True with a predictor that has more than two levels raises."""
    df = _sample_df()
    df["x"] = pd.cut(df["x"], bins=4).astype(str)
    with pytest.raises(ValueError, match="requires exactly 2 levels"):
        flexplot("y ~ x", data=df, related=True)


def test_related_non_bool_raises():
    """related='yes' raises TypeError."""
    df = _sample_df()
    with pytest.raises(TypeError, match="related must be a bool"):
        flexplot("y ~ x", data=df, related="yes")