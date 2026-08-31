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


def test_related_default_false():
    """related=False (default) renders normally without error."""
    df = _sample_df()
    p = flexplot("y ~ x", data=df)
    assert isinstance(p, ggplot)


def test_related_true_renders_without_change():
    """related=True is a no-op (plotnine already shares scales by default)."""
    df = _sample_df()
    p_default = flexplot("y ~ x", data=df)
    p_related = flexplot("y ~ x", data=df, related=True)
    # Same number of layers; related shouldn't change the visual output
    # because plotnine facets share scales by default.
    assert len(p_default.layers) == len(p_related.layers)


def test_related_with_facets_still_works():
    """related=True with a faceted formula doesn't error."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=120),
        "y": rng.normal(size=120),
        "g": rng.choice(["A", "B", "C"], size=120),
    })
    p = flexplot("y ~ x | g", data=df, related=True)
    assert isinstance(p, ggplot)


def test_related_non_bool_raises():
    """related='yes' raises TypeError."""
    df = _sample_df()
    with pytest.raises(TypeError, match="related must be a bool"):
        flexplot("y ~ x", data=df, related="yes")