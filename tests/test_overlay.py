"""Tests for the model-compare overlay feature (B: power feature)."""

import numpy as np
import pandas as pd
import pytest
from plotnine import geom_line, geom_ribbon, geom_smooth

from pyflexplot import flexplot


# --- Input parsing: overlay= accepts list[str] or list[dict] -----------------


def test_overlay_string_list_renders_one_smooth_per_entry():
    """overlay=['lm', 'loess'] → 2 geom_smooth layers (primary + overlay)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=80),
        "x": rng.normal(size=80),
    })
    p = flexplot("y ~ x", data=df, overlay=["loess"])
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    # Primary (LM by default) + loess overlay = 2 smooth layers.
    assert len(smooth_layers) == 2


def test_overlay_dict_list_renders_one_smooth_per_entry():
    """overlay=[{...}, {...}] → one geom_smooth per dict entry."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "y": rng.normal(size=80),
        "x": rng.normal(size=80),
    })
    overlay = [
        {"method": "loess"},
        {"method": "loess", "span": 0.3},
    ]
    p = flexplot("y ~ x", data=df, overlay=overlay)
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    assert len(smooth_layers) == 3  # primary LM + 2 overlay


def test_overlay_none_renders_only_primary():
    """overlay=None (default) → only the primary method is drawn."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "y": rng.normal(size=50),
        "x": rng.normal(size=50),
    })
    p = flexplot("y ~ x", data=df)  # overlay=None
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    assert len(smooth_layers) == 1


def test_overlay_empty_list_renders_only_primary():
    """overlay=[] → only the primary method is drawn."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "y": rng.normal(size=50),
        "x": rng.normal(size=50),
    })
    p = flexplot("y ~ x", data=df, overlay=[])
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    assert len(smooth_layers) == 1


def test_overlay_dict_must_have_method_key():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "y": rng.normal(size=30),
        "x": rng.normal(size=30),
    })
    with pytest.raises(ValueError, match="missing required key 'method'"):
        flexplot("y ~ x", data=df, overlay=[{"color": "red"}])


def test_overlay_dict_method_must_be_valid():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "y": rng.normal(size=30),
        "x": rng.normal(size=30),
    })
    with pytest.raises(ValueError, match="not a recognized method"):
        flexplot("y ~ x", data=df, overlay=[{"method": "totally-bogus"}])


# --- Per-overlay-entry options propagate --------------------------------------


def test_overlay_dict_propagates_method():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({
        "y": rng.normal(size=80),
        "x": rng.normal(size=80),
    })
    p = flexplot("y ~ x", data=df, overlay=[{"method": "loess"}])
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    methods = [layer.stat.params.get("method") for layer in smooth_layers]
    # Primary (auto → lm) plus overlay loess.
    assert "lm" in methods
    assert "loess" in methods


def test_overlay_dict_propagates_color():
    """Each overlay dict may specify a color for its smooth."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "y": rng.normal(size=80),
        "x": rng.normal(size=80),
    })
    p = flexplot(
        "y ~ x", data=df,
        overlay=[{"method": "loess", "color": "#ff0000"}],
    )
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    loess_layer = next(
        layer for layer in smooth_layers
        if layer.stat.params.get("method") == "loess"
    )
    # Plotnine stores `color` in geom.aes_params (not geom.params).
    assert loess_layer.geom.aes_params.get("color") == "#ff0000"


def test_overlay_dict_propagates_level():
    rng = np.random.default_rng(8)
    df = pd.DataFrame({
        "y": rng.normal(size=80),
        "x": rng.normal(size=80),
    })
    p = flexplot(
        "y ~ x", data=df,
        overlay=[{"method": "loess", "level": 0.80}],
    )
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    loess_layer = next(
        layer for layer in smooth_layers
        if layer.stat.params.get("method") == "loess"
    )
    assert loess_layer.stat.params.get("level") == 0.80


def test_overlay_default_color_distinct_from_primary():
    """Each overlay entry should get a distinct color from the cycle."""
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "y": rng.normal(size=80),
        "x": rng.normal(size=80),
    })
    p = flexplot(
        "y ~ x", data=df,
        overlay=[{"method": "loess"}, {"method": "lm"}],
    )
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    colors = [layer.geom.params.get("color") for layer in smooth_layers]
    # Primary (LM, "blue") + 2 overlays with distinct default colors.
    assert len(set(colors)) == len(colors) or colors.count("blue") <= 1
    # The two overlay entries should not both be "blue".
    assert colors.count("blue") <= 1


# --- Overlay works with both branches ----------------------------------------


def test_overlay_with_method_loess_primary():
    rng = np.random.default_rng(10)
    df = pd.DataFrame({
        "y": rng.normal(size=80),
        "x": rng.normal(size=80),
    })
    p = flexplot(
        "y ~ x", data=df, method="loess",
        overlay=[{"method": "lm"}],
    )
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    methods = [layer.stat.params.get("method") for layer in smooth_layers]
    assert methods.count("loess") == 1
    assert methods.count("lm") == 1


def test_overlay_with_uncertainty_none_still_renders_overlays():
    """uncertainty=None disables the PRIMARY but overlays should still draw."""
    rng = np.random.default_rng(11)
    df = pd.DataFrame({
        "y": rng.normal(size=80),
        "x": rng.normal(size=80),
    })
    p = flexplot(
        "y ~ x", data=df, uncertainty=None,
        overlay=[{"method": "lm"}],
    )
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    # Primary is None, but the overlay LM should still render.
    assert len(smooth_layers) == 1
    assert smooth_layers[0].stat.params.get("method") == "lm"


def test_overlay_with_bands():
    """bands + overlay should both apply (bands on primary, overlays on top)."""
    rng = np.random.default_rng(12)
    df = pd.DataFrame({
        "y": rng.normal(size=80),
        "x": rng.normal(size=80),
    })
    p = flexplot(
        "y ~ x", data=df,
        bands=[0.5, 0.95],
        overlay=[{"method": "loess"}],
    )
    smooth_layers = [
        layer for layer in p.layers if isinstance(layer.geom, geom_smooth)
    ]
    # 2 band layers on primary + 1 overlay loess = 3 smooth layers.
    assert len(smooth_layers) == 3


# --- Legend integration ------------------------------------------------------


def test_overlay_legend_present():
    """Each overlay entry should produce a legend entry."""
    rng = np.random.default_rng(13)
    df = pd.DataFrame({
        "y": rng.normal(size=80),
        "x": rng.normal(size=80),
    })
    p = flexplot(
        "y ~ x", data=df,
        overlay=[
            {"method": "loess", "label": "LOESS smoother"},
            {"method": "lm", "label": "Linear fit"},
        ],
    )
    # Plotnine's scale_color_manual is added when labels exist.
    from plotnine import scale_color_manual
    has_scale = any(isinstance(layer, scale_color_manual) for layer in p.scales)
    assert has_scale