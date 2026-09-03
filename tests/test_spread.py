"""Tests for v0.6.4 spread= parameter on flexplot().

The spread parameter controls the dispersion marker for the discrete-x
branch (geom_jitter + summary). It mirrors R-flexplot's ``spread`` arg:
  - None / "ci": bootstrap CI (legacy default)
  - "stdev": mean +/- 1 SD
  - "range": min-max range
  - "iqr": Q1-Q3 IQR
  - "no": no summary layer at all
"""

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

from pyflexplot import flexplot
from pyflexplot.core import _VALID_SPREAD, _make_spread_fn


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_valid_spread_values():
    """_VALID_SPREAD contains the expected tokens (R aliases added v0.8.0)."""
    assert _VALID_SPREAD == frozenset(
        {None, "stdev", "range", "iqr", "no", "ci", "quartiles", "sterr"}
    )


def test_spread_r_aliases_map_correctly():
    """R-flexplot tokens map: quartiles == iqr; sterr == ci."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": ["a", "b", "c"] * 20,
        "y": rng.normal(size=60),
    })
    p_iqr = flexplot("y ~ x", data=df, spread="iqr")
    p_quartiles = flexplot("y ~ x", data=df, spread="quartiles")
    p_ci = flexplot("y ~ x", data=df, spread="ci")
    p_sterr = flexplot("y ~ x", data=df, spread="sterr")
    # quartiles produces the same layer schema as iqr (pointrange).
    types_iqr = sorted(l.geom.__class__.__name__ for l in p_iqr.layers)
    types_quartiles = sorted(l.geom.__class__.__name__ for l in p_quartiles.layers)
    assert types_iqr == types_quartiles
    # sterr produces the same layer schema as ci (stat_summary).
    types_ci = sorted(l.geom.__class__.__name__ for l in p_ci.layers)
    types_sterr = sorted(l.geom.__class__.__name__ for l in p_sterr.layers)
    assert types_ci == types_sterr


def test_flexplot_rejects_unknown_spread():
    """spread='bogus' raises ValueError."""
    df = pd.DataFrame({
        "x": ["a", "b", "c", "a", "b", "c"] * 5,
        "y": np.random.default_rng(0).normal(size=30),
    })
    with pytest.raises(ValueError, match="spread must be one of"):
        flexplot("y ~ x", data=df, spread="bogus")


# ---------------------------------------------------------------------------
# spread="no"
# ---------------------------------------------------------------------------


def test_spread_no_renders_only_jitter():
    """spread='no': only geom_jitter, no summary layer."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": ["a", "b", "c"] * 20,
        "y": rng.normal(size=60),
    })
    p = flexplot("y ~ x", data=df, spread="no")
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_jitter" in layer_types
    # No pointrange, no stat_summary-equivalent geom_pointrange layer.
    assert "geom_pointrange" not in layer_types


# ---------------------------------------------------------------------------
# spread="stdev" / "range" / "iqr" / "ci"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spread_value", ["stdev", "range", "iqr", "ci", None])
def test_spread_dispersion_values_render_a_summary_layer(spread_value):
    """Each spread value renders a ggplot with a summary layer (not 'no')."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": ["a", "b", "c"] * 20,
        "y": rng.normal(size=60),
    })
    p = flexplot("y ~ x", data=df, spread=spread_value)
    assert isinstance(p, ggplot)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_jitter" in layer_types
    # For spread in {None, "ci"}, the summary is stat_summary (rendered as
    # a different geom). For {stdev, range, iqr}, it's geom_pointrange.
    has_summary = (
        any("StatSummary" in t or t == "geom_pointrange" for t in layer_types)
        or len(layer_types) > 1
    )
    assert has_summary


# ---------------------------------------------------------------------------
# _make_spread_fn helper
# ---------------------------------------------------------------------------


def test_make_spread_fn_stdev_shape():
    """_make_spread_fn(mean, std) returns a DataFrame with y, ymin, ymax."""
    fn = _make_spread_fn(np.mean, lambda x: np.std(x, ddof=1))
    out = fn(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert list(out.columns) == ["y", "ymin", "ymax"]
    assert len(out) == 1
    # mean of [1..5] = 3; sample std = ~1.414
    assert abs(out["y"].iloc[0] - 3.0) < 1e-6
    half = float(np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1))
    assert abs(out["ymin"].iloc[0] - (3.0 - half)) < 1e-6
    assert abs(out["ymax"].iloc[0] - (3.0 + half)) < 1e-6


def test_make_spread_fn_range_shape():
    """_make_spread_fn(mean, (min,max)) returns center +/- extremes."""
    fn = _make_spread_fn(np.mean, lambda x: (np.min(x), np.max(x)))
    out = fn(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert out["y"].iloc[0] == 3.0
    assert out["ymin"].iloc[0] == 1.0
    assert out["ymax"].iloc[0] == 5.0


def test_make_spread_fn_iqr_shape():
    """_make_spread_fn(median, (Q1, Q3)) returns median with quartile range."""
    fn = _make_spread_fn(np.median, lambda x: (np.percentile(x, 25), np.percentile(x, 75)))
    out = fn(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert out["y"].iloc[0] == 3.0  # median
    assert out["ymin"].iloc[0] == 2.0  # Q1
    assert out["ymax"].iloc[0] == 4.0  # Q3


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_default_spread_is_none_and_uses_quartiles():
    """No spread= specified: R-default quartiles (median +/- IQR)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": ["a", "b", "c"] * 20,
        "y": rng.normal(size=60),
    })
    p_default = flexplot("y ~ x", data=df)
    p_quartiles = flexplot("y ~ x", data=df, spread="quartiles")
    # Same number of layers (both should produce jitter + summary).
    assert len(p_default.layers) == len(p_quartiles.layers)
    # Explicit CI remains available.
    p_ci = flexplot("y ~ x", data=df, spread="ci")
    assert len(p_ci.layers) == len(p_default.layers)