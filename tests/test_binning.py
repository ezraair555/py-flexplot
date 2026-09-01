"""Tests for v0.6.4 numeric-x binning: ``bins`` / ``labels`` / ``breaks``.

Covers the new ``flexplot()`` parameters added in v0.6.4 to close the
auto-bin gap with R-flexplot. When x is numeric and not already discrete,
passing ``bins=N`` or ``breaks=[...]`` discretizes x via ``pd.cut`` so
the discrete-x branch (geom_jitter + stat_summary) applies.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

from pyflexplot import flexplot
from pyflexplot.core import _validate_binning_params, _maybe_bin_numeric_x


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_binning_no_args_is_noop():
    """All None: no error."""
    _validate_binning_params(None, None, None, pd.Series([1, 2, 3]))


def test_validate_binning_rejects_non_int_bins():
    """bins=2.5 raises TypeError."""
    with pytest.raises(TypeError, match="bins must be an int"):
        _validate_binning_params(2.5, None, None, pd.Series([1, 2, 3]))


def test_validate_binning_rejects_bins_lt_2():
    """bins=1 raises ValueError."""
    with pytest.raises(ValueError, match="bins must be >= 2"):
        _validate_binning_params(1, None, None, pd.Series([1, 2, 3]))


def test_validate_binning_rejects_short_breaks():
    """breaks with < 2 points raises ValueError."""
    with pytest.raises(ValueError, match=">= 2 cut points"):
        _validate_binning_params(None, None, [1.0], pd.Series([1, 2, 3]))


def test_validate_binning_rejects_non_monotonic_breaks():
    """Non-monotonic breaks raise ValueError."""
    with pytest.raises(ValueError, match="monotonically increasing"):
        _validate_binning_params(None, None, [1.0, 0.5, 2.0], pd.Series([1, 2, 3]))


def test_validate_binning_rejects_wrong_labels_length_with_breaks():
    """labels length must equal len(breaks) - 1."""
    with pytest.raises(ValueError, match="labels length"):
        _validate_binning_params(
            None, ["a", "b", "c"], [1.0, 2.0, 3.0], pd.Series([1, 2, 3])
        )


def test_validate_binning_rejects_wrong_labels_length_with_bins():
    """labels length must equal bins when bins is given."""
    with pytest.raises(ValueError, match="labels length"):
        _validate_binning_params(
            3, ["a", "b"], None, pd.Series([1, 2, 3])
        )


def test_validate_binning_rejects_non_string_labels():
    """labels entries must be strings."""
    with pytest.raises(TypeError, match="must all be strings"):
        _validate_binning_params(
            None, ["a", 2, "c"], [1.0, 2.0, 3.0], pd.Series([1, 2, 3])
        )


def test_validate_binning_warns_when_both_bins_and_breaks():
    """Setting both bins and breaks emits a UserWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _validate_binning_params(3, None, [1.0, 2.0, 3.0], pd.Series([1, 2, 3]))
    msgs = [str(w.message) for w in caught if "breaks takes precedence" in str(w.message)]
    assert len(msgs) == 1


# ---------------------------------------------------------------------------
# _maybe_bin_numeric_x
# ---------------------------------------------------------------------------


def test_maybe_bin_no_args_returns_copy_unchanged():
    """No bins/breaks: returns (copy, False) and x is unmodified."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out, was_binned = _maybe_bin_numeric_x(df, "x")
    assert was_binned is False
    assert out["x"].tolist() == df["x"].tolist()


def test_maybe_bin_with_bins_creates_n_levels():
    """bins=3 creates 3 discrete levels in x."""
    df = pd.DataFrame({"x": np.linspace(0, 10, 11)})
    out, was_binned = _maybe_bin_numeric_x(df, "x", bins=3)
    assert was_binned is True
    n_unique = out["x"].nunique()
    assert n_unique == 3


def test_maybe_bin_with_breaks_uses_explicit_cuts():
    """breaks=[0, 5, 10] creates 2 levels."""
    df = pd.DataFrame({"x": [0.5, 2.0, 4.5, 5.5, 8.0, 9.5]})
    out, was_binned = _maybe_bin_numeric_x(df, "x", breaks=[0, 5, 10])
    assert was_binned is True
    n_unique = out["x"].nunique()
    assert n_unique == 2


def test_maybe_bin_with_labels_uses_custom_labels():
    """labels override default cut labels."""
    df = pd.DataFrame({"x": [0.5, 4.5, 9.5]})
    out, was_binned = _maybe_bin_numeric_x(
        df, "x", breaks=[0, 5, 10], labels=["low", "high"]
    )
    assert was_binned is True
    assert set(out["x"].unique()) == {"low", "high"}


# ---------------------------------------------------------------------------
# Integration: flexplot(bins=...)
# ---------------------------------------------------------------------------


def test_flexplot_bins_routes_through_discrete_branch():
    """bins=4 on numeric x produces geom_jitter + a summary layer (discrete branch)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.uniform(0, 100, size=80),
        "y": rng.normal(size=80),
    })
    p = flexplot("y ~ x", data=df, bins=4)
    assert isinstance(p, ggplot)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    # discrete branch: jitter + summary. plotnine's stat_summary is
    # implemented as geom_pointrange (the class name); check for either.
    assert "geom_jitter" in layer_types
    assert any(t in {"stat_summary", "geom_pointrange"} for t in layer_types)


def test_flexplot_breaks_with_labels_uses_custom_labels_in_plot():
    """breaks + labels: x levels in the plot are the custom labels."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.uniform(0, 10, size=60),
        "y": rng.normal(size=60),
    })
    p = flexplot(
        "y ~ x", data=df,
        breaks=[0, 3, 7, 10],
        labels=["low", "mid", "high"],
    )
    assert isinstance(p, ggplot)
    # The plotnine labels attribute should reflect the new x levels.
    # We can inspect the data passed to the geom layers.
    for layer in p.layers:
        if hasattr(layer, "data") and layer.data is not None and "x" in layer.data.columns:
            levels = set(layer.data["x"].astype(str).unique())
            # Each layer should only see the 3 binned labels.
            assert levels.issubset({"low", "mid", "high"}), f"unexpected levels: {levels}"
            break


def test_flexplot_no_bins_keeps_numeric_branch():
    """No bins: numeric x still goes through geom_smooth (LM/loess)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.uniform(0, 100, size=80),
        "y": rng.normal(size=80),
    })
    p = flexplot("y ~ x", data=df)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    # Should have a smooth layer (lm path), not jitter.
    assert any("geom_smooth" in t for t in layer_types)


def test_flexplot_bins_silently_ignored_on_discrete_x():
    """bins=N on already-discrete x: warning? or no-op? — no-op (caller
    pre-checks; the helper returns the data unchanged)."""
    df = pd.DataFrame({
        "x": ["a", "b", "c", "a", "b", "c"],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    # Should not raise; should render through the discrete branch.
    p = flexplot("y ~ x", data=df, bins=3)
    assert isinstance(p, ggplot)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_jitter" in layer_types


def test_flexplot_breaks_takes_precedence_over_bins():
    """When both bins and breaks are passed, breaks wins (with a warning)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.uniform(0, 100, size=80),
        "y": rng.normal(size=80),
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = flexplot("y ~ x", data=df, bins=10, breaks=[0, 50, 100])
    precedence_warnings = [
        w for w in caught if "breaks takes precedence" in str(w.message)
    ]
    assert len(precedence_warnings) == 1
    # 2 breaks = 2 bins. Verify the plot has 2 discrete levels.
    for layer in p.layers:
        if hasattr(layer, "data") and layer.data is not None and "x" in layer.data.columns:
            levels = layer.data["x"].astype(str).unique()
            assert len(set(levels)) == 2
            break