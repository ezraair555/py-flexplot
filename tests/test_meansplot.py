"""Tests for pyflexplot.meansplot — port of R's fifer::meansplot()."""
import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

from pyflexplot import meansplot


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "y": rng.normal(size=120),
        "g": rng.choice(["A", "B", "C"], size=120),
    })


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_meansplot_returns_ggplot(sample_df):
    """meansplot returns a plotnine ggplot."""
    p = meansplot("y ~ g", data=sample_df)
    assert isinstance(p, ggplot)


def test_meansplot_default_renders_point_and_errorbar(sample_df):
    """Default (error='se', connect=True) renders geom_point + geom_errorbar + geom_line."""
    p = meansplot("y ~ g", data=sample_df)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_point" in layer_types
    assert "geom_errorbar" in layer_types
    assert "geom_line" in layer_types


def test_meansplot_error_no_skips_errorbar(sample_df):
    """error='no' omits the geom_errorbar layer."""
    p = meansplot("y ~ g", data=sample_df, error="no")
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_errorbar" not in layer_types
    assert "geom_point" in layer_types


def test_meansplot_connect_false_skips_line(sample_df):
    """connect=False omits the geom_line connector."""
    p = meansplot("y ~ g", data=sample_df, connect=False)
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_line" not in layer_types
    assert "geom_point" in layer_types


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_meansplot_rejects_invalid_error(sample_df):
    """error='bogus' raises ValueError."""
    with pytest.raises(ValueError, match="error must be one of"):
        meansplot("y ~ g", data=sample_df, error="bogus")


def test_meansplot_rejects_formula_with_color():
    """y ~ g + z (color) raises ValueError."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=60),
        "g": rng.choice(["A", "B"], size=60),
        "z": rng.choice(["X", "Y"], size=60),
    })
    with pytest.raises(ValueError, match="color"):
        meansplot("y ~ g + z", data=df)


def test_meansplot_rejects_non_numeric_y(sample_df):
    """Non-numeric y raises ValueError."""
    df = sample_df.copy()
    df["y"] = df["y"].astype(str)
    with pytest.raises(ValueError, match="numeric y"):
        meansplot("y ~ g", data=df)


def test_meansplot_rejects_given_term():
    """A formula with a `given` term (after `|`) raises ValueError."""
    df = pd.DataFrame({
        "y": np.random.default_rng(0).normal(size=60),
        "g": np.random.default_rng(0).choice(["A", "B", "C"], size=60),
        "h": np.random.default_rng(0).choice(["X", "Y"], size=60),
    })
    with pytest.raises(ValueError, match="given"):
        meansplot("y ~ g | h", data=df)


# ---------------------------------------------------------------------------
# Numeric correctness
# ---------------------------------------------------------------------------


def test_meansplot_means_match_groupby():
    """Per-group mean values match a manual groupby-aggregation."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "g": ["A", "A", "A", "B", "B", "B"],
    })
    p = meansplot("y ~ g", data=df, error="no", connect=False)
    # plotnine inherits the summary dataframe from the parent ggplot.
    summary = p.data
    means_in_layer = sorted(summary["mean"].to_numpy().tolist())
    # Manual: A mean = 2.0, B mean = 5.0.
    assert means_in_layer == [2.0, 5.0]
    # Counts and SDs should also be present.
    counts = sorted(summary["count"].to_numpy().tolist())
    assert counts == [3, 3]


def test_meansplot_se_error_bars_have_correct_width():
    """SE error bar half-width matches std / sqrt(n) (pandas ddof=1 default)."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "g": ["A"] * 5 + ["B"] * 5,
    })
    p = meansplot("y ~ g", data=df, error="se")
    summary = p.data
    # pandas std() default ddof=1.
    # For A (y=[1..5]): var = 10/(5-1) = 2.5; std = sqrt(2.5); SE = sqrt(2.5/5) = sqrt(0.5).
    # For B (y=[6..10]): same.
    half_widths = (summary["__upper"] - summary["mean"]).abs().to_numpy()
    for hw in half_widths:
        assert abs(hw - np.sqrt(0.5)) < 1e-9


def test_meansplot_sd_error_bars_use_sample_std():
    """SD error bar half-width matches the sample std (ddof=1)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "g": rng.choice(["A", "B"], size=200),
    })
    p = meansplot("y ~ g", data=df, error="sd")
    summary = p.data
    for _, row in summary.iterrows():
        g_val = row["g"]
        subset = df[df["g"] == g_val]["y"]
        expected_sd = float(subset.std(ddof=1))
        half_width = float(abs(row["__upper"] - row["mean"]))
        assert abs(half_width - expected_sd) < 1e-9


def test_meansplot_with_numeric_x_discretizes():
    """Numeric x with few unique values is coerced to discrete levels."""
    df = pd.DataFrame({
        "y": np.random.default_rng(0).normal(size=60),
        "g": np.random.default_rng(0).choice([1, 2, 3], size=60),  # numeric, 3 levels
    })
    p = meansplot("y ~ g", data=df)
    assert isinstance(p, ggplot)
    # Should render normally; no error.
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert "geom_point" in layer_types