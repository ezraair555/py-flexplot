"""Tests for the uncertainty quantification module (A: power feature)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from pyflexplot.uncertainty import (
    compute_bootstrap_ci,
    compute_prediction_band,
    format_band_label,
    validate_uncertainty_params,
)


# --- validate_uncertainty_params ----------------------------------------------


def test_validate_accepts_none():
    validate_uncertainty_params(None, None, None, "auto")


def test_validate_accepts_ci():
    validate_uncertainty_params("ci", 0.95, None, "auto")
    validate_uncertainty_params("ci", 0.5, None, "loess")


def test_validate_accepts_prediction():
    validate_uncertainty_params("prediction", 0.95, None, "auto")
    validate_uncertainty_params("prediction", 0.99, None, "loess")


def test_validate_accepts_bootstrap_for_loess_and_auto():
    validate_uncertainty_params("bootstrap", 0.95, None, "loess")
    validate_uncertainty_params("bootstrap", 0.95, None, "auto")


def test_validate_rejects_invalid_uncertainty_string():
    with pytest.raises(ValueError, match="uncertainty must be one of"):
        validate_uncertainty_params("bogus", 0.95, None, "auto")


def test_validate_rejects_level_zero():
    with pytest.raises(ValueError, match="level must be a number"):
        validate_uncertainty_params("ci", 0.0, None, "auto")


def test_validate_rejects_level_one():
    with pytest.raises(ValueError, match="level must be a number"):
        validate_uncertainty_params("ci", 1.0, None, "auto")


def test_validate_rejects_level_above_one():
    with pytest.raises(ValueError, match="level must be a number"):
        validate_uncertainty_params("ci", 1.5, None, "auto")


def test_validate_rejects_bands_not_list():
    with pytest.raises(ValueError, match="bands must be a list"):
        validate_uncertainty_params("ci", None, "not-a-list", "auto")


def test_validate_accepts_bands_list():
    validate_uncertainty_params("ci", None, [0.5, 0.8, 0.95], "auto")


def test_validate_rejects_bands_with_invalid_member():
    with pytest.raises(ValueError, match="Each band level"):
        validate_uncertainty_params("ci", None, [0.5, 1.0], "auto")


def test_validate_rejects_bootstrap_for_lm():
    """Bootstrap CI only makes sense for non-parametric smoothers."""
    with pytest.raises(ValueError, match="only supported for method='loess'"):
        validate_uncertainty_params("bootstrap", 0.95, None, "lm")


def test_validate_accepts_both_level_and_bands():
    # Both is permitted — caller decides precedence.
    validate_uncertainty_params("ci", 0.95, [0.5, 0.8], "auto")


# --- format_band_label --------------------------------------------------------


def test_format_band_label_ci_default():
    assert format_band_label(0.95) == "95% CI"


def test_format_band_label_pi():
    assert format_band_label(0.80, kind="prediction") == "80% PI"


def test_format_band_label_bootstrap():
    assert format_band_label(0.95, kind="bootstrap") == "95% bootstrap CI"


def test_format_band_label_unknown_kind():
    assert format_band_label(0.50, kind="weird") == "50% weird"


def test_format_band_label_rounds():
    assert format_band_label(0.954) == "95% CI"


# --- compute_bootstrap_ci -----------------------------------------------------


def _linear_smooth(x_eval, y_sorted):
    """A simple smoother for unit tests: linear interpolation over sorted x."""
    x_sorted = np.sort(np.unique(x_eval)) if len(x_eval) else np.array([])
    if len(x_sorted) == 0:
        return np.array([])
    # Just return y_sorted aligned to x_eval order; the test checks shape
    # and percentile bounds, not smoothness quality.
    return np.interp(x_eval, np.linspace(0, 1, len(y_sorted)), y_sorted)


def test_bootstrap_ci_shape_and_finite():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(scale=0.5, size=n)
    x_eval = np.linspace(-3, 3, 50)
    x_out, lower, upper = compute_bootstrap_ci(
        x, y, _linear_smooth, n_resamples=50, level=0.95, x_eval=x_eval,
        random_state=0,
    )
    assert x_out.shape == lower.shape == upper.shape
    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))


def test_bootstrap_ci_lower_below_upper():
    rng = np.random.default_rng(1)
    n = 150
    x = rng.normal(size=n)
    y = x ** 2 + rng.normal(scale=0.3, size=n)
    x_eval = np.linspace(-2, 2, 30)
    _, lower, upper = compute_bootstrap_ci(
        x, y, _linear_smooth, n_resamples=30, level=0.90,
        x_eval=x_eval, random_state=1,
    )
    assert np.all(lower <= upper)


def test_bootstrap_ci_random_state_reproducible():
    rng = np.random.default_rng(2)
    x = rng.normal(size=80)
    y = x + rng.normal(size=80)
    x_eval = np.linspace(-2, 2, 10)
    _, l1, u1 = compute_bootstrap_ci(
        x, y, _linear_smooth, n_resamples=20, random_state=42, x_eval=x_eval,
    )
    _, l2, u2 = compute_bootstrap_ci(
        x, y, _linear_smooth, n_resamples=20, random_state=42, x_eval=x_eval,
    )
    np.testing.assert_array_equal(l1, l2)
    np.testing.assert_array_equal(u1, u2)


def test_bootstrap_ci_handles_failed_resample_gracefully():
    """A smooth_fn that raises on bootstrap samples should not crash."""

    def bad_smooth(x_eval, y_sorted):
        if len(y_sorted) != 80:  # Fail on bootstrap (size != full n)
            raise ValueError("simulated singular fit")
        return np.interp(x_eval, np.linspace(0, 1, len(y_sorted)), y_sorted)

    rng = np.random.default_rng(3)
    x = rng.normal(size=80)
    y = x + rng.normal(size=80)
    x_eval = np.linspace(-2, 2, 10)
    _, lower, upper = compute_bootstrap_ci(
        x, y, bad_smooth, n_resamples=20, random_state=0, x_eval=x_eval,
    )
    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))


def test_bootstrap_ci_rejects_mismatched_x_y():
    with pytest.raises(ValueError, match="equal length"):
        compute_bootstrap_ci(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
            _linear_smooth,
            n_resamples=5,
        )


# --- compute_prediction_band --------------------------------------------------


def test_prediction_band_symmetric_around_pred():
    rng = np.random.default_rng(4)
    n = 300
    y_true = rng.normal(size=n)
    y_pred = np.zeros(n)
    lower, upper = compute_prediction_band(y_true, y_pred, level=0.95)
    assert lower.shape == (n,)
    assert upper.shape == (n,)
    np.testing.assert_array_equal(lower, -upper)


def test_prediction_band_half_width_matches_z_times_sigma():
    rng = np.random.default_rng(5)
    n = 1000
    y_true = rng.normal(loc=0.0, scale=2.0, size=n)
    y_pred = np.zeros(n)
    level = 0.95
    lower, upper = compute_prediction_band(y_true, y_pred, level=level)
    z = scipy_stats.norm.ppf(1 - (1 - level) / 2)
    expected_hw = z * 2.0  # sigma == 2.0
    assert np.allclose(upper, expected_hw, atol=0.05)
    assert np.allclose(lower, -expected_hw, atol=0.05)


def test_prediction_band_higher_level_is_wider():
    rng = np.random.default_rng(6)
    n = 200
    y_true = rng.normal(size=n)
    y_pred = np.zeros(n)
    l_50, u_50 = compute_prediction_band(y_true, y_pred, level=0.50)
    l_99, u_99 = compute_prediction_band(y_true, y_pred, level=0.99)
    assert (u_99 - l_99).mean() > (u_50 - l_50).mean()


def test_prediction_band_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        compute_prediction_band(np.zeros(5), np.zeros(7))


# --- integration: flexplot() uncertainty threading ----------------------------


def test_flexplot_uncertainty_none_no_fit_layer():
    """Setting uncertainty=None should produce a plot with no geom_smooth."""
    from pyflexplot import flexplot

    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "y": rng.normal(size=50),
        "x": rng.normal(size=50),
    })
    p = flexplot("y ~ x", data=df, uncertainty=None)
    # Walk the plotnine's layers and confirm no geom_smooth present.
    layers = list(p.layers)
    from plotnine import geom_smooth
    assert not any(isinstance(layer.geom, geom_smooth) for layer in layers)


def test_flexplot_uncertainty_ci_default_renders_smooth():
    """Default uncertainty='ci' should keep the existing geom_smooth behavior."""
    from pyflexplot import flexplot

    rng = np.random.default_rng(8)
    df = pd.DataFrame({
        "y": rng.normal(size=50),
        "x": rng.normal(size=50),
    })
    p = flexplot("y ~ x", data=df)  # All defaults
    from plotnine import geom_smooth
    layers = list(p.layers)
    assert any(isinstance(layer.geom, geom_smooth) for layer in layers)


def test_flexplot_custom_level_propagates():
    """level= param should land in the geom_smooth stat's params."""
    from pyflexplot import flexplot
    from plotnine import geom_smooth

    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "y": rng.normal(size=50),
        "x": rng.normal(size=50),
    })
    p = flexplot("y ~ x", data=df, level=0.80)
    smooth_layers = [
        layer for layer in p.layers
        if isinstance(layer.geom, geom_smooth)
    ]
    assert len(smooth_layers) == 1
    # Plotnine stores `level` in the stat params, not geom params.
    assert smooth_layers[0].stat.params["level"] == 0.80


def test_flexplot_bands_creates_multiple_smooth_layers():
    """bands=[...] should produce one geom_smooth per level."""
    from pyflexplot import flexplot
    from plotnine import geom_smooth

    rng = np.random.default_rng(10)
    df = pd.DataFrame({
        "y": rng.normal(size=50),
        "x": rng.normal(size=50),
    })
    p = flexplot("y ~ x", data=df, bands=[0.5, 0.8, 0.95])
    smooth_layers = [
        layer for layer in p.layers
        if isinstance(layer.geom, geom_smooth)
    ]
    # 3 nested levels → 3 smooth layers.
    levels = sorted(
        layer.stat.params.get("level", 0.95) for layer in smooth_layers
    )
    assert levels == [0.5, 0.8, 0.95]


def test_flexplot_binomial_ci_renders_smooth():
    """Numeric binary y now correctly routes to the binomial GLM branch.

    Before v0.6.1, ``pd.api.types.is_numeric_dtype([0, 1])`` returned True,
    so the LM/loess branch was always taken for int/float [0, 1] y and the
    binomial branch was dead code. The fix adds a binary pre-check that
    detects [0, 1] values BEFORE the numeric-dtype dispatch.
    """
    from pyflexplot import flexplot
    from plotnine import geom_smooth

    df = pd.DataFrame({
        "y": [0, 1] * 25,
        "x": np.random.normal(size=50),
    })
    p = flexplot("y ~ x", data=df, uncertainty="ci")
    smooth_layers = [
        layer for layer in p.layers
        if isinstance(layer.geom, geom_smooth)
    ]
    assert len(smooth_layers) >= 1
    layer = smooth_layers[0]
    # Binomial GLM should now be drawn with method="glm" and
    # method_args={"family": "binomial"} in the stat params.
    assert layer.stat.params.get("method") == "glm"
    assert layer.stat.params.get("method_args") == {"family": "binomial"}
    assert layer.stat.params.get("level") == 0.95


def test_flexplot_invalid_uncertainty_raises():
    from pyflexplot import flexplot

    rng = np.random.default_rng(11)
    df = pd.DataFrame({
        "y": rng.normal(size=30),
        "x": rng.normal(size=30),
    })
    with pytest.raises(ValueError, match="uncertainty must be one of"):
        flexplot("y ~ x", data=df, uncertainty="bogus")


def test_flexplot_invalid_level_raises():
    from pyflexplot import flexplot

    rng = np.random.default_rng(12)
    df = pd.DataFrame({
        "y": rng.normal(size=30),
        "x": rng.normal(size=30),
    })
    with pytest.raises(ValueError, match="level must be a number"):
        flexplot("y ~ x", data=df, level=1.0)


def test_flexplot_bootstrap_with_lm_raises():
    from pyflexplot import flexplot

    rng = np.random.default_rng(13)
    df = pd.DataFrame({
        "y": rng.normal(size=30),
        "x": rng.normal(size=30),
    })
    with pytest.raises(ValueError, match="only supported for method='loess'"):
        flexplot("y ~ x", data=df, method="lm", uncertainty="bootstrap")