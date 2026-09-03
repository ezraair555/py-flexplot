"""Tests for v0.7.0 ``interaction_model=True`` on flexplot().

Closes the largest semantic gap in the v0.6.2 R-audit: when the formula
contains ``*`` or ``:`` syntax, the fit is additive by default (parallel
slopes per color group). v0.7.0 adds ``interaction_model=True`` which
fits the actual interaction term and overlays non-parallel per-group
regression lines.
"""
import warnings

import numpy as np
import pandas as pd
from plotnine import ggplot

from pyflexplot import flexplot


def _interaction_df(n=120, seed=0):
    """Generate data with a real x:z interaction."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    z = rng.choice(["A", "B"], size=n)
    # y = x + 2*x*z_is_B + noise  (so z='B' has a steeper slope than z='A')
    y = x + 2.0 * x * (z == "B").astype(float) + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


# ---------------------------------------------------------------------------
# Warning suppression
# ---------------------------------------------------------------------------


def test_interaction_syntax_emits_warning_by_default():
    """y ~ x*z without interaction_model=True: UserWarning emitted."""
    df = _interaction_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flexplot("y ~ x*z", data=df)
    interaction_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "interaction" in str(w.message).lower()
        and "additive" in str(w.message).lower()
    ]
    assert len(interaction_warnings) == 1


def test_interaction_model_true_suppresses_warning():
    """y ~ x*z with interaction_model=True: no additive-fit warning."""
    df = _interaction_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flexplot("y ~ x*z", data=df, interaction_model=True)
    interaction_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "additive" in str(w.message).lower()
    ]
    assert interaction_warnings == []


# ---------------------------------------------------------------------------
# Behavior: per-group non-parallel lines
# ---------------------------------------------------------------------------


def test_interaction_model_true_draws_per_group_lines():
    """interaction_model=True: one geom_line per color group (n_groups >= 2)."""
    df = _interaction_df()
    p = flexplot("y ~ x*z", data=df, interaction_model=True)
    line_layers = [
        layer for layer in p.layers
        if layer.geom.__class__.__name__ == "geom_line"
    ]
    # Two color groups ('A', 'B') => two fitted lines.
    assert len(line_layers) == 2


def test_interaction_model_true_lines_have_distinct_slopes():
    """The two per-group lines should NOT have the same slope (non-parallel).

    We verify by computing the slope of each line via numpy.polyfit on the
    line layer's data and asserting the slopes differ by at least 0.5.
    """
    df = _interaction_df()
    p = flexplot("y ~ x*z", data=df, interaction_model=True)
    line_layers = [
        layer for layer in p.layers
        if layer.geom.__class__.__name__ == "geom_line"
    ]
    slopes = []
    for layer in line_layers:
        layer_data = getattr(layer, "data", None) or getattr(layer, "_data", None)
        if layer_data is not None and "x" in layer_data.columns and "y" in layer_data.columns:
            slope, _ = np.polyfit(layer_data["x"].to_numpy(), layer_data["y"].to_numpy(), 1)
            slopes.append(slope)
    assert len(slopes) >= 2
    # The true model has slope 1 for z='A' and slope 3 for z='B'.
    # Allow a generous tolerance; the test is that the slopes differ.
    assert abs(slopes[0] - slopes[1]) > 0.5, f"Slopes too similar: {slopes}"


def test_interaction_model_default_still_additive():
    """Without interaction_model=True, the fit is additive (1 line total)."""
    df = _interaction_df()
    # Suppress the additive warning so we can count layers cleanly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
    p = flexplot("y ~ x*z", data=df)
    line_layers = [
        layer for layer in p.layers
        if layer.geom.__class__.__name__ == "geom_line"
    ]
    # Additive fit => no per-group geom_line (plotnine handles color
    # automatically via geom_smooth's color aes, not via separate layers).
    assert len(line_layers) == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_interaction_model_true_requires_interaction_term():
    """interaction_model=True with non-interaction formula: no error, behaves
    like the additive path (helper detects no interaction term and falls back).
    """
    df = _interaction_df()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
    # y ~ x + z has no `*` or `:`, so interaction_model has nothing to act on.
    p = flexplot("y ~ x + z", data=df, interaction_model=True)
    assert isinstance(p, ggplot)


def test_interaction_model_true_requires_color_group():
    """y ~ x:y (interaction but no separate color) still works via fallback."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.uniform(-3, 3, size=60),
        "z": rng.choice(["A", "B"], size=60),
        "y": rng.normal(size=60),
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
    # y ~ x:z is parsed as x being the first atom of the interaction
    # term; the parser sets color=None because there's no + sign.
    # interaction_model=True should not crash on this.
    p = flexplot("y ~ x:z", data=df, interaction_model=True)
    assert isinstance(p, ggplot)


def test_interaction_model_true_with_single_color_level():
    """If color has only one level, fall back to additive (no interaction to estimate)."""
    df = _interaction_df()
    df_single = df[df["z"] == "A"].reset_index(drop=True)  # only z='A'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
    p = flexplot("y ~ x*z", data=df_single, interaction_model=True)
    assert isinstance(p, ggplot)


# ---------------------------------------------------------------------------
# CI / ribbon
# ---------------------------------------------------------------------------


def test_interaction_model_with_uncertainty_draws_ribbons():
    """interaction_model=True + uncertainty='ci': ribbons per group."""
    df = _interaction_df()
    p = flexplot("y ~ x*z", data=df, interaction_model=True, uncertainty="ci")
    n_ribbons = sum(
        1 for layer in p.layers
        if layer.geom.__class__.__name__ == "geom_ribbon"
    )
    assert n_ribbons == 2


def test_interaction_model_with_uncertainty_none_skips_ribbons():
    """interaction_model=True + uncertainty=None: lines only, no ribbons."""
    df = _interaction_df()
    p = flexplot("y ~ x*z", data=df, interaction_model=True, uncertainty=None)
    n_lines = sum(
        1 for layer in p.layers
        if layer.geom.__class__.__name__ == "geom_line"
    )
    n_ribbons = sum(
        1 for layer in p.layers
        if layer.geom.__class__.__name__ == "geom_ribbon"
    )
    # 2 lines, 0 ribbons.
    assert n_lines == 2
    assert n_ribbons == 0


def test_interaction_model_with_bands_draws_nested_ribbons():
    """interaction_model=True + bands=[0.5, 0.95]: nested ribbons per group."""
    df = _interaction_df()
    p = flexplot(
        "y ~ x*z", data=df,
        interaction_model=True,
        bands=[0.5, 0.95],
    )
    n_ribbons = sum(
        1 for layer in p.layers
        if layer.geom.__class__.__name__ == "geom_ribbon"
    )
    # 2 groups × 2 bands = 4 ribbons.
    assert n_ribbons == 4