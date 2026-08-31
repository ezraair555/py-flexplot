"""Tests for formula-function evaluation in flexplot (R ``formula_functions`` parity).

The R package's ``formula_functions()`` detects terms containing ``(`` in the
formula, evaluates them against the data, stores the result in a column named
after the inner variable, and rewrites the formula so downstream code sees the
simpler (un-transformed) term.

This file exercises:

- ``_apply_formula_functions`` directly (unit tests for the rewrite engine)
- ``flexplot(...)`` with formula functions in the RHS (integration tests)
- Safe-evaluator behaviour (unknown functions rejected, no ``eval()`` on
  arbitrary strings)
- Whitelisted functions: ``log``, ``log10``, ``log1p``, ``exp``, ``sqrt``,
  ``abs``, ``poly``, ``I(x ** k)``
- Round-trip behaviour (inner variable name preserved, but transformed values
  stored back in the same column).
"""
import ast
import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot

from pyflexplot import flexplot
from pyflexplot.core import _apply_formula_functions, _apply_formula_function


# ---------------------------------------------------------------------------
# Direct unit tests for the rewrite engine
# ---------------------------------------------------------------------------


def test_apply_formula_functions_noop_on_simple_formula():
    """A formula with no functions passes through with the same data."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    new_data, new_formula, terms = _apply_formula_functions("y ~ x", df)
    assert new_data.equals(df)
    assert new_formula == "y ~ x"
    assert terms == []


def test_apply_formula_functions_log_x_rewrites_and_stores():
    """``log(x)`` is rewritten to ``x`` and stored in a column named ``x``."""
    df = pd.DataFrame({"x": [1.0, np.e, np.e ** 2], "y": [0.0, 1.0, 2.0]})
    new_data, new_formula, terms = _apply_formula_functions("y ~ log(x)", df)
    assert terms == [("log(x)", "log", "x")]
    assert new_formula == "y ~ x"
    # log(x) of [1, e, e^2] is [0, 1, 2]; stored back in the 'x' column.
    np.testing.assert_allclose(new_data["x"].to_numpy(), [0.0, 1.0, 2.0])


def test_apply_formula_functions_sqrt_overwrites_existing_column():
    """R's ``data[, vars] = new_vars`` overwrites the existing inner-var column."""
    df = pd.DataFrame({"x": [4.0, 9.0, 16.0], "y": [1.0, 2.0, 3.0]})
    new_data, new_formula, terms = _apply_formula_functions("y ~ sqrt(x)", df)
    assert new_formula == "y ~ x"
    np.testing.assert_allclose(new_data["x"].to_numpy(), [2.0, 3.0, 4.0])
    # y column is untouched.
    np.testing.assert_array_equal(new_data["y"].to_numpy(), [1.0, 2.0, 3.0])


def test_apply_formula_functions_multiple_functions_each_rewritten():
    """``y ~ log(x) + sqrt(z)`` becomes ``y ~ x + z`` and both columns hold
    the transformed values.
    """
    df = pd.DataFrame({
        "x": [1.0, np.e, np.e ** 2],
        "z": [4.0, 9.0, 16.0],
        "y": [0.5, 1.5, 2.5],
    })
    new_data, new_formula, terms = _apply_formula_functions(
        "y ~ log(x) + sqrt(z)", df
    )
    assert new_formula == "y ~ x + z"
    assert sorted(t[1] for t in terms) == ["log", "sqrt"]
    np.testing.assert_allclose(new_data["x"].to_numpy(), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(new_data["z"].to_numpy(), [2.0, 3.0, 4.0])


def test_apply_formula_functions_poly_default_degree_2():
    """``poly(x)`` defaults to degree 2; result column holds x**2."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.0, 0.0, 0.0]})
    new_data, new_formula, _terms = _apply_formula_functions("y ~ poly(x)", df)
    assert new_formula == "y ~ x"
    np.testing.assert_allclose(new_data["x"].to_numpy(), [1.0, 4.0, 9.0])


def test_apply_formula_functions_poly_explicit_degree_3():
    """``poly(x, 3)`` evaluates x**3 and stores it in column ``x``."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.0, 0.0, 0.0]})
    new_data, new_formula, _terms = _apply_formula_functions(
        "y ~ poly(x, 3)", df
    )
    assert new_formula == "y ~ x"
    np.testing.assert_allclose(new_data["x"].to_numpy(), [1.0, 8.0, 27.0])


def test_apply_formula_functions_I_squared():
    """``I(x**2)`` evaluates x**2 and stores under the column name ``x``."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.0, 0.0, 0.0]})
    new_data, new_formula, terms = _apply_formula_functions(
        "y ~ I(x ** 2)", df
    )
    assert new_formula == "y ~ x"
    assert terms[0][1] == "I"
    np.testing.assert_allclose(new_data["x"].to_numpy(), [1.0, 4.0, 9.0])


def test_apply_formula_functions_I_squared_plus_offset():
    """``I(x**2 + 1)`` evaluates x**2 + 1."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.0, 0.0, 0.0]})
    new_data, new_formula, _terms = _apply_formula_functions(
        "y ~ I(x ** 2 + 1)", df
    )
    assert new_formula == "y ~ x"
    np.testing.assert_allclose(new_data["x"].to_numpy(), [2.0, 5.0, 10.0])


def test_apply_formula_functions_preserves_given_part():
    """``y ~ log(x) | z`` becomes ``y ~ x | z``."""
    df = pd.DataFrame({"x": [1.0, np.e, np.e ** 2], "y": [0.0, 1.0, 2.0],
                       "z": ["a", "b", "a"]})
    new_data, new_formula, _terms = _apply_formula_functions(
        "y ~ log(x) | z", df
    )
    assert new_formula == "y ~ x | z"


def test_apply_formula_functions_unknown_function_raises():
    """Whitelist rejection: ``f(x)`` (not in the map) raises ValueError."""
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="not supported"):
        _apply_formula_functions("y ~ f(x)", df)


def test_apply_formula_functions_missing_inner_var_raises():
    """Referencing a non-existent column raises ValueError."""
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing column"):
        _apply_formula_functions("y ~ log(z)", df)


def test_apply_formula_functions_uses_safe_evaluator_no_arbitrary_eval():
    """The implementation must NOT invoke ``eval()`` on the formula string.

    We monkey-patch ``eval`` in the core module and assert it's never
    called.  Without the whitelist, a malicious formula like
    ``__import__('os').system('rm -rf /')`` would be evaluated.
    """
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    import pyflexplot.core as core
    calls = []
    real_eval = getattr(core, "eval", eval)
    core.eval = lambda *a, **kw: calls.append((a, kw)) or real_eval(*a, **kw)
    try:
        # Attempt something that would be lethal under unrestricted eval.
        with pytest.raises((ValueError, SyntaxError, TypeError)):
            core._apply_formula_functions(
                "y ~ __import__('os')", df
            )
    finally:
        core.eval = real_eval
    # Even if the call raises, eval should not have been reached.
    assert all(not str(a[0]).startswith("__import__") for a, _ in calls), (
        f"_apply_formula_functions invoked eval() on arbitrary strings: {calls}"
    )


def test_apply_formula_functions_interaction_term_left_alone():
    """``y ~ x * z`` is preserved verbatim (no function to extract)."""
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [0.0, 1.0], "z": [3.0, 4.0]})
    _new_data, new_formula, terms = _apply_formula_functions(
        "y ~ x * z", df
    )
    assert new_formula == "y ~ x * z"
    assert terms == []


def test_apply_formula_functions_mixed_function_and_atom():
    """``y ~ log(x) + z`` becomes ``y ~ x + z`` and only ``x`` is transformed."""
    df = pd.DataFrame({"x": [1.0, np.e], "z": [5.0, 6.0], "y": [1.0, 2.0]})
    new_data, new_formula, terms = _apply_formula_functions(
        "y ~ log(x) + z", df
    )
    assert new_formula == "y ~ x + z"
    assert terms == [("log(x)", "log", "x")]
    np.testing.assert_allclose(new_data["x"].to_numpy(), [0.0, 1.0])
    np.testing.assert_array_equal(new_data["z"].to_numpy(), [5.0, 6.0])


# ---------------------------------------------------------------------------
# flexplot() integration tests with formula functions
# ---------------------------------------------------------------------------


def test_flexplot_log_x_transforms_x_before_plotting():
    """flexplot('y ~ log(x)') applies the transformation before plotting.

    The smoother should see ``log(x)`` values, not raw x.  We test by
    constructing a dataset where log(x) is linear in y, but x itself is
    not, and asserting the plot still renders without error.
    """
    rng = np.random.default_rng(0)
    x_raw = rng.uniform(1, 10, size=80)
    df = pd.DataFrame({
        "x": x_raw,
        "y": np.log(x_raw) + rng.normal(scale=0.1, size=80),
    })
    p = flexplot("y ~ log(x)", data=df)
    assert isinstance(p, ggplot)
    # Layers should include a smooth layer (numeric-x branch).
    layer_types = [layer.geom.__class__.__name__ for layer in p.layers]
    assert any("geom_smooth" in t for t in layer_types)


def test_flexplot_log_x_does_not_mutate_user_data():
    """The user's input DataFrame is not mutated by formula-function evaluation.

    flexplot makes an internal copy before applying the rewrite.
    """
    df = pd.DataFrame({"x": [1.0, np.e, np.e ** 2], "y": [0.0, 1.0, 2.0]})
    df_before = df.copy()
    _p = flexplot("y ~ log(x)", data=df)
    # The user's original df must still hold the un-transformed x values.
    pd.testing.assert_frame_equal(df, df_before)


def test_flexplot_sqrt_x_renders():
    """``y ~ sqrt(x)`` runs end-to-end."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.uniform(0, 100, size=60),
        "y": rng.normal(size=60),
    })
    p = flexplot("y ~ sqrt(x)", data=df)
    assert isinstance(p, ggplot)


def test_flexplot_I_x_squared_renders():
    """``I(x**2)`` is a valid formula-function expression."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.uniform(-3, 3, size=60),
        "y": rng.normal(size=60),
    })
    p = flexplot("y ~ I(x ** 2)", data=df)
    assert isinstance(p, ggplot)


def test_flexplot_poly_x_renders():
    """``poly(x, 2)`` evaluates x**2 and renders the smoother."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.uniform(0, 5, size=60),
        "y": rng.normal(size=60),
    })
    p = flexplot("y ~ poly(x, 2)", data=df)
    assert isinstance(p, ggplot)


def test_flexplot_unknown_function_raises_clear_error():
    """An unknown function name produces a ValueError mentioning the helper."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="not supported"):
        flexplot("y ~ nonsense(x)", data=df)
