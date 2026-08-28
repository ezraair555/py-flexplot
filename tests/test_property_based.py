"""Hypothesis property-based tests.

These tests use Hypothesis to generate random inputs and verify invariants
that hand-written tests miss.  Two surfaces are covered:

* parse_flexplot_formula() round-trips valid formulas and rejects
  malformed ones consistently.
* mixed_model() rescaling matches the declared (mean, sd) spec for
  continuous predictors, and categorical predictors only take values
  from the declared level set.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from pyflexplot.bluepill import estimate_sd, mixed_model
from pyflexplot.core import parse_flexplot_formula


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Variable names that are plausible identifiers (start with letter, alnum/underscore)
variable_name = st.from_regex(r"[a-z][a-z0-9_]{0,7}", fullmatch=True).filter(
    lambda s: s not in {"Intercept", "const"}  # reserved names
)

# A small palette of variable names so duplicates are unlikely when generating lists.
small_var_pool = st.sampled_from(["y", "x", "x1", "x2", "x3", "a", "b", "c", "z", "w"])

# Generate a list of unique variable names (no duplicates within a formula).
@st.composite
def unique_var_list(draw, min_size=1, max_size=4, allow_y=True):
    names = draw(st.lists(small_var_pool, min_size=min_size, max_size=max_size, unique=True))
    if not allow_y and "y" in names:
        names = [n for n in names if n != "y"] or ["alt_y"]
    return names


# A valid flexplot formula: outcome ~ predictors [| givens]
@st.composite
def valid_formula(draw):
    y = "y"
    predictors = draw(unique_var_list(min_size=1, max_size=2, allow_y=False))
    has_given = draw(st.booleans())
    if has_given:
        givens = draw(unique_var_list(min_size=1, max_size=2, allow_y=False))
        return f"{y} ~ {' + '.join(predictors)} | {' + '.join(givens)}"
    return f"{y} ~ {' + '.join(predictors)}"


# An intercept-only formula: y ~ 1
intercept_formula = st.just("y ~ 1")


# ---------------------------------------------------------------------------
# parse_flexplot_formula round-trip and invariants
# ---------------------------------------------------------------------------

class TestParseFormulaInvariants:
    @given(valid_formula())
    @settings(max_examples=50, deadline=None)
    def test_outcome_is_y(self, formula: str) -> None:
        out = parse_flexplot_formula(formula)
        assert out["y"] == "y"

    @given(valid_formula())
    @settings(max_examples=50, deadline=None)
    def test_x_is_first_predictor(self, formula: str) -> None:
        out = parse_flexplot_formula(formula)
        x_name = out["x"]
        # x_name should appear as a token in the main-part after '~'.
        main_part = formula.split("|", 1)[0]
        x_part = main_part.split("~", 1)[1]
        tokens = [t.strip() for t in x_part.split("+")]
        assert x_name == tokens[0]

    @given(valid_formula())
    @settings(max_examples=50, deadline=None)
    def test_color_is_second_predictor_or_none(self, formula: str) -> None:
        out = parse_flexplot_formula(formula)
        if out["color"] is not None:
            main_part = formula.split("|", 1)[0]
            x_part = main_part.split("~", 1)[1]
            tokens = [t.strip() for t in x_part.split("+")]
            assert len(tokens) >= 2
            assert out["color"] == tokens[1]

    @given(valid_formula())
    @settings(max_examples=50, deadline=None)
    def test_given_count_matches_pipe(self, formula: str) -> None:
        out = parse_flexplot_formula(formula)
        if "|" in formula:
            given_part = formula.split("|", 1)[1].strip()
            expected = [g.strip() for g in given_part.split("+")]
            assert out["given"] == expected
        else:
            assert out["given"] == []

    @given(valid_formula())
    @settings(max_examples=50, deadline=None)
    def test_all_x_includes_x_and_color(self, formula: str) -> None:
        out = parse_flexplot_formula(formula)
        all_x = out["all_x"]
        # all_x should contain x_name and color (if set) in order.
        if out["color"] is None:
            assert all_x == [out["x"]]
        else:
            assert all_x == [out["x"], out["color"]]

    @given(intercept_formula)
    @settings(max_examples=10, deadline=None)
    def test_intercept_only(self, formula: str) -> None:
        out = parse_flexplot_formula(formula)
        assert out["intercept_only"] is True
        assert out["x"] is None
        assert out["color"] is None
        assert out["given"] == []

    @given(st.text(min_size=1, max_size=40).filter(lambda s: "~" not in s))
    @settings(max_examples=30, deadline=None)
    def test_no_tilde_raises(self, formula: str) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            parse_flexplot_formula(formula)

    @given(
        st.sampled_from([
            "y ~ x | a | b",
            "y ~ x || c",
            "y ~ x | a | b | c",
            "y ~ x | a + b | c",
            "y ~ x | a + b + c | d",
        ])
    )
    @settings(max_examples=20, deadline=None)
    def test_multiple_pipes_raises(self, formula: str) -> None:
        with pytest.raises(ValueError, match="at most one"):
            parse_flexplot_formula(formula)

    @given(st.sampled_from(["y ~ ", "~ x", "y  | a"]))
    @settings(max_examples=10, deadline=None)
    def test_malformed_formulas_raise(self, formula: str) -> None:
        with pytest.raises(ValueError):
            parse_flexplot_formula(formula)

    @given(valid_formula())
    @settings(max_examples=30, deadline=None)
    def test_parser_is_deterministic(self, formula: str) -> None:
        # Parsing the same formula twice returns equivalent dicts.
        a = parse_flexplot_formula(formula)
        b = parse_flexplot_formula(formula)
        assert a == b


# ---------------------------------------------------------------------------
# mixed_model rescaling invariants
# ---------------------------------------------------------------------------

class TestMixedModelRescalingInvariants:
    @given(
        target_mean=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        target_sd=st.floats(min_value=0.5, max_value=5, allow_nan=False, allow_infinity=False),
        digits=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_continuous_predictor_rescales_to_target(
        self, target_mean, target_sd, digits,
    ) -> None:
        spec = (target_mean, target_sd, digits)
        n_clusters = 5
        df = mixed_model(
            fixed=[0.0, 0.5],
            random=[0.1, 0.1],
            sigma=0.3,
            clusters=n_clusters,
            n_per=[10, 2],
            vars={
                "y": spec,
                "x": (5.0, 2.0, 1),
                "cluster": [f"c{i}" for i in range(n_clusters)],
            },
            seed=42,
        )
        observed_mean = float(df["y"].mean())
        observed_sd = float(df["y"].std(ddof=0))
        n = len(df)
        # The rescaling is approximate: it's a linear transform of a
        # standardized random vector, so mean / sd have sampling noise
        # proportional to 1/sqrt(n).  Allow generous tolerances.
        tol_mean = max(4 * target_sd / np.sqrt(n), 0.1)
        tol_sd = max(0.5 * target_sd, 0.1)
        assert abs(observed_mean - target_mean) < tol_mean, (
            f"mean: observed {observed_mean:.3f} vs target {target_mean:.3f}"
        )
        assert abs(observed_sd - target_sd) < tol_sd, (
            f"sd: observed {observed_sd:.3f} vs target {target_sd:.3f}"
        )

    @given(
        n_clusters=st.integers(min_value=2, max_value=8),
        levels_per_cat=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=15, deadline=None)
    def test_categorical_predictor_uses_only_declared_levels(
        self, n_clusters, levels_per_cat,
    ) -> None:
        levels = [f"L{i}" for i in range(levels_per_cat)]
        df = mixed_model(
            fixed=[0.0, 0.2, 0.5],
            random=[0.1, 0.1, 0.1],
            sigma=0.3,
            clusters=n_clusters,
            n_per=[10, 2],
            vars={
                "y": (10.0, 3.0, 0),
                "x": (5.0, 2.0, 0),
                "cat": levels,
                "cluster": [f"c{i}" for i in range(n_clusters)],
            },
            seed=0,
        )
        observed = set(df["cat"].astype(str).unique())
        assert observed.issubset(set(levels)), (
            f"observed values {observed} not in declared levels {levels}"
        )

    @given(
        clusters=st.integers(min_value=2, max_value=10),
        n_per_mean=st.floats(min_value=5, max_value=20, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=15, deadline=None)
    def test_total_n_at_least_clusters(self, clusters, n_per_mean) -> None:
        # Cluster sizes are at least 1 each, so total_n >= clusters.
        df = mixed_model(
            fixed=[0.0, 0.5],
            random=[0.1, 0.1],
            sigma=0.3,
            clusters=clusters,
            n_per=[n_per_mean, 1.0],
            vars={
                "y": (10.0, 3.0, 0),
                "x": (5.0, 2.0, 0),
                "cluster": [f"c{i}" for i in range(clusters)],
            },
            seed=0,
        )
        assert len(df) >= clusters


# ---------------------------------------------------------------------------
# estimate_sd invariants
# ---------------------------------------------------------------------------

class TestEstimateSdInvariants:
    @given(
        mean=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        half_range=st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
        num_sds=st.floats(min_value=0.5, max_value=6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_estimate_sd_returns_positive_value(self, mean, half_range, num_sds):
        # Construct a valid symmetric range around the mean.
        sd = estimate_sd(mean, mean - half_range, mean + half_range, num_sds=num_sds)
        assert sd > 0
        # Larger num_sds -> smaller SD (covers more of the range per unit SD).
        sd_larger = estimate_sd(mean, mean - half_range, mean + half_range, num_sds=num_sds * 2)
        assert sd_larger <= sd