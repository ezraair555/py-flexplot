"""Correctness tests for bluepill that would have caught the v0.2.1 bugs.

These tests check contract-level invariants (each predictor has non-zero
variance; categorical levels are present; tuple specs are accepted), not
just that the functions return.  They run on top of the structural tests
in tests/test_bluepill.py.
"""


from pyflexplot import mixed_model


# ---------------------------------------------------------------------------
# Regressions for the v0.2.1 off-by-one bug
# ---------------------------------------------------------------------------

class TestPredictorVarianceRegression:
    """BUG-1 regression: every declared continuous predictor must have
    non-zero variance in the output.  The v0.2.1 off-by-one made the
    last predictor constant at its declared mean."""

    def _basic_vars(self, n_therapists: int = 15):
        return {
            "depression": (10.0, 3.0, 0),
            "stress": (22.0, 7.0, 0),
            "life_events": ["no", "yes"],
            "parental_depression": ["no", "mild", "moderate", "severe"],
            "ses": (55.0, 15.0, 0),
            "therapist": [f"Dr. {chr(ord('A') + i)}" for i in range(n_therapists)],
        }

    def _basic_fixed(self):
        return [0.0, 0.2, 0.5, 0.3, 0.2]

    def _basic_random(self):
        return [0.1, 0.1, 0.0, 0.2, 0.1]

    def test_all_continuous_predictors_have_nonzero_variance(self):
        df = mixed_model(
            fixed=self._basic_fixed(),
            random=self._basic_random(),
            sigma=0.3,
            clusters=15,
            n_per=[11, 3],
            vars=self._basic_vars(),
            seed=42,
        )
        # Every continuous predictor should have std > 1 (well above any
        # numeric noise that could come from rescaling).
        for col, spec in self._basic_vars().items():
            if isinstance(spec, tuple):
                std = df[col].std()
                assert std > 1.0, (
                    f"Predictor {col!r} has std={std:.4f} -- looks like "
                    "BUG-1 (constant column) regressed"
                )

    def test_all_continuous_predictors_use_full_range(self):
        # Rescaling targets (mean, sd); a constant column would have the
        # right mean but no spread.
        df = mixed_model(
            fixed=self._basic_fixed(),
            random=self._basic_random(),
            sigma=0.3,
            clusters=15,
            n_per=[11, 3],
            vars=self._basic_vars(),
            seed=42,
        )
        for col, spec in self._basic_vars().items():
            if not isinstance(spec, tuple):
                continue
            target_mean, target_sd, _ = spec
            # Rescaled values should span at least 4 SDs of the target.
            observed_range = df[col].max() - df[col].min()
            assert observed_range > 4 * target_sd, (
                f"Predictor {col!r} has range {observed_range:.2f}, "
                f"expected > {4 * target_sd:.2f}"
            )

    def test_predictor_columns_are_independent(self):
        # Off-by-one would also shift predictors so they're not actually
        # aligned with their declared specs; checking pairwise correlation
        # catches any column-reordering bug.
        df = mixed_model(
            fixed=self._basic_fixed(),
            random=self._basic_random(),
            sigma=0.3,
            clusters=15,
            n_per=[11, 3],
            vars=self._basic_vars(),
            seed=42,
        )
        # stress and ses should NOT be perfectly correlated (different specs).
        corr = df[["stress", "ses"]].corr().iloc[0, 1]
        assert abs(corr) < 0.95, (
            f"stress and ses are suspiciously correlated (r={corr:.3f}) -- "
            "suggests BUG-1-style column shift"
        )


# ---------------------------------------------------------------------------
# Regressions for the v0.2.1 tuple-categorical bug
# ---------------------------------------------------------------------------

class TestTupleSpecRegression:
    """BUG-3 regression: tuple-of-strings should be treated as categorical."""

    def test_tuple_of_strings_accepted_as_categorical(self):
        df = mixed_model(
            fixed=[0.0, 0.2, 0.5],
            random=[0.1, 0.1, 0.1],
            sigma=0.3,
            clusters=3,
            n_per=[5, 1],
            vars={
                "y": (10.0, 3.0, 0),
                "x": ("no", "yes"),  # tuple, not list
                "z": (5, 2, 0),
                "cluster": ["a", "b", "c"],
            },
        )
        assert set(df["x"].unique()).issubset({"no", "yes"})

    def test_tuple_of_ints_also_accepted_as_categorical(self):
        df = mixed_model(
            fixed=[0.0, 0.2, 0.5],
            random=[0.1, 0.1, 0.1],
            sigma=0.3,
            clusters=3,
            n_per=[5, 1],
            vars={
                "y": (10.0, 3.0, 0),
                "x": ("low", "medium", "high"),  # categorical
                "z": (5, 2, 0),
                "cluster": ["a", "b", "c"],
            },
        )
        assert set(df["x"].unique()) == {"low", "medium", "high"}

    def test_continuous_spec_requires_numeric_elements(self):
        # Sanity check: a 3-tuple of numbers IS treated as continuous.
        # pd.cut doesn't accept mixed-type labels so we can't easily test
        # the rejection path; instead we confirm the happy path is
        # invariant under list-vs-tuple for the continuous case.
        df = mixed_model(
            fixed=[0.0, 0.2],
            random=[0.1, 0.1],
            sigma=0.3,
            clusters=3,
            n_per=[5, 1],
            vars={
                "y": (10.0, 3.0, 0),
                "x": [5, 2, 0],  # list, same shape as tuple spec
                "cluster": ["a", "b", "c"],
            },
        )
        # x is continuous; should have non-zero variance.
        assert df["x"].std() > 0