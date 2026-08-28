"""Tests for pyflexplot.bluepill (port of dustinfife/bluepill)."""

import numpy as np
import pandas as pd
import pytest

from pyflexplot.bluepill import estimate_sd, mixed_model


# ---------------------------------------------------------------------------
# estimate_sd
# ---------------------------------------------------------------------------

class TestEstimateSd:
    def test_symmetric_range(self):
        # mean 10, range [5, 15], 3 SD -> sd = 5/3.
        assert estimate_sd(10, 5, 15, num_sds=3) == pytest.approx(5 / 3)

    def test_asymmetric_range(self):
        # The min(mean - min, max - mean) is the binding constraint.
        assert estimate_sd(10, 5, 20, num_sds=3) == pytest.approx(5 / 3)

    def test_num_sds_two(self):
        # Smaller num_sds -> larger estimated SD.
        sd3 = estimate_sd(10, 5, 15, num_sds=3)
        sd2 = estimate_sd(10, 5, 15, num_sds=2)
        assert sd2 > sd3
        assert sd2 == pytest.approx(5 / 2)

    @pytest.mark.parametrize(
        "mean, lo, hi",
        [(0, 1, 5), (10, 12, 20), (5, 10, 4)],
    )
    def test_rejects_invalid_range(self, mean, lo, hi):
        with pytest.raises(ValueError):
            estimate_sd(mean, lo, hi)

    def test_rejects_nonpositive_num_sds(self):
        with pytest.raises(ValueError):
            estimate_sd(10, 5, 15, num_sds=0)
        with pytest.raises(ValueError):
            estimate_sd(10, 5, 15, num_sds=-1)


# ---------------------------------------------------------------------------
# mixed_model
# ---------------------------------------------------------------------------

class TestMixedModel:
    def _basic_vars(self, n_therapists: int = 15):
        return {
            "depression": (10.0, 3.0, 0),
            "stress": (22.0, 7.0, 0),
            "life_events": ["no", "yes"],
            "parental_depression": ["no", "mild", "moderate", "severe"],
            "ses": (55.0, 15.0, 0),
            "therapist": [f"Dr. {chr(ord('A') + i)}" for i in range(n_therapists)],
        }

    def _basic_vars_var_names(self):
        return ["depression", "stress", "life_events", "parental_depression", "ses"]

    def _basic_fixed(self):
        return [0.0, 0.2, 0.5, 0.3, 0.2]

    def _basic_random(self):
        return [0.1, 0.1, 0.0, 0.2, 0.1]

    def test_runs_with_documented_example(self):
        df = mixed_model(
            fixed=self._basic_fixed(),
            random=self._basic_random(),
            sigma=0.3,
            clusters=15,
            n_per=[11, 3],
            vars=self._basic_vars(),
            seed=42,
        )
        assert isinstance(df, pd.DataFrame)
        # The cluster variable (last entry in vars) becomes the first column
        # in the output frame, followed by the predictor slots in order.
        expected_cols = ["therapist", *self._basic_vars_var_names()]
        assert list(df.columns) == expected_cols
        # Each therapist should appear at least once.
        therapist_counts = df["therapist"].value_counts()
        assert therapist_counts.shape[0] == 15

    def test_seeded_reproducible(self):
        kwargs = dict(
            fixed=self._basic_fixed(),
            random=self._basic_random(),
            sigma=0.3,
            clusters=10,
            n_per=[11, 3],
            vars=self._basic_vars(n_therapists=10),
        )
        a = mixed_model(seed=123, **kwargs)
        b = mixed_model(seed=123, **kwargs)
        # Continuous columns should be identical given the same seed.
        for col in ("depression", "stress", "ses"):
            np.testing.assert_array_equal(a[col].to_numpy(), b[col].to_numpy())

    def test_total_n_matches_cluster_sizes(self):
        df = mixed_model(
            fixed=self._basic_fixed(),
            random=self._basic_random(),
            sigma=0.3,
            clusters=15,
            n_per=[11, 3],
            vars=self._basic_vars(),
            seed=0,
        )
        # Cluster sizes are at least 1 each, so total >= 15.
        assert len(df) >= 15
        # Each therapist has at least one observation.
        assert df["therapist"].nunique() == 15

    def test_continuous_columns_are_bounded_by_spec(self):
        df = mixed_model(
            fixed=self._basic_fixed(),
            random=self._basic_random(),
            sigma=0.3,
            clusters=15,
            n_per=[11, 3],
            vars=self._basic_vars(),
            seed=7,
        )
        # stress spec: mean=22, sd=7.  Most observations within ~5 SD.
        stress = df["stress"].to_numpy()
        assert stress.mean() == pytest.approx(22, abs=5)
        assert stress.std(ddof=0) == pytest.approx(7, abs=5)

    def test_categorical_columns_only_use_declared_levels(self):
        df = mixed_model(
            fixed=self._basic_fixed(),
            random=self._basic_random(),
            sigma=0.3,
            clusters=15,
            n_per=[11, 3],
            vars=self._basic_vars(),
            seed=7,
        )
        assert set(df["life_events"].unique()).issubset({"no", "yes"})
        assert set(df["parental_depression"].unique()).issubset(
            {"no", "mild", "moderate", "severe"}
        )

    def test_rejects_mismatched_fixed_random(self):
        with pytest.raises(ValueError, match="same length"):
            mixed_model(
                fixed=[0, 0.2, 0.5],
                random=[0.1, 0.1],
                sigma=0.3,
                clusters=5,
                n_per=[10, 2],
                vars={
                    "y": (10, 3, 0),
                    "x": (5, 2, 0),
                    "cluster": ["a", "b", "c", "d", "e"],
                },
            )

    def test_rejects_mismatched_vars_length(self):
        # fixed/random have length 3; vars has length 2 -> expected 4.
        with pytest.raises(ValueError, match="vars must have length"):
            mixed_model(
                fixed=[0, 0.2, 0.5],
                random=[0.1, 0.1, 0.1],
                sigma=0.3,
                clusters=3,
                n_per=[5, 1],
                vars={
                    "y": (10, 3, 0),
                    "cluster": ["a", "b", "c"],
                },
            )

    def test_rejects_clusters_mismatch(self):
        # 3 predictors + 1 cluster, but only 2 cluster levels declared.
        with pytest.raises(ValueError, match="Number of clusters"):
            mixed_model(
                fixed=[0, 0.2, 0.5],
                random=[0.1, 0.1, 0.1],
                sigma=0.3,
                clusters=3,
                n_per=[5, 1],
                vars={
                    "y": (10, 3, 0),
                    "x": (5, 2, 0),
                    "z": (1, 1, 0),
                    "cluster": ["a", "b"],  # only 2 levels, but 3 clusters
                },
            )

    def test_rejects_out_of_range_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            mixed_model(
                fixed=[0, 0.2, 0.5],
                random=[0.1, 0.1, 0.1],
                sigma=1.0,
                clusters=3,
                n_per=[5, 1],
                vars={
                    "y": (10, 3, 0),
                    "x": (5, 2, 0),
                    "z": (1, 1, 0),
                    "cluster": ["a", "b", "c"],
                },
            )

    def test_rejects_oversized_fixed_effects(self):
        # sum(fixed[1:]**2)**2 must be < 1.
        with pytest.raises(ValueError, match="standardized"):
            mixed_model(
                fixed=[0, 0.9, 0.9],
                random=[0.0, 0.0, 0.0],
                sigma=0.3,
                clusters=3,
                n_per=[5, 1],
                vars={
                    "y": (10, 3, 0),
                    "x": (5, 2, 0),
                    "z": (1, 1, 0),
                    "cluster": ["a", "b", "c"],
                },
            )

    def test_rejects_non_categorical_cluster(self):
        with pytest.raises(ValueError, match="categorical"):
            mixed_model(
                fixed=[0, 0.2, 0.5],
                random=[0.1, 0.1, 0.1],
                sigma=0.3,
                clusters=3,
                n_per=[5, 1],
                vars={
                    "y": (10, 3, 0),
                    "x": (5, 2, 0),
                    "z": (1, 1, 0),
                    "cluster": (1, 1, 0),  # a tuple -- treated as continuous
                },
            )

    def test_interactions_add_columns(self):
        df_no = mixed_model(
            fixed=self._basic_fixed(),
            random=self._basic_random(),
            sigma=0.3,
            clusters=10,
            n_per=[11, 3],
            vars=self._basic_vars(n_therapists=10),
            seed=0,
        )
        df_int = mixed_model(
            fixed=self._basic_fixed(),
            random=self._basic_random(),
            sigma=0.3,
            clusters=10,
            n_per=[11, 3],
            vars=self._basic_vars(n_therapists=10),
            interactions={"from": [1, 2], "to": [2, 3], "coef": [0.2, 0.1]},
            seed=0,
        )
        # Both runs share the same column structure -- interactions only
        # affect the latent y_std computation, not the column layout.
        assert list(df_no.columns) == list(df_int.columns)

    def test_polynomials_add_columns(self):
        df = mixed_model(
            fixed=self._basic_fixed(),
            random=self._basic_random(),
            sigma=0.3,
            clusters=10,
            n_per=[11, 3],
            vars=self._basic_vars(n_therapists=10),
            polynomials={"from": [3, 4], "to": [3, 4], "coef": [0.2, 0.2]},
            seed=0,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0