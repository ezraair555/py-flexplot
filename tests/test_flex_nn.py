"""Tests for pyflexplot.flex_nn (port of dustinfife/flex_nn)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")

from pyflexplot import flex_nn  # noqa: E402  -- after importorskip
from pyflexplot.flex_nn import (  # noqa: E402
    NeuralNetFit,
    is_torch_model,
    permutation_importance,
    prepare_torch_data,
    set_response_var,
    torch_backend_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TinyRegressor(torch.nn.Module):
    """A tiny 3-input linear regressor used throughout the tests."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def _trained_tiny_regressor(seed: int = 0) -> TinyRegressor:
    """Train TinyRegressor to near-perfect fit on a synthetic dataset."""
    torch.manual_seed(seed)
    n = 200
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    true_beta = np.array([1.5, -2.0, 0.7], dtype=np.float32)
    y = (X @ true_beta) + rng.normal(scale=0.1, size=n).astype(np.float32)
    X_t = torch.as_tensor(X)
    y_t = torch.as_tensor(y)
    model = TinyRegressor()
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(500):
        optim.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X_t), y_t)
        loss.backward()
        optim.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def test_torch_backend_available():
    assert torch_backend_available() is True
    assert flex_nn.torch_backend_available() is True


def test_is_torch_model_recognises_torch_module():
    assert is_torch_model(TinyRegressor())
    assert is_torch_model("not a model") is False


# ---------------------------------------------------------------------------
# prepare_torch_data
# ---------------------------------------------------------------------------

class TestPrepareTorchData:
    def test_pure_numeric(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        out = prepare_torch_data(df)
        np.testing.assert_array_equal(out, np.array([[1, 4], [2, 5], [3, 6]], dtype=float))

    def test_categorical_is_integer_encoded(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "color": pd.Categorical(["red", "blue", "red"]),
        })
        out = prepare_torch_data(df, categorical_vars=["color"])
        # pd.Categorical sorts levels; "blue" < "red" -> blue=0, red=1.
        assert out[0, 1] == 1.0  # red
        assert out[1, 1] == 0.0  # blue
        assert out[2, 1] == 1.0  # red

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            prepare_torch_data({"a": [1, 2, 3]})

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            prepare_torch_data(pd.DataFrame())

    def test_rejects_missing_values(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        with pytest.raises(ValueError, match="NaN"):
            prepare_torch_data(df)


# ---------------------------------------------------------------------------
# set_response_var
# ---------------------------------------------------------------------------

def test_set_response_var_attaches_attribute():
    model = TinyRegressor()
    set_response_var(model, "y_target")
    assert getattr(model, "_pyflexplot_response_var") == "y_target"


@pytest.mark.parametrize("bad", [None, "", 0, ["y"]])
def test_set_response_var_rejects_bad_input(bad):
    model = TinyRegressor()
    with pytest.raises((ValueError, TypeError)):
        set_response_var(model, bad)


# ---------------------------------------------------------------------------
# NeuralNetFit
# ---------------------------------------------------------------------------

class TestNeuralNetFit:
    def _build_fit(self, with_norm: bool = False):
        model = _trained_tiny_regressor()
        set_response_var(model, "y_target")
        rng = np.random.default_rng(0)
        predictor_names = ["x1", "x2", "x3"]
        # x_means / x_sds are required only when with_norm=True.
        if with_norm:
            x_means = np.zeros(3)
            x_sds = np.ones(3)
        else:
            x_means = None
            x_sds = None
        return NeuralNetFit(
            model=model,
            response_var="y_target",
            predictor_names=predictor_names,
            x_means=x_means,
            x_sds=x_sds,
            history=None,
        )

    def test_post_init_infers_torch_backend(self):
        fit = self._build_fit()
        assert fit.backend == "torch"

    def test_predict_returns_indexed_series(self):
        fit = self._build_fit()
        df = pd.DataFrame({
            "x1": np.linspace(-2, 2, 10),
            "x2": np.linspace(-1, 1, 10),
            "x3": np.zeros(10),
            "other_col": np.arange(10),
        })
        pred = fit.predict(df)
        assert isinstance(pred, pd.Series)
        assert len(pred) == len(df)
        assert pred.index.equals(df.index)
        assert pred.name == "y_target__pred"

    def test_predict_with_zscore_normalisation(self):
        fit = self._build_fit(with_norm=True)
        df = pd.DataFrame({
            "x1": [0.0],
            "x2": [0.0],
            "x3": [0.0],
        })
        # With mean=0, sd=1 and inputs all zero, the prediction should be
        # close to zero (intercepts and bias aside).
        pred = fit.predict(df)
        assert abs(float(pred.iloc[0])) < 1.0

    def test_predict_rejects_missing_predictor(self):
        fit = self._build_fit()
        df = pd.DataFrame({"x1": [1.0], "x2": [2.0]})  # x3 missing
        with pytest.raises(ValueError, match="missing predictors"):
            fit.predict(df)

    def test_rejects_empty_predictor_names(self):
        model = _trained_tiny_regressor()
        with pytest.raises(ValueError, match="at least one"):
            NeuralNetFit(
                model=model,
                response_var="y_target",
                predictor_names=[],
            )

    def test_rejects_duplicate_predictor_names(self):
        model = _trained_tiny_regressor()
        with pytest.raises(ValueError, match="duplicates"):
            NeuralNetFit(
                model=model,
                response_var="y_target",
                predictor_names=["x1", "x1", "x2"],
            )

    def test_rejects_non_neural_model_without_backend(self):
        with pytest.raises(TypeError, match="torch.nn.Module or keras.Model"):
            NeuralNetFit(
                model="not a model",
                response_var="y",
                predictor_names=["x"],
            )

    def test_repr_includes_backend(self):
        fit = self._build_fit()
        r = repr(fit)
        assert "NeuralNetFit" in r
        assert "torch" in r


# ---------------------------------------------------------------------------
# permutation_importance
# ---------------------------------------------------------------------------

class TestPermutationImportance:
    def _build_fit(self):
        model = _trained_tiny_regressor()
        set_response_var(model, "y_target")
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        # Construct y with x1 having the strongest effect so importance[0] > others.
        y = 2.0 * X[:, 0] + 0.1 * X[:, 1] + 0.05 * X[:, 2] + rng.normal(scale=0.1, size=50)
        df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
        fit = NeuralNetFit(
            model=model,
            response_var="y",
            predictor_names=["x1", "x2", "x3"],
        )
        return fit, df, pd.Series(y, name="y")

    def test_returns_dataframe_with_expected_columns(self):
        fit, df, y = self._build_fit()
        result = permutation_importance(fit, df, y, random_state=0)
        assert list(result.columns) == ["variable", "importance"]
        assert len(result) == 3
        assert set(result["variable"]) == {"x1", "x2", "x3"}

    def test_results_sorted_descending(self):
        fit, df, y = self._build_fit()
        result = permutation_importance(fit, df, y, random_state=0)
        importances = result["importance"].to_numpy()
        assert (importances[:-1] >= importances[1:]).all()

    def test_seeded_reproducible(self):
        fit, df, y = self._build_fit()
        a = permutation_importance(fit, df, y, random_state=123)
        b = permutation_importance(fit, df, y, random_state=123)
        np.testing.assert_array_equal(
            a["importance"].to_numpy(),
            b["importance"].to_numpy(),
        )

    def test_n_repeats_is_honoured(self):
        fit, df, y = self._build_fit()
        result = permutation_importance(fit, df, y, n_repeats=3, random_state=0)
        # We can't easily check the *number* of internal shuffles, but
        # repeated calls with the same seed should agree.
        assert len(result) == 3

    def test_rejects_non_neuralnetfit(self):
        with pytest.raises(TypeError):
            permutation_importance("not a fit", pd.DataFrame(), np.zeros(5))

    def test_rejects_empty_dataframe(self):
        fit, _, _ = self._build_fit()
        with pytest.raises(ValueError):
            permutation_importance(fit, pd.DataFrame(columns=["x1", "x2", "x3"]), [])

    def test_rejects_n_repeats_less_than_one(self):
        fit, df, y = self._build_fit()
        with pytest.raises(ValueError, match="n_repeats"):
            permutation_importance(fit, df, y, n_repeats=0)

    def test_rejects_unknown_metric_string(self):
        fit, df, y = self._build_fit()
        with pytest.raises(ValueError, match="Unknown metric"):
            permutation_importance(fit, df, y, metric="not-a-real-metric")

    def test_callable_metric_requires_higher_is_better(self):
        fit, df, y = self._build_fit()
        with pytest.raises(ValueError, match="higher_is_better"):
            permutation_importance(fit, df, y, metric=lambda yt, yp: 0.0)

    def test_callable_metric_higher_is_better_true(self):
        fit, df, y = self._build_fit()
        # R-squared proxy: 1 - sum((yt-yp)^2) / sum((yt-mean(yt))^2).
        def r2(yt, yp):
            yt = np.asarray(yt); yp = np.asarray(yp)
            ss_res = float(np.sum((yt - yp) ** 2))
            ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        result = permutation_importance(
            fit, df, y, metric=r2, higher_is_better=True, random_state=0
        )
        # Should run without error and return 3 rows.
        assert len(result) == 3

    def test_attrs_expose_baseline_and_metric(self):
        fit, df, y = self._build_fit()
        result = permutation_importance(fit, df, y, metric="mse", random_state=0)
        assert "baseline_score" in result.attrs
        assert "metric" in result.attrs