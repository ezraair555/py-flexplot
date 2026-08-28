"""Tests for pyflexplot.flex_nn against a Keras 3 backend.

These tests require a working Keras 3 install (typically via ``pip install
keras`` or ``pip install tensorflow``).  When keras is not importable they
skip cleanly, mirroring the pattern used by the torch tests.
"""

import numpy as np
import pandas as pd
import pytest

keras = pytest.importorskip("keras", reason="keras is not installed")

from pyflexplot.flex_nn import (  # noqa: E402
    NeuralNetFit,
    is_keras_model,
    keras_backend_available,
    permutation_importance,
    prepare_torch_data,
    set_response_var,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_keras_regressor(seed: int = 0):
    """Return a tiny compiled Keras 3 regressor trained on synthetic data."""
    rng = np.random.default_rng(seed)
    n = 200
    X = rng.normal(size=(n, 3)).astype("float32")
    true_beta = np.array([1.5, -2.0, 0.7], dtype="float32")
    y = (X @ true_beta) + rng.normal(scale=0.1, size=n).astype("float32")

    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(3,)),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.05), loss="mse")
    model.fit(X, y, epochs=200, batch_size=16, verbose=0)
    return model, X, y


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def test_keras_backend_available():
    assert keras_backend_available() is True


def test_is_keras_model_recognises_keras_model():
    model, _, _ = _build_keras_regressor()
    assert is_keras_model(model) is True
    assert is_keras_model("not a model") is False


# ---------------------------------------------------------------------------
# set_response_var
# ---------------------------------------------------------------------------

def test_set_response_var_attaches_attribute_on_keras_model():
    model, _, _ = _build_keras_regressor()
    set_response_var(model, "y_target")
    assert getattr(model, "_pyflexplot_response_var") == "y_target"


# ---------------------------------------------------------------------------
# NeuralNetFit with keras backend
# ---------------------------------------------------------------------------

class TestNeuralNetFitKeras:
    def test_backend_inferred_as_keras(self):
        model, _, _ = _build_keras_regressor()
        fit = NeuralNetFit(
            model=model,
            response_var="y",
            predictor_names=["x1", "x2", "x3"],
        )
        assert fit.backend == "keras"

    def test_predict_returns_indexed_series(self):
        model, _, _ = _build_keras_regressor()
        fit = NeuralNetFit(
            model=model,
            response_var="y",
            predictor_names=["x1", "x2", "x3"],
        )
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.normal(size=(10, 3)), columns=["x1", "x2", "x3"])
        pred = fit.predict(df)
        assert isinstance(pred, pd.Series)
        assert len(pred) == len(df)
        assert pred.index.equals(df.index)
        assert pred.name == "y__pred"

    def test_predict_handles_two_d_output(self):
        # Keras returns shape (n, 1) for a single-output Dense(1).  The
        # wrapper should ravel it to a 1-D Series.
        model, _, _ = _build_keras_regressor()
        fit = NeuralNetFit(
            model=model,
            response_var="y",
            predictor_names=["x1", "x2", "x3"],
        )
        df = pd.DataFrame(np.zeros((5, 3)), columns=["x1", "x2", "x3"])
        pred = fit.predict(df)
        assert pred.ndim == 1
        assert len(pred) == 5

    def test_predict_respects_zscore_normalisation(self):
        model, _, _ = _build_keras_regressor()
        means = np.zeros(3)
        sds = np.ones(3)
        fit = NeuralNetFit(
            model=model,
            response_var="y",
            predictor_names=["x1", "x2", "x3"],
            x_means=means,
            x_sds=sds,
        )
        df = pd.DataFrame(np.zeros((3, 3)), columns=["x1", "x2", "x3"])
        pred = fit.predict(df)
        # With zero inputs and identity normaliser, prediction should match
        # what the underlying model returns for an all-zero input -- a small
        # finite value, not a NaN or unbounded number.
        assert np.isfinite(pred.to_numpy()).all()

    def test_predict_with_dropout_layer_is_deterministic(self):
        # Models with Dropout are the *reason* we pass training=False.  Make
        # sure repeated predictions return the same values.
        keras.utils.set_random_seed(0)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3)).astype("float32")
        y = X @ np.array([0.5, 0.5, 0.5], dtype="float32")

        model = keras.Sequential([
            keras.layers.Input(shape=(3,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dropout(0.5),  # would be stochastic without training=False
            keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=20, verbose=0)

        fit = NeuralNetFit(
            model=model,
            response_var="y",
            predictor_names=["x1", "x2", "x3"],
        )
        df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
        a = fit.predict(df).to_numpy()
        b = fit.predict(df).to_numpy()
        np.testing.assert_array_equal(a, b)

    def test_explicit_backend_keras(self):
        model, _, _ = _build_keras_regressor()
        fit = NeuralNetFit(
            model=model,
            response_var="y",
            predictor_names=["x1", "x2", "x3"],
            backend="keras",
        )
        assert fit.backend == "keras"

    def test_explicit_backend_keras_when_keras_missing(self, monkeypatch):
        # If the user passes backend="keras" but keras isn't importable,
        # construction must fail loudly with a clear message.
        from pyflexplot import flex_nn as fn_mod
        monkeypatch.setattr(fn_mod, "_KERAS_AVAILABLE", False)
        model, _, _ = _build_keras_regressor()
        with pytest.raises(RuntimeError, match="backend='keras'"):
            NeuralNetFit(
                model=model,
                response_var="y",
                predictor_names=["x1", "x2", "x3"],
                backend="keras",
            )

    def test_repr_includes_backend(self):
        model, _, _ = _build_keras_regressor()
        fit = NeuralNetFit(
            model=model,
            response_var="y",
            predictor_names=["x1", "x2", "x3"],
        )
        assert "keras" in repr(fit)
        assert "params=" in repr(fit)


# ---------------------------------------------------------------------------
# permutation_importance on a Keras model
# ---------------------------------------------------------------------------

class TestPermutationImportanceKeras:
    def test_runs_against_keras_model(self):
        model, X, y = _build_keras_regressor()
        fit = NeuralNetFit(
            model=model,
            response_var="y",
            predictor_names=["x1", "x2", "x3"],
        )
        df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
        result = permutation_importance(
            fit, df, y, metric="mse", random_state=0,
        )
        assert len(result) == 3
        assert set(result["variable"]) == {"x1", "x2", "x3"}
        # Results should be sorted descending by importance.
        importances = result["importance"].to_numpy()
        assert (importances[:-1] >= importances[1:]).all()

    def test_seed_reproducible(self):
        model, X, y = _build_keras_regressor()
        fit = NeuralNetFit(
            model=model,
            response_var="y",
            predictor_names=["x1", "x2", "x3"],
        )
        df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
        a = permutation_importance(fit, df, y, metric="mse", random_state=7)
        b = permutation_importance(fit, df, y, metric="mse", random_state=7)
        np.testing.assert_array_equal(a["importance"].to_numpy(), b["importance"].to_numpy())


# ---------------------------------------------------------------------------
# prepare_torch_data is backend-agnostic -- one sanity check here too.
# ---------------------------------------------------------------------------

def test_prepare_torch_data_works_for_keras_pipeline():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    out = prepare_torch_data(df)
    np.testing.assert_array_equal(
        out, np.array([[1, 4], [2, 5], [3, 6]], dtype=float)
    )