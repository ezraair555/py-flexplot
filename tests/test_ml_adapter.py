"""Tests for pyflexplot.ml.RFAdapter.

Thin wrapper around sklearn estimators so they can be used with
compare_fits() and the rest of the py-flexplot visualization surface.
"""

import numpy as np
import pandas as pd
import pytest

# sklearn is required for these tests. Skip if unavailable.
pytest.importorskip("sklearn")
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

from pyflexplot.ml import RFAdapter, make_rf_adapter, _SKLEARN_AVAILABLE


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x1": rng.normal(size=100),
        "x2": rng.normal(size=100),
        "y": rng.normal(size=100) + 0.5 * rng.normal(size=100),
    })


@pytest.fixture
def classification_df():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=100)
    return pd.DataFrame({
        "x1": x1,
        "x2": rng.normal(size=100),
        "y": (x1 + rng.normal(scale=0.5, size=100) > 0).astype(int),
    })


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_sklearn_available():
    """Sanity check: sklearn is importable in the test environment."""
    assert _SKLEARN_AVAILABLE is True


def test_rf_adapter_constructs_from_regressor(regression_df):
    """RFAdapter accepts a fitted RF regressor + metadata."""
    X = regression_df[["x1", "x2"]]
    y = regression_df["y"]
    rf = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    adapter = RFAdapter(rf, response_var="y", predictor_names=["x1", "x2"])
    assert adapter.response_var == "y"
    assert adapter.predictor_names == ["x1", "x2"]


def test_rf_adapter_rejects_non_string_predictor_names():
    """predictor_names with non-string entries raises TypeError."""
    rf = RandomForestRegressor(n_estimators=2, random_state=0).fit(
        np.random.default_rng(0).normal(size=(10, 2)),
        np.random.default_rng(1).normal(size=10),
    )
    with pytest.raises(TypeError, match="must all be strings"):
        RFAdapter(rf, response_var="y", predictor_names=["x1", 42])


def test_rf_adapter_rejects_non_string_response_var():
    """response_var=42 raises TypeError."""
    rf = RandomForestRegressor(n_estimators=2, random_state=0).fit(
        np.random.default_rng(0).normal(size=(10, 2)),
        np.random.default_rng(1).normal(size=10),
    )
    with pytest.raises(TypeError, match="response_var must be a string"):
        RFAdapter(rf, response_var=42, predictor_names=["x1", "x2"])


def test_make_rf_adapter_infers_predictor_names(regression_df):
    """make_rf_adapter pulls predictor names from data.columns when omitted."""
    X = regression_df[["x1", "x2"]]
    y = regression_df["y"]
    rf = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    adapter = make_rf_adapter(rf, data=regression_df, response_var="y")
    assert set(adapter.predictor_names) == {"x1", "x2"}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_rf_adapter_predict_on_dataframe(regression_df):
    """predict(X) where X is a DataFrame aligns columns by name."""
    X = regression_df[["x1", "x2"]]
    y = regression_df["y"]
    rf = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    adapter = RFAdapter(rf, response_var="y", predictor_names=["x1", "x2"])
    preds = adapter.predict(regression_df)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(regression_df),)


def test_rf_adapter_predict_on_numpy(regression_df):
    """predict(X) where X is a numpy array uses positional access."""
    X = regression_df[["x1", "x2"]].to_numpy()
    y = regression_df["y"].to_numpy()
    rf = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    adapter = RFAdapter(rf, response_var="y", predictor_names=["x1", "x2"])
    preds = adapter.predict(X)
    assert preds.shape == (len(regression_df),)


def test_rf_adapter_predict_df_returns_single_column(regression_df):
    """predict_df(data) returns a DataFrame with pred_<response_var> column."""
    X = regression_df[["x1", "x2"]]
    y = regression_df["y"]
    rf = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    adapter = RFAdapter(rf, response_var="y", predictor_names=["x1", "x2"])
    out = adapter.predict_df(regression_df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["pred_y"]
    assert len(out) == len(regression_df)


def test_rf_adapter_dataframe_with_extra_columns(regression_df):
    """predict() on a DataFrame with extra (non-predictor) columns still works."""
    X = regression_df[["x1", "x2"]]
    y = regression_df["y"]
    rf = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    adapter = RFAdapter(rf, response_var="y", predictor_names=["x1", "x2"])
    # Add an extra column; adapter should ignore it (column alignment by name).
    regression_df["extra"] = 1.0
    preds = adapter.predict(regression_df)
    assert preds.shape == (len(regression_df),)


# ---------------------------------------------------------------------------
# Integration: compare_fits
# ---------------------------------------------------------------------------


def test_rf_adapter_works_with_compare_fits(regression_df):
    """An RFAdapter can be passed to compare_fits alongside a statsmodels fit."""
    import statsmodels.formula.api as smf
    from pyflexplot import compare_fits

    X = regression_df[["x1", "x2"]]
    y = regression_df["y"]
    rf = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    adapter = make_rf_adapter(rf, data=regression_df, response_var="y")

    ols = smf.ols("y ~ x1 + x2", data=regression_df).fit()
    # compare_fits accepts the adapter (which exposes .predict()).
    p = compare_fits("y ~ x1 + x2", data=regression_df, model1=ols, model2=adapter)
    assert p is not None  # ggplot returned


def test_rf_adapter_classifier(classification_df):
    """An RFAdapter wrapping a classifier still has a working .predict()."""
    X = classification_df[["x1", "x2"]]
    y = classification_df["y"]
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    adapter = RFAdapter(rf, response_var="y", predictor_names=["x1", "x2"])
    preds = adapter.predict(classification_df)
    # Classifier predictions are 0/1.
    assert set(np.unique(preds)).issubset({0, 1})


def test_rf_adapter_arbitrary_sklearn_estimator(regression_df):
    """RFAdapter works with any sklearn estimator that has .predict()."""
    X = regression_df[["x1", "x2"]].to_numpy()
    y = regression_df["y"].to_numpy()
    lr = LinearRegression().fit(X, y)
    adapter = RFAdapter(lr, response_var="y", predictor_names=["x1", "x2"])
    preds = adapter.predict(regression_df)
    assert preds.shape == (len(regression_df),)