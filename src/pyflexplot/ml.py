"""
ml: Random-forest (and sklearn-estimator) wrappers for py-flexplot.

This module provides thin adapters that make scikit-learn estimators
(``RandomForestRegressor``, ``RandomForestClassifier``, ``GradientBoosting*``,
and any estimator with a ``.predict()`` method) usable with py-flexplot's
visualization API: ``compare_fits()``, ``flexplot(overlay=...)``,
``visualize()``, and ``estimates()``.

It does NOT fit models. Its job is to wrap an already-trained estimator
with the metadata required to make it a first-class citizen of
py-flexplot's visualization API:

* ``RFAdapter`` -- thin wrapper bundling an sklearn regressor/classifier
  with column-name metadata so ``compare_fits()`` and friends can call
  ``.predict()`` and get a properly-aligned ``pandas.Series``.

Why an adapter?
---------------
scikit-learn estimators expose ``.predict(X)`` but not the predictor
names; py-flexplot's visualization layer needs to know which columns
the model used so it can build evaluation DataFrames without losing
row alignment. ``RFAdapter`` carries that metadata alongside the
fitted model.

Usage example::

    from sklearn.ensemble import RandomForestRegressor
    from pyflexplot.ml import RFAdapter

    rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X, y)
    fit = RFAdapter(rf, response_var="y", predictor_names=list(X.columns))

    # Now use it in compare_fits:
    from pyflexplot import compare_fits
    p = compare_fits("y ~ x1 + x2", data=df, model1=statsmodels_fit, model2=fit)

This is intentionally a thin glue layer — R-flexplot's
``flexplot::compare.fits()`` accepted any R model with a ``predict()``
method, and that's the surface this module recreates in Python.

Optional dependency
-------------------
scikit-learn is **not** a declared dependency of py-flexplot; the
adapter raises a clear ``ImportError`` at import time if sklearn is
missing, and py-flexplot's core surface works fine without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator  # noqa: F401
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover -- environment without sklearn
    _SKLEARN_AVAILABLE = False


def _check_sklearn_available():
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for pyflexplot.ml; install with "
            "`pip install scikit-learn`."
        )


@dataclass
class RFAdapter:
    """Adapter that makes an sklearn estimator usable with py-flexplot.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        A fitted scikit-learn estimator with a ``predict(X)`` method.
        ``RandomForestRegressor`` / ``RandomForestClassifier`` are the
        primary targets, but any regressor / classifier works.
    response_var : str
        Name of the response variable in the original DataFrame. Used by
        py-flexplot to build evaluation DataFrames.
    predictor_names : list of str
        Column names of the predictors used during fitting. Order matters;
        must match the column order in the X matrix passed to ``.fit()``.

    Notes
    -----
    The adapter deliberately does not subclass any sklearn base class to
    avoid surprising scikit-learn's ``check_is_fitted`` machinery. It is
    a pure metadata wrapper.
    """

    estimator: Any
    response_var: str
    predictor_names: List[str]

    def __post_init__(self):
        _check_sklearn_available()
        if not isinstance(self.predictor_names, (list, tuple)):
            raise TypeError(
                f"predictor_names must be a list/tuple of strings; "
                f"got {type(self.predictor_names).__name__}."
            )
        if any(not isinstance(n, str) for n in self.predictor_names):
            raise TypeError("predictor_names must all be strings.")
        if not isinstance(self.response_var, str):
            raise TypeError(
                f"response_var must be a string; got {type(self.response_var).__name__}."
            )

    def predict(self, X):
        """Predict on a DataFrame or 2-D array.

        Accepts either a ``pandas.DataFrame`` (column-aligned by name when
        possible) or a 2-D ``numpy.ndarray``. Returns a 1-D
        ``numpy.ndarray``.
        """
        if isinstance(X, pd.DataFrame):
            # Align by name when the columns match; otherwise fall back to
            # positional access (the caller has positional data).
            missing = [c for c in self.predictor_names if c not in X.columns]
            if not missing:
                X_arr = X[self.predictor_names].to_numpy()
            else:
                X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)
        return np.asarray(self.estimator.predict(X_arr))

    def predict_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict on a DataFrame and return a single-column DataFrame.

        The returned column is named ``pred_<response_var>`` to match the
        convention used by ``compare_fits(return_preds=True)``.
        """
        preds = self.predict(data)
        return pd.DataFrame({f"pred_{self.response_var}": preds})


def make_rf_adapter(
    estimator: Any,
    data: pd.DataFrame,
    response_var: str,
    predictor_names: Optional[Sequence[str]] = None,
) -> RFAdapter:
    """Convenience constructor that pulls predictor names from a DataFrame.

    Parameters
    ----------
    estimator
        A fitted sklearn estimator.
    data : pd.DataFrame
        The training DataFrame. Used to infer ``predictor_names`` if not
        provided.
    response_var
        Name of the response column.
    predictor_names
        Explicit list of predictor column names. If ``None`` (default),
        uses ``[c for c in data.columns if c != response_var]``.

    Returns
    -------
    RFAdapter
    """
    _check_sklearn_available()
    if predictor_names is None:
        predictor_names = [c for c in data.columns if c != response_var]
    return RFAdapter(
        estimator=estimator,
        response_var=response_var,
        predictor_names=list(predictor_names),
    )