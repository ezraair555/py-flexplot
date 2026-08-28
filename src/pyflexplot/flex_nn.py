"""
flex_nn: Neural-network visualization utilities for py-flexplot.

This module is a Python port of the spirit (not the surface) of Dustin Fife's
``flex_nn`` R package (https://github.com/dustinfife/flex_nn). The R package
extends ``flexplot`` to handle Keras/TensorFlow models in ``compare.fits()`` and
related calls. In Python we cover the same conceptual surface for ``torch``
models (default backend) and provide a thin Keras 3 shim when available.

The module deliberately does NOT fit neural networks -- that is left to the
caller. Its job is to wrap an already-trained network with the metadata
required to make it a first-class citizen of py-flexplot's visualization API:

    * ``set_response_var(model, name)`` -- attach the response variable name.
    * ``NeuralNetFit`` -- a thin wrapper bundling the network with training
      metadata so ``compare_fits()`` and friends can call ``.predict()`` and
      get a properly-aligned ``pandas.Series``.
    * ``permutation_importance(fit, X, y, metric=None)`` -- variable-importance
      via column-wise shuffling, mirroring the R implementation.
    * ``prepare_torch_data(data, categorical_vars=None)`` -- DataFrame -> 2-D
      float tensor, with deterministic integer encoding for categoricals.

Backend selection
-----------------
The default backend is ``torch``. ``keras`` is supported opportunistically:
if ``keras`` is importable, ``is_keras_model(obj)`` will recognise
``keras.Model`` instances, and the same ``NeuralNetFit`` class wraps them
transparently. No keras dependency is declared in ``pyproject.toml`` -- the
support is best-effort and tested only when keras is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    import torch  # noqa: F401  -- presence is detected by ``_torch_available``
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover -- environment without torch
    _TORCH_AVAILABLE = False

try:
    import keras  # noqa: F401
    _KERAS_AVAILABLE = True
except Exception:  # pragma: no cover -- environment without keras
    _KERAS_AVAILABLE = False


__all__ = [
    "NeuralNetFit",
    "is_keras_model",
    "is_torch_model",
    "set_response_var",
    "permutation_importance",
    "prepare_torch_data",
    "torch_backend_available",
    "keras_backend_available",
]


# ---------------------------------------------------------------------------
# Backend detection helpers
# ---------------------------------------------------------------------------

def torch_backend_available() -> bool:
    """Return True if a usable ``torch`` is importable."""
    return _TORCH_AVAILABLE


def keras_backend_available() -> bool:
    """Return True if a usable ``keras`` is importable."""
    return _KERAS_AVAILABLE


def is_torch_model(obj: Any) -> bool:
    """Return True if *obj* looks like a ``torch.nn.Module``."""
    if not _TORCH_AVAILABLE:
        return False
    import torch as _torch
    return isinstance(obj, _torch.nn.Module)


def is_keras_model(obj: Any) -> bool:
    """Return True if *obj* is a ``keras.Model`` instance."""
    if not _KERAS_AVAILABLE:
        return False
    import keras as _keras
    return isinstance(obj, _keras.Model)


# ---------------------------------------------------------------------------
# prepare_torch_data
# ---------------------------------------------------------------------------

def prepare_torch_data(
    data: pd.DataFrame,
    categorical_vars: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Convert a DataFrame into the dense float matrix a network expects.

    Categorical columns (in *categorical_vars* that also exist in *data*) are
    integer-encoded starting at zero.  All other columns are coerced to
    float.  Missing values raise ``ValueError`` -- imputation is the caller's
    responsibility, mirroring the R package's strict behaviour.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data must be a pandas DataFrame, got {type(data).__name__}"
        )
    if data.empty:
        raise ValueError("data must be a non-empty DataFrame")

    out = data.copy()
    if categorical_vars:
        existing = [c for c in categorical_vars if c in out.columns]
        for col in existing:
            out[col] = out[col].astype("category").cat.codes.astype(float)

    if out.isna().any().any():
        missing = sorted(out.columns[out.isna().any()])
        raise ValueError(
            "prepare_torch_data does not impute missing values; "
            f"found NaN in columns: {missing}"
        )

    return out.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# set_response_var
# ---------------------------------------------------------------------------

def set_response_var(model: Any, response_var: str) -> Any:
    """Attach the response-variable name as an attribute on *model*.

    Works for any object that supports ``setattr`` -- typically a fitted
    ``torch.nn.Module`` or ``keras.Model``.  The attribute is consulted by
    ``NeuralNetFit`` and the visualization paths so the network knows which
    column of *data* it is predicting.
    """
    if not isinstance(response_var, str) or not response_var:
        raise ValueError(
            f"response_var must be a non-empty string, got {response_var!r}"
        )
    try:
        setattr(model, "_pyflexplot_response_var", response_var)
    except Exception as exc:  # pragma: no cover -- pytorch modules allow this
        raise TypeError(
            f"Cannot set attribute on model of type {type(model).__name__}: {exc}"
        ) from exc
    return model


def _get_response_var(model: Any) -> Optional[str]:
    """Internal: read back the response variable attached by ``set_response_var``."""
    return getattr(model, "_pyflexplot_response_var", None)


# ---------------------------------------------------------------------------
# NeuralNetFit
# ---------------------------------------------------------------------------

@dataclass
class NeuralNetFit:
    """Wrapper bundling a fitted network with the metadata flexplot needs.

    Attributes
    ----------
    model
        The underlying fitted network (``torch.nn.Module`` or ``keras.Model``).
    response_var
        Name of the column the network predicts.
    predictor_names
        Ordered names of the input columns the network was trained on.
    x_means, x_sds
        Per-column mean/sd used for z-score standardisation, if any. ``None``
        means the network was trained on raw inputs.
    history
        Free-form training history (e.g. a Keras ``History`` object, a list
        of torch loss/epoch dicts, or simply ``None``).
    backend
        Either ``"torch"`` or ``"keras"``; inferred if not supplied.
    """

    model: Any
    response_var: str
    predictor_names: List[str]
    x_means: Optional[np.ndarray] = None
    x_sds: Optional[np.ndarray] = None
    history: Any = None
    backend: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.response_var, str) or not self.response_var:
            raise ValueError(
                f"response_var must be a non-empty string, got {self.response_var!r}"
            )
        if not isinstance(self.predictor_names, (list, tuple)):
            raise TypeError(
                f"predictor_names must be a list/tuple, got {type(self.predictor_names).__name__}"
            )
        if len(self.predictor_names) == 0:
            raise ValueError(
                "predictor_names must contain at least one predictor name"
            )
        if len(set(self.predictor_names)) != len(self.predictor_names):
            raise ValueError("predictor_names must not contain duplicates")

        if self.backend is None:
            if is_torch_model(self.model):
                self.backend = "torch"
            elif is_keras_model(self.model):
                self.backend = "keras"
            else:
                raise TypeError(
                    "NeuralNetFit.model must be a torch.nn.Module or keras.Model "
                    f"(got {type(self.model).__name__}); set backend explicitly "
                    "if you are wrapping a duck-typed object."
                )

        if self.backend == "torch" and not _TORCH_AVAILABLE:
            raise RuntimeError(
                "backend='torch' but torch is not importable in this environment"
            )
        if self.backend == "keras" and not _KERAS_AVAILABLE:
            raise RuntimeError(
                "backend='keras' but keras is not importable in this environment"
            )

        # Honour set_response_var() on the wrapped model if response_var is
        # still default-ish, but always trust the explicit constructor arg.
        existing = _get_response_var(self.model)
        if existing is None:
            set_response_var(self.model, self.response_var)

    # -- prediction --------------------------------------------------------

    def _prepare_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Slice *data* to *predictor_names* and apply stored normalisation."""
        missing = [c for c in self.predictor_names if c not in data.columns]
        if missing:
            raise ValueError(
                f"data is missing predictors required by NeuralNetFit: {missing}"
            )
        X = data[list(self.predictor_names)].to_numpy(dtype=float)
        if self.x_means is not None and self.x_sds is not None:
            X = (X - self.x_means) / np.where(self.x_sds == 0, 1, self.x_sds)
        return X

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Return predictions for *data* aligned to its row index.

        For Keras models, predictions are made with ``training=False`` so
        dropout/batchnorm layers behave as they did at evaluation time.  For
        Torch models the computation runs under ``torch.no_grad()``.
        """
        X = self._prepare_matrix(data)

        if self.backend == "torch":
            import torch as _torch
            with _torch.no_grad():
                tensor = _torch.as_tensor(X, dtype=_torch.float32)
                raw = self.model(tensor)
            arr = raw.detach().cpu().numpy()

        elif self.backend == "keras":
            arr = self._keras_predict(X)

        else:  # pragma: no cover -- validated in __post_init__
            raise RuntimeError(f"unsupported backend: {self.backend!r}")

        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        elif arr.ndim > 2:
            raise ValueError(
                f"Network predictions must be 1-D (or 2-D with one output column); "
                f"got shape {arr.shape}"
            )

        return pd.Series(arr, index=data.index, name=f"{self.response_var}__pred")

    def _keras_predict(self, X: np.ndarray) -> np.ndarray:
        """Call a Keras model in inference mode.

        Keras3's ``Model.predict`` accepts a ``training=False`` argument; we
        pass it explicitly so models with Dropout or BatchNorm behave the
        same way they did during validation.  Some custom ``Model`` subclasses
        don't accept ``training`` as a kwarg -- we retry without it in that
        case, after setting the model's ``training`` attribute to False if
        available.
        """
        import keras as _keras

        # If the model exposes a mutable training flag, force inference first
        # so a custom call() implementation that ignores the kwarg still works.
        if hasattr(self.model, "training"):
            try:
                self.model.training = False
            except Exception:
                pass

        try:
            return np.asarray(self.model.predict(X, verbose=0, training=False))
        except TypeError:
            # Fall back for custom Models whose call() doesn't accept training=.
            return np.asarray(self.model.predict(X, verbose=0))

    # -- introspection -----------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover -- cosmetic
        n_params = None
        if self.backend == "torch":
            try:
                n_params = sum(p.numel() for p in self.model.parameters())
            except Exception:
                pass
        elif self.backend == "keras":
            try:
                n_params = self.model.count_params()
            except Exception:
                pass
        head = f"NeuralNetFit(backend={self.backend!r}, response={self.response_var!r}"
        if n_params is not None:
            head += f", params={n_params}"
        head += f", predictors={len(self.predictor_names)})"
        return head


# ---------------------------------------------------------------------------
# permutation_importance
# ---------------------------------------------------------------------------

_DEFAULT_METRICS = {
    # metrics where higher is better -- importance = baseline - permuted
    "accuracy": "higher",
    "auc": "higher",
    "precision": "higher",
    "recall": "higher",
    "f1": "higher",
    "r2": "higher",
    # metrics where lower is better -- importance = permuted - baseline
    "loss": "lower",
    "mse": "lower",
    "mae": "lower",
    "mean_absolute_error": "lower",
    "mean_squared_error": "lower",
    "rmse": "lower",
}


def _default_metric(model: Any, backend: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return a sensible default scorer for *model*."""
    if backend == "torch":
        # For torch models we default to MSE -- the most common regression
        # loss and the analogue of the R package's mean_absolute_error default.
        def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
        return _mse
    if backend == "keras":
        last = getattr(model, "loss", None)
        if isinstance(last, str) and "binary" in last:
            def _wrong(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                return float(np.mean((np.asarray(y_true) > 0.5) != (np.asarray(y_pred) > 0.5)))
            return _wrong
        def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
        return _mse
    raise ValueError(f"unsupported backend: {backend!r}")


def permutation_importance(
    fit: NeuralNetFit,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray, list],
    *,
    metric: Optional[Union[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    n_repeats: int = 1,
    random_state: Optional[int] = None,
    higher_is_better: Optional[bool] = None,
) -> pd.DataFrame:
    """Permutation feature importance for a fitted neural network.

    Parameters
    ----------
    fit
        A :class:`NeuralNetFit` wrapper.
    X
        Predictor matrix used to score the model.
    y
        True response -- either a ``pd.Series`` aligned to ``X.index`` or a
        1-D array/list of the same length as ``X``.
    metric
        Either the name of a known metric (``"mse"``, ``"mae"``,
        ``"accuracy"``, ...) or a callable ``(y_true, y_pred) -> float``.
        ``None`` picks a backend-aware default.
    n_repeats
        How many independent permutations to average per column.
    random_state
        Optional seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Columns ``variable``, ``importance`` (higher = more important),
        ``baseline`` (the unscored metric value), sorted by importance
        descending.
    """
    if not isinstance(fit, NeuralNetFit):
        raise TypeError(
            f"fit must be a NeuralNetFit, got {type(fit).__name__}"
        )
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"X must be a pandas DataFrame, got {type(X).__name__}"
        )
    if X.empty:
        raise ValueError("X must be non-empty")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    if isinstance(y, pd.Series):
        if not y.index.equals(X.index):
            y = y.reindex(X.index)
        y_arr = y.to_numpy()
    else:
        y_arr = np.asarray(y)
        if y_arr.shape[0] != len(X):
            raise ValueError(
                f"y must have the same length as X ({len(X)}); got {y_arr.shape[0]}"
            )

    rng = np.random.default_rng(random_state)

    scorer: Callable[[np.ndarray, np.ndarray], float]
    direction: Optional[bool]  # True = higher is better
    if metric is None:
        scorer = _default_metric(fit.model, fit.backend)
        # MSE / wrong-rate: lower is better.
        direction = False
    elif callable(metric):
        if higher_is_better is None:
            raise ValueError(
                "When metric is a callable, higher_is_better must be True or False "
                "so importance has a defined sign."
            )
        scorer = metric
        direction = bool(higher_is_better)
    elif isinstance(metric, str):
        key = metric.lower()
        if key not in _DEFAULT_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}. Pass a callable scorer or one of "
                f"{sorted(_DEFAULT_METRICS)}."
            )
        direction = _DEFAULT_METRICS[key] == "higher"
        if key in ("mse", "mean_squared_error"):
            def scorer(yt, yp):
                return float(np.mean((np.asarray(yt) - np.asarray(yp)) ** 2))
        elif key in ("mae", "mean_absolute_error"):
            def scorer(yt, yp):
                return float(np.mean(np.abs(np.asarray(yt) - np.asarray(yp))))
        elif key in ("rmse",):
            def scorer(yt, yp):
                return float(np.sqrt(np.mean((np.asarray(yt) - np.asarray(yp)) ** 2)))
        elif key in ("r2",):
            def scorer(yt, yp):
                yt = np.asarray(yt); yp = np.asarray(yp)
                ss_res = float(np.sum((yt - yp) ** 2))
                ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
                return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        elif key == "accuracy":
            def scorer(yt, yp):
                return float(np.mean((np.asarray(yt) > 0.5) == (np.asarray(yp) > 0.5)))
        if direction is None:
            # Best-effort default for named metrics not explicitly handled above.
            def scorer(yt, yp):
                # Best-effort default: classification accuracy for >0.5,
                # MSE for everything else.
                yt = np.asarray(yt); yp = np.asarray(yp)
                if set(np.unique(yt)).issubset({0, 1}):
                    return float(np.mean((yt > 0.5) == (yp > 0.5)))
                return float(np.mean((yt - yp) ** 2))
    else:
        raise TypeError(
            f"metric must be None, str, or callable; got {type(metric).__name__}"
        )

    # Build the scored-once matrix for the baseline score.
    baseline_pred = fit.predict(X).to_numpy()
    baseline_score = scorer(y_arr, baseline_pred)

    n_cols = X.shape[1]
    columns = list(X.columns)
    if len(columns) != n_cols:
        columns = [f"x{i}" for i in range(n_cols)]

    X_arr = X.to_numpy()
    importances = np.zeros(n_cols, dtype=float)

    for j in range(n_cols):
        scores: List[float] = []
        for _ in range(n_repeats):
            X_perm = X_arr.copy()
            X_perm[:, j] = X_arr[rng.permutation(X_arr.shape[0]), j]
            perm_df = pd.DataFrame(X_perm, columns=columns, index=X.index)
            perm_pred = fit.predict(perm_df).to_numpy()
            scores.append(scorer(y_arr, perm_pred))
        mean_perm = float(np.mean(scores))
        if direction:
            importances[j] = baseline_score - mean_perm
        else:
            importances[j] = mean_perm - baseline_score

    out = pd.DataFrame({
        "variable": columns,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    out.attrs["baseline_score"] = baseline_score
    out.attrs["metric"] = getattr(metric, "__name__", metric)
    return out