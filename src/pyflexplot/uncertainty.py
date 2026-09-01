"""Uncertainty quantification for fitted lines.

Public surface:
- ``validate_uncertainty_params``: validation entry point used by
  :func:`pyflexplot.core.flexplot`.
- ``compute_bootstrap_ci``: case-resampled bootstrap CI for a smoother
  (used when ``uncertainty='bootstrap'`` on the loess branch).
- ``compute_prediction_band``: residual-based symmetric prediction
  interval (used when ``uncertainty='prediction'`` on the numeric branch).
- ``format_band_label``: legend-label helper used by the plot layer.

Design notes
------------
- Bootstrap uses case (row) resampling, not residual bootstrap, to be
  robust to model misspecification.
- Prediction intervals assume approximately normal residuals. For
  non-normal residuals, prefer bootstrap (loess) or transform the outcome.
- ``bands`` (nested coverage levels) is layered at the flexplot() call
  site; this module only knows how to compute a single band.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# Public set used by core.flexplot() to validate the user-facing
# ``uncertainty`` parameter.
VALID_UNCERTAINTY = frozenset({None, "ci", "prediction", "bootstrap"})


def validate_uncertainty_params(
    uncertainty: Optional[str],
    level: Optional[float],
    bands: Optional[list],
    method: Optional[str],
) -> None:
    """Validate uncertainty-related parameters and method compatibility.

    Raises ``ValueError`` with a precise message on the first violation.
    """
    if uncertainty not in VALID_UNCERTAINTY:
        raise ValueError(
            f"uncertainty must be one of {sorted(VALID_UNCERTAINTY, key=str)}; "
            f"got {uncertainty!r}."
        )

    if level is not None:
        if not isinstance(level, (int, float)) or not 0 < level < 1:
            raise ValueError(
                f"level must be a number in (0, 1); got {level!r}."
            )

    if bands is not None:
        if not isinstance(bands, (list, tuple)):
            raise ValueError(
                f"bands must be a list or tuple of floats; "
                f"got type {type(bands).__name__}."
            )
        for b in bands:
            if not isinstance(b, (int, float)) or not 0 < b < 1:
                raise ValueError(
                    f"Each band level must be a number in (0, 1); got {b!r}."
                )

    # Method/uncertainty compatibility.
    if uncertainty == "bootstrap" and method not in ("loess", "auto"):
        raise ValueError(
            f"uncertainty='bootstrap' is only supported for method='loess'; "
            f"got method={method!r}. Use 'ci' or 'prediction' for LM fits."
        )


def compute_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    smooth_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_resamples: int = 200,
    level: float = 0.95,
    x_eval: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Case-resampled bootstrap CI for a smoother.

    Parameters
    ----------
    x, y : 1-D arrays of equal length
    smooth_fn : callable(x_eval, y_at_x_eval_sorted_by_x) -> yhat
        Fitted values evaluated at ``x_eval``. The smoother is invoked
        once on the full data and ``n_resamples`` times on bootstrap
        samples.
    n_resamples : int, default 200
    level : float in (0, 1), default 0.95
    x_eval : 1-D array of points at which to evaluate the smoother;
        defaults to the sorted unique ``x`` values.
    random_state : int or None for non-deterministic

    Returns
    -------
    (x_eval, lower, upper) : three 1-D arrays of equal length.
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n != len(y):
        raise ValueError(
            f"x and y must have equal length; got {n} and {len(y)}."
        )
    if x_eval is None:
        x_eval = np.sort(np.unique(x))
    x_eval = np.asarray(x_eval)

    # Fit on full data first (for fallback on failed bootstrap samples).
    yhat_full = smooth_fn(x_eval, y[np.argsort(x)])

    boot_preds = np.empty((n_resamples, len(x_eval)))
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        x_b, y_b = x[idx], y[idx]
        try:
            boot_preds[i] = smooth_fn(x_eval, y_b[np.argsort(x_b)])
        except Exception:
            # Singular fits or other numerical issues fall back to the
            # full-data fit for that resample. This avoids losing
            # observations from the percentile calculation.
            boot_preds[i] = yhat_full

    alpha = 1.0 - level
    lower = np.percentile(boot_preds, 100.0 * alpha / 2.0, axis=0)
    upper = np.percentile(boot_preds, 100.0 * (1.0 - alpha / 2.0), axis=0)

    return x_eval, lower, upper


def compute_prediction_band(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric residual-based prediction interval.

    Assumes approximately normal residuals with constant variance.
    Uses the residual standard error (sigma) computed with ddof=0
    (matches OLS residual-variance convention).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got "
            f"{y_true.shape} and {y_pred.shape}."
        )
    residuals = y_true - y_pred
    sigma = float(np.sqrt(np.mean(residuals ** 2)))
    z = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    half_width = z * sigma
    lower = y_pred - half_width
    upper = y_pred + half_width
    return lower, upper


def format_band_label(level: float, kind: str = "ci") -> str:
    """Format a legend label for a band.

    Examples
    --------
    >>> format_band_label(0.95)
    '95% CI'
    >>> format_band_label(0.80, kind='prediction')
    '80% PI'
    >>> format_band_label(0.95, kind='bootstrap')
    '95% bootstrap CI'
    """
    pct = int(round(level * 100))
    if kind == "ci":
        return f"{pct}% CI"
    if kind == "prediction":
        return f"{pct}% PI"
    if kind == "bootstrap":
        return f"{pct}% bootstrap CI"
    return f"{pct}% {kind}"