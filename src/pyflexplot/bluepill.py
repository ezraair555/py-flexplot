"""
bluepill: simulated-dataset utilities for py-flexplot.

This module is a Python port of Dustin Fife's ``bluepill`` R package
(https://github.com/dustinfife/bluepill -- "An R package for creating
simulated dataset.").  It exposes two ideas from that package:

    * ``estimate_sd(mean, min, max, num_sds=3)`` -- back out a plausible
      standard deviation from a known mean and a known min/max range.  Useful
      when designing a simulation and you have a target distribution shape
      but no variance handy.

    * ``mixed_model(...)`` -- generate a synthetic data frame with the
      structural properties of a mixed-effects model: fixed and random
      effects per predictor, configurable cluster sizes, residual noise,
      interactions, and polynomial terms.  Categorical variables are
      supported.

The function is invaluable for teaching examples (Titanic-style demos),
the power-analysis work py-flexplot is often used for, and for stress-
testing the visualization code with data whose ground truth is known.

Notes on the port
-----------------
The R source has several long-standing typos (e.g. a ``prediction_matrix``
variable that is never assigned, broken test expectations in
``expect_error`` comments).  This Python port follows the same conceptual
design but has those typos fixed.  Behaviour the R docs describe but the R
code does not deliver (e.g. multi-class handling) is implemented here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "estimate_sd",
    "mixed_model",
]


# A *var* entry is either:
#   * a 3-tuple (mean, sd, digits) for a continuous variable, or
#   * a sequence of strings for a categorical variable.
# The final entry in ``vars`` is the cluster id and must be categorical.
VarSpec = Union[Tuple[float, float, int], Sequence[str]]


# ---------------------------------------------------------------------------
# estimate_sd
# ---------------------------------------------------------------------------

def estimate_sd(
    mean: float,
    min_val: float,
    max_val: float,
    num_sds: float = 3,
) -> float:
    """Estimate a standard deviation from a mean and a known min/max range.

    Parameters
    ----------
    mean
        The target mean of the distribution.
    min_val, max_val
        Known extreme values the distribution should comfortably reach.
        (Named ``min_val`` / ``max_val`` rather than ``min`` / ``max`` so
        they don't shadow the Python built-ins.)
    num_sds
        How many standard deviations wide the range should be.  The R
        default is 3 -- i.e. the range covers ``+/- 3 SD`` around the mean.

    Returns
    -------
    float
        Estimated standard deviation.  Larger ``num_sds`` yields a smaller
        SD (more of the range is "inside" the distribution).

    Raises
    ------
    ValueError
        If ``max_val < mean`` or ``min_val > mean`` or ``num_sds <= 0``.
    """
    if max_val < mean:
        raise ValueError(
            f"max ({max_val}) must be >= mean ({mean}) for estimate_sd"
        )
    if min_val > mean:
        raise ValueError(
            f"min ({min_val}) must be <= mean ({mean}) for estimate_sd"
        )
    if num_sds <= 0:
        raise ValueError(f"num_sds must be positive, got {num_sds}")

    # Distance from mean to whichever extreme is closer.
    extent = min(mean - min_val, max_val - mean)
    return extent / num_sds


# ---------------------------------------------------------------------------
# mixed_model
# ---------------------------------------------------------------------------

def mixed_model(
    fixed: Sequence[float],
    random: Sequence[float],
    sigma: float,
    clusters: int,
    n_per: Sequence[float],
    vars: Dict[str, VarSpec],
    interactions: Optional[Dict[str, Sequence]] = None,
    polynomials: Optional[Dict[str, Sequence]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a synthetic mixed-model data frame.

    Conceptual layout, modelled on the R package's design:

    * ``fixed`` and ``random`` are length-``n+1`` vectors where ``fixed[0]``
      is the intercept coefficient and ``random[0]`` is the intercept's
      random-effect SD.  ``fixed[1:]`` and ``random[1:]`` correspond to
      the predictors declared in ``vars`` (excluding the cluster id).
    * Each cluster draws one coefficient vector from
      ``N(fixed, random)`` and replicates it across that cluster's
      observations.  If ``random[j] == 0`` the j-th coefficient is fixed.
    * Each observation's predictor values are independent ``N(0, 1)`` draws
      (or a single replicated draw if the j-th effect is fixed).
    * The standardised response is
      ``y_std = coef_matrix @ predictor_matrix + N(0, sigma * res_cor)``.
    * Columns are then rescaled to their declared (mean, sd, digits) or
      binned into the declared categorical levels.

    Parameters
    ----------
    fixed
        Standardised fixed-effect coefficients, length ``len(vars)``.
        ``fixed[0]`` is the intercept.
    random
        Standardised random-effect SDs, same length as *fixed*.  A value of
        0 means the corresponding coefficient is fixed across clusters.
    sigma
        Proportion of total variance remaining unexplained at the residual
        level (must be in (0, 1)).
    clusters
        Number of clusters.
    n_per
        ``(mean, sd)`` of the per-cluster sample size.  Cluster sizes are
        drawn from a normal truncated at 1.
    vars
        Mapping from variable name to either ``(mean, sd, digits)`` for a
        continuous variable, or a sequence of category labels for a
        categorical variable.  The **last** entry must be the cluster id
        (categorical with ``clusters`` unique levels).
    interactions, polynomials
        Optional dictionaries with keys ``"from"`` (predictor indices into
        the *predictor* slot, not the intercept), ``"to"`` (polynomial target
        index, ignored for pure interactions), and ``"coef"`` (effect
        coefficients).  Indices are 1-based and count only the predictors
        in ``vars`` excluding the cluster id, matching the R package's
        convention.
    seed
        Optional seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        One row per observation, columns in the order of *vars*.  Continuous
        variables are rounded to the requested number of digits; categorical
        variables are discretised by quantile binning.

    Raises
    ------
    ValueError
        On inconsistent *fixed*/*random*/*vars* lengths, sigma out of range,
        cluster-id count mismatch, or ``sum(fixed[1:]**2)**2 >= 1`` (the
        R package's variance-explained guard).
    """
    _check_errors(fixed, random, vars, clusters, sigma)

    rng = np.random.default_rng(seed)

    var_names = list(vars.keys())
    cluster_var = var_names[-1]
    predictor_names = var_names[:-1]
    cluster_levels = list(vars[cluster_var])  # type: ignore[arg-type]
    if len(cluster_levels) != clusters:
        # Defensive; _check_errors should have caught this.
        raise ValueError(
            f"Number of clusters ({clusters}) does not match the cluster "
            f"variable {cluster_var!r} (length {len(cluster_levels)})"
        )

    # Per-cluster sizes (rounded normal, floored at 1).
    mean_n, sd_n = float(n_per[0]), float(n_per[1])
    raw_sizes = rng.normal(mean_n, sd_n, size=clusters)
    sizes = np.maximum(np.rint(raw_sizes).astype(int), 1)
    total_n = int(sizes.sum())

    # Layout:
    #   predictor_matrix shape (total_n, n_pred)  -- one column per predictor slot
    #                                              (no intercept column yet).
    #   coef_matrix      shape (total_n, n_pred + 1)
    #                                              -- extra leading column = intercept.
    n_pred = len(fixed)
    predictor_matrix = np.zeros((total_n, n_pred))
    coef_matrix = np.zeros((total_n, n_pred))

    row = 0
    for c_idx, size in enumerate(sizes):
        cluster_coefs = rng.normal(loc=np.asarray(fixed), scale=np.asarray(random))
        for j in range(n_pred):
            if random[j] == 0:
                # R uses rnorm(1, 0, 1) and replicates -- one constant
                # value per cluster, not per row.
                values = np.full(size, rng.normal(0.0, 1.0))
            else:
                values = rng.normal(0.0, 1.0, size=size)
            predictor_matrix[row:row + size, j] = values
        coef_matrix[row:row + size, :] = cluster_coefs
        row += size

    # Intercept column = all ones (R's `mutate(intercept = 1)`).
    predictor_matrix[:, 0] = 1.0

    # Optional interactions / polynomial terms (additive in the standardized space).
    if interactions is not None:
        predictor_matrix, coef_matrix = _add_interactions(
            predictor_matrix, coef_matrix, interactions
        )
    if polynomials is not None:
        predictor_matrix, coef_matrix = _add_polynomials(
            predictor_matrix, coef_matrix, polynomials
        )

    # y_std = sum(coef_matrix[k] * predictor_matrix[k]) + noise.
    explained = float(np.sum(np.asarray(fixed[1:]) ** 2) ** 2)
    res_cor = sigma * np.sqrt(max(1.0 - explained, 0.0))
    y_std = (coef_matrix * predictor_matrix).sum(axis=1) + rng.normal(0.0, res_cor, size=total_n)

    # Build the output frame:
    #   * first predictor slot -> response (rescaled/binned).
    #   * remaining predictor slots -> raw predictor values, rescaled/binned.
    #   * cluster column -> cluster id, repeated according to ``sizes``.
    out = pd.DataFrame(index=range(total_n))
    out[cluster_var] = np.repeat(cluster_levels, sizes)

    first_spec = vars[predictor_names[0]]
    out[predictor_names[0]] = _apply_spec(y_std, first_spec)

    for j, name in enumerate(predictor_names[1:], start=1):
        spec = vars[name]
        # predictor_matrix column layout: column 0 = intercept (all 1s);
        # columns 1..n_pred-1 = the random-draw predictor values for the
        # slots 1..n_pred-1.  predictor_names[0] is the response (handled
        # above from y_std), so predictor_names[k] for k >= 1 corresponds to
        # predictor_matrix column k.
        col_idx = j
        col = (
            predictor_matrix[:, col_idx]
            if predictor_matrix.shape[1] > col_idx
            else np.zeros(total_n)
        )
        out[name] = _apply_spec(col, spec)

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_continuous_spec(spec: Any) -> bool:
    """Return True if *spec* is a continuous ``(mean, sd, digits)`` tuple.

    A continuous spec is a 3-tuple of non-bool numbers.  Any list, or a
    tuple of non-numeric elements, is treated as categorical levels.
    """
    return (
        isinstance(spec, tuple)
        and len(spec) == 3
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in spec)
    )


def _check_errors(
    fixed: Sequence[float],
    random: Sequence[float],
    vars: Dict[str, VarSpec],
    clusters: int,
    sigma: float,
) -> None:
    fixed_l = len(fixed)
    random_l = len(random)
    vars_l = len(vars)
    if fixed_l != random_l:
        raise ValueError(
            f"fixed and random must have the same length ({fixed_l} vs {random_l})"
        )
    if fixed_l != vars_l - 1:
        raise ValueError(
            f"vars must have length len(fixed) + 1 ({fixed_l + 1}); got {vars_l}"
        )
    cluster_var = list(vars.keys())[-1]
    cluster_levels = vars[cluster_var]
    if _is_continuous_spec(cluster_levels):
        raise ValueError(
            f"Final vars entry (cluster variable {cluster_var!r}) must be categorical "
            f"(list/tuple of levels), got {type(cluster_levels).__name__}"
        )
    if not isinstance(cluster_levels, (list, tuple)):
        raise ValueError(
            f"Final vars entry (cluster variable {cluster_var!r}) must be categorical "
            f"(list/tuple of levels), got {type(cluster_levels).__name__}"
        )
    if len(cluster_levels) != clusters:
        raise ValueError(
            f"Number of clusters ({clusters}) must equal length of cluster "
            f"variable {cluster_var!r} ({len(cluster_levels)})"
        )
    if len(set(cluster_levels)) != len(cluster_levels):
        raise ValueError(
            f"Cluster variable {cluster_var!r} has duplicate levels: {cluster_levels}"
        )
    if not (0.0 < sigma < 1.0):
        raise ValueError(f"sigma must be in (0, 1), got {sigma}")
    explained = float(np.sum(np.asarray(fixed[1:]) ** 2) ** 2)
    if explained >= 1.0:
        raise ValueError(
            f"sum(fixed[1:]**2)**2 = {explained} must be < 1 (standardized "
            "coefficients are too large)"
        )


def _rescale_continuous(x: np.ndarray, spec: Tuple[float, float, int]) -> np.ndarray:
    """Rescale *x* to (mean, sd, digits)."""
    mean, sd, digits = float(spec[0]), float(spec[1]), int(spec[2])
    cur_mean = float(np.mean(x))
    cur_sd = float(np.std(x, ddof=0))
    if cur_sd == 0:
        centred = x - cur_mean
        rescaled = centred + mean
    else:
        rescaled = mean + (x - cur_mean) * (sd / cur_sd)
    if digits >= 0:
        return np.round(rescaled, digits)
    return rescaled


def _apply_spec(x: np.ndarray, spec: VarSpec) -> np.ndarray:
    """Apply a continuous or categorical spec to a 1-D array.

    Mirrors the validation logic in :func:`_check_errors`: a 3-tuple of
    numbers is the continuous ``(mean, sd, digits)`` spec; anything else
    list-or-tuple is categorical levels.
    """
    if _is_continuous_spec(spec):
        return _rescale_continuous(x, spec)  # type: ignore[arg-type]
    if not isinstance(spec, (list, tuple)):
        raise TypeError(
            f"var spec must be a (mean, sd, digits) tuple or a list of levels; "
            f"got {type(spec).__name__}"
        )
    return np.asarray(pd.cut(x, bins=len(spec), labels=list(spec)))


def _interaction_polynomial_checks(spec: Dict[str, Sequence]) -> None:
    if set(spec.keys()) != {"from", "to", "coef"}:
        raise ValueError(
            f"interactions/polynomials dict must have exactly the keys "
            f"'from', 'to', 'coef'; got {sorted(spec.keys())}"
        )
    n_from = len(spec["from"])
    n_to = len(spec["to"])
    n_coef = len(spec["coef"])
    if not (n_from == n_to == n_coef):
        raise ValueError(
            f"interactions/polynomials arrays have inconsistent lengths: "
            f"from={n_from}, to={n_to}, coef={n_coef}"
        )


def _add_interactions(
    predictor_matrix: np.ndarray,
    coef_matrix: np.ndarray,
    interactions: Dict[str, Sequence],
) -> Tuple[np.ndarray, np.ndarray]:
    if not interactions:
        return predictor_matrix, coef_matrix
    _interaction_polynomial_checks(interactions)

    # R's indices are 1-based over the predictor slots.  Our predictor_matrix
    # layout is [intercept, pred_0, pred_1, ...], so slot k lives at column k+1.
    from_idx = [int(i) + 1 for i in interactions["from"]]
    to_idx = [int(i) + 1 for i in interactions["to"]]
    coefs = [float(c) for c in interactions["coef"]]

    for a, b, c in zip(from_idx, to_idx, coefs):
        new_col = predictor_matrix[:, a] * predictor_matrix[:, b]
        predictor_matrix = np.column_stack([predictor_matrix, new_col])
        coef_matrix = np.column_stack([coef_matrix, np.full(predictor_matrix.shape[0], c)])
    return predictor_matrix, coef_matrix


def _add_polynomials(
    predictor_matrix: np.ndarray,
    coef_matrix: np.ndarray,
    polynomials: Dict[str, Sequence],
) -> Tuple[np.ndarray, np.ndarray]:
    if not polynomials:
        return predictor_matrix, coef_matrix
    _interaction_polynomial_checks(polynomials)

    # R's indices are 1-based over the predictor slots.  Our predictor_matrix
    # layout is [intercept, pred_0, pred_1, ...], so slot k lives at column k+1.
    var_idx = [int(i) + 1 for i in polynomials["from"]]
    coefs = [float(c) for c in polynomials["coef"]]
    for col, c in zip(var_idx, coefs):
        new_col = predictor_matrix[:, col] ** 2
        predictor_matrix = np.column_stack([predictor_matrix, new_col])
        coef_matrix = np.column_stack([coef_matrix, np.full(predictor_matrix.shape[0], c)])
    return predictor_matrix, coef_matrix