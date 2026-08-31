"""Auto data-quality diagnostics for regression-style formulas (C: power feature).

Public surface:
- :func:`diagnose`: run a diagnostic suite on a flexplot formula + data.
- :func:`format_summary`: pretty-print a diagnosis dict as a one-paragraph
  terminal/email/log-friendly summary.

Design notes
------------
Diagnostics are designed to surface "why might my fit be off" rather than to
gate-keep model usage. Every diagnostic returns raw test statistics and
p-values alongside a plain-English interpretation; users can drill in
themselves if they want.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# --- Internal helpers ---------------------------------------------------------


def _safe_numeric(s: pd.Series) -> Optional[np.ndarray]:
    """Return ``s`` as a float array, or None if conversion fails."""
    try:
        return s.to_numpy(dtype=float)
    except (ValueError, TypeError):
        return None


def _missingness(data: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    """Per-column missing counts and overall pattern heuristic."""
    per_col = {c: int(data[c].isna().sum()) for c in columns if c in data.columns}
    total_rows = len(data)
    any_missing = sum(per_col.values())
    complete_cases = int(data[list(per_col.keys())].dropna().shape[0])

    # Pattern heuristic: if missingness is concentrated in one column it's
    # likely non-random; spread across all columns suggests MCAR.
    if any_missing == 0:
        pattern = "none"
    elif max(per_col.values(), default=0) > 0.5 * any_missing:
        pattern = "concentrated (likely MNAR/MAR)"
    else:
        pattern = "spread (likely MCAR)"

    return {
        "per_column": per_col,
        "total_missing": any_missing,
        "complete_cases": complete_cases,
        "rows": total_rows,
        "pattern": pattern,
    }


def _outliers(
    y: np.ndarray, X: np.ndarray, threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Cook's distance for a fitted OLS model.

    Returns the count of influential points (Cook's D > 4/n) and their row
    indices in the input data (after any internal NaN filtering).
    """
    import statsmodels.api as sm

    n = max(len(y), 1)
    if threshold is None:
        threshold = 4.0 / n

    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, Xc).fit()
    infl = model.get_influence()
    cooks_d, _ = infl.cooks_distance
    influential_idx = np.where(cooks_d > threshold)[0].tolist()
    return {
        "n_outliers": len(influential_idx),
        "indices": influential_idx,
        "max_cooks_d": float(np.max(cooks_d)),
        "threshold": float(threshold),
        "method": "Cook's distance > 4/n",
    }


def _linearity_test(residuals: np.ndarray, fitted: np.ndarray) -> Dict[str, Any]:
    """Ramsey RESET test for functional form misspecification.

    Augmented regression: add fitted^2 and fitted^3 to the design and test
    whether the added terms are jointly zero. A small p-value indicates
    non-linearity.
    """
    import statsmodels.api as sm

    X_aug = np.column_stack([fitted, fitted ** 2, fitted ** 3])
    Xc = sm.add_constant(X_aug, has_constant="add")
    model = sm.OLS(residuals, Xc).fit()
    # The F-test is for whether the augmented terms have zero coefficients
    # (excluding the constant).
    r_matrix = np.zeros((3, Xc.shape[1]))
    r_matrix[0, 1] = 1  # fitted^1
    r_matrix[1, 2] = 1  # fitted^2
    r_matrix[2, 3] = 1  # fitted^3
    try:
        test = model.f_test(r_matrix)
        statistic = float(np.squeeze(test.fvalue))
        p_value = float(np.squeeze(test.pvalue))
    except Exception:
        statistic, p_value = float("nan"), float("nan")

    return {
        "test": "Ramsey RESET",
        "statistic": statistic,
        "p_value": p_value,
        "reject_linearity": bool(p_value < 0.05) if not np.isnan(p_value) else None,
        "interpretation": (
            "Reject linearity at alpha=0.05; functional form may be misspecified."
            if (p_value < 0.05 if not np.isnan(p_value) else False)
            else "Fail to reject linearity at alpha=0.05."
        ),
    }


def _heteroscedasticity_test(
    residuals: np.ndarray, fitted: np.ndarray, X_with_const: np.ndarray
) -> Dict[str, Any]:
    """Breusch-Pagan test for non-constant error variance.

    Returns the LM statistic, p-value, and an interpretation.
    """
    from statsmodels.stats.diagnostic import het_breuschpagan

    try:
        lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, X_with_const)
    except Exception:
        return {
            "test": "Breusch-Pagan",
            "statistic": float("nan"),
            "p_value": float("nan"),
            "reject_homoscedasticity": None,
            "interpretation": "Test could not be computed.",
        }
    return {
        "test": "Breusch-Pagan",
        "statistic": float(lm),
        "p_value": float(lm_pvalue),
        "reject_homoscedasticity": bool(lm_pvalue < 0.05),
        "interpretation": (
            "Reject homoscedasticity at alpha=0.05; variance is non-constant."
            if lm_pvalue < 0.05
            else "Fail to reject homoscedasticity at alpha=0.05."
        ),
    }


# --- Public API ---------------------------------------------------------------


def diagnose(
    formula: str,
    data: pd.DataFrame,
    verbose: bool = True,
    outlier_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a data-quality diagnostic on a flexplot formula + data.

    Parameters
    ----------
    formula : str
        Flexplot formula of the form ``y ~ x1 + x2 [+ ...]``. Only the
        outcome and predictors after ``~`` are used.
    data : pd.DataFrame
        Non-empty data frame holding the referenced columns.
    verbose : bool, default True
        If True, prints a one-paragraph summary to stdout.
    outlier_threshold : float, default 4.0 / n
        Cook's distance cutoff. Default is the conventional ``4/n`` value
        (applied automatically as ``4.0 / n_complete``). Pass an explicit
        float to override.

    Returns
    -------
    dict
        Structured diagnostics with keys:

        - ``n_obs`` (int), ``n_complete`` (int), ``columns`` (list[str])
        - ``missing`` (dict): per-column counts and pattern heuristic
        - ``outliers`` (dict): Cook's D count and threshold
        - ``linearity`` (dict): Ramsey RESET test
        - ``heteroscedasticity`` (dict): Breusch-Pagan test

    Raises
    ------
    ValueError
        If the formula has no outcome or no predictors.

    Examples
    --------
    Verbose (prints to stdout, returns the dict):

    >>> import pandas as pd
    >>> import numpy as np
    >>> from pyflexplot.quality import diagnose
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "y": rng.normal(size=200),
    ...     "x": rng.normal(size=200),
    ... })
    >>> diag = diagnose("y ~ x", data=df)  # doctest: +SKIP

    Quiet (returns the dict without printing):

    >>> diag = diagnose("y ~ x", data=df, verbose=False)
    >>> diag["linearity"]["reject_linearity"]
    False

    Notes
    -----
    Designed to surface *why* a fit might be off, not to gate-keep model
    usage. All test statistics and p-values are returned alongside the
    plain-English interpretation so users can drill in themselves. The
    pattern heuristic (`` none`` / ``concentrated`` / ``spread``) is a
    rough first cut; for missing-data formal tests, see ``statsmodels``.
    """
    from .core import parse_flexplot_formula

    variables = parse_flexplot_formula(formula)
    y_name = variables["y"]
    x_names = [v for v in variables["all_x"] if v]
    if not x_names:
        raise ValueError(
            f"diagnose() requires a formula with at least one predictor; got {formula!r}."
        )

    # Drop any predictors that aren't numeric — non-numeric predictors
    # (categorical color, given groups) belong to the formula but not the
    # regression design matrix.
    numeric_x_names = [
        c for c in x_names if c in data.columns
        and pd.api.types.is_numeric_dtype(data[c])
    ]
    if not numeric_x_names:
        raise ValueError(
            f"diagnose() found no numeric predictors in formula {formula!r}; "
            f"predictors={x_names}."
        )

    columns = [y_name] + numeric_x_names
    # Strip any column not actually present (parse_flexplot_formula is permissive).
    columns = [c for c in columns if c in data.columns]
    if len(columns) < 2:
        raise ValueError(
            f"diagnose() needs at least one predictor present in data; got columns={columns}."
        )

    missing_summary = _missingness(data, columns)

    # Fit on the complete-case subset.
    complete = data[[y_name] + numeric_x_names].dropna()
    n_complete = len(complete)
    if n_complete < 4:
        out: Dict[str, Any] = {
            "n_obs": int(len(data)),
            "n_complete": n_complete,
            "columns": columns,
            "missing": missing_summary,
            "outliers": {
                "n_outliers": 0, "indices": [], "max_cooks_d": None,
                "threshold": (
                outlier_threshold if outlier_threshold is not None
                else 4.0 / max(n_complete, 1)
            ),
                "method": "Cook's distance > 4/n",
            },
            "linearity": {
                "test": "Ramsey RESET",
                "statistic": None, "p_value": None, "reject_linearity": None,
                "interpretation": "Not enough complete cases to run the test.",
            },
            "heteroscedasticity": {
                "test": "Breusch-Pagan",
                "statistic": None, "p_value": None, "reject_homoscedasticity": None,
                "interpretation": "Not enough complete cases to run the test.",
            },
        }
        if verbose:
            print(format_summary(out))
        return out

    y = _safe_numeric(complete[y_name])
    X = np.column_stack([_safe_numeric(complete[c]) for c in numeric_x_names])

    # Re-fit on the complete-case subset to get residuals/fitted.
    import statsmodels.api as sm

    Xc = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(y, Xc).fit()
    residuals = fit.resid
    fitted = fit.fittedvalues

    outlier_summary = _outliers(
        y, X,
        threshold=(
            outlier_threshold
            if outlier_threshold is not None
            else 4.0 / max(n_complete, 1)
        ),
    )
    linearity_summary = _linearity_test(residuals, fitted)
    hetero_summary = _heteroscedasticity_test(residuals, fitted, Xc)

    out = {
        "n_obs": int(len(data)),
        "n_complete": int(n_complete),
        "columns": columns,
        "missing": missing_summary,
        "outliers": outlier_summary,
        "linearity": linearity_summary,
        "heteroscedasticity": hetero_summary,
        "_r_squared": float(fit.rsquared),
    }

    if verbose:
        print(format_summary(out))

    return out


def format_summary(diag: Dict[str, Any]) -> str:
    """Format a diagnosis dict as a one-paragraph human-readable summary.

    Suitable for terminal output, log lines, or email bodies.
    """
    n_obs = diag.get("n_obs", "?")
    n_complete = diag.get("n_complete", "?")
    columns = diag.get("columns", [])
    col_str = ", ".join(columns)

    missing = diag.get("missing", {})
    total_missing = missing.get("total_missing", 0)
    pattern = missing.get("pattern", "unknown")

    outliers = diag.get("outliers", {})
    n_outliers = outliers.get("n_outliers", 0)
    max_cooks = outliers.get("max_cooks_d")

    linearity = diag.get("linearity", {})
    lin_interp = linearity.get("interpretation", "n/a")

    hetero = diag.get("heteroscedasticity", {})
    het_interp = hetero.get("interpretation", "n/a")

    r2 = diag.get("_r_squared")

    lines = [
        f"Diagnostic for {col_str} (n={n_obs}, complete cases={n_complete})",
        f"  Missingness: {total_missing} missing values total; pattern = {pattern}.",
    ]
    if max_cooks is not None:
        lines.append(
            f"  Outliers: {n_outliers} influential points (max Cook's D = {max_cooks:.3f})."
        )
    else:
        lines.append(f"  Outliers: {n_outliers} influential points.")
    lines.append(f"  Linearity: {lin_interp}")
    lines.append(f"  Heteroscedasticity: {het_interp}")
    if r2 is not None:
        lines.append(f"  R-squared (OLS reference): {r2:.3f}")
    return "\n".join(lines)