"""Tests for the auto data-quality diagnostic (C: power feature)."""

import numpy as np
import pandas as pd
import pytest

from pyflexplot.quality import diagnose, format_summary


# --- Helpers ------------------------------------------------------------------


def _make_clean_linear(n: int = 200, seed: int = 0):
    """Clean linear relationship, homoscedastic, no outliers."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x": x})


def _make_with_outliers(n: int = 200, seed: int = 0, n_outliers: int = 5):
    """Clean linear data with a few large-residual outliers added."""
    df = _make_clean_linear(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(df.index, size=n_outliers, replace=False)
    df.loc[idx, "y"] = df.loc[idx, "y"] + 20.0
    return df


def _make_heteroscedastic(n: int = 500, seed: int = 0):
    """Exponential variance grows strongly with x — Breusch-Pagan rejects."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    sigma = np.exp(x - 1.0)
    y = x + rng.normal(scale=sigma)
    return pd.DataFrame({"y": y, "x": x})


def _make_nonlinear(n: int = 300, seed: int = 0):
    """Quadratic relationship — Ramsey RESET should reject."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = x ** 2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x": x})


# --- Input validation ---------------------------------------------------------


def test_diagnose_requires_predictor():
    df = _make_clean_linear()
    with pytest.raises(ValueError, match="at least one predictor"):
        diagnose("y ~ 1", data=df, verbose=False)


def test_diagnose_missing_columns_in_data_raises():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="no numeric predictors"):
        diagnose("y ~ nonexistent", data=df, verbose=False)


# --- Missingness summary ------------------------------------------------------


def test_diagnose_missingness_zero_when_complete():
    df = _make_clean_linear()
    diag = diagnose("y ~ x", data=df, verbose=False)
    assert diag["missing"]["total_missing"] == 0
    assert diag["missing"]["pattern"] == "none"
    assert diag["missing"]["complete_cases"] == len(df)


def test_diagnose_missingness_per_column():
    df = _make_clean_linear(n=100, seed=0)
    df.loc[df.index[:10], "x"] = np.nan
    df.loc[df.index[20:25], "y"] = np.nan
    diag = diagnose("y ~ x", data=df, verbose=False)
    assert diag["missing"]["per_column"]["x"] == 10
    assert diag["missing"]["per_column"]["y"] == 5
    assert diag["missing"]["total_missing"] == 15


def test_diagnose_missingness_pattern_concentrated():
    """All missingness in one column => pattern=concentrated."""
    df = _make_clean_linear(n=100, seed=0)
    df.loc[df.index[:50], "x"] = np.nan
    diag = diagnose("y ~ x", data=df, verbose=False)
    assert diag["missing"]["pattern"] == "concentrated (likely MNAR/MAR)"


def test_diagnose_missingness_pattern_spread():
    """Missingness spread across columns => pattern=spread."""
    df = _make_clean_linear(n=100, seed=0)
    # 5 missing in each column, balanced
    df.loc[df.index[:5], "x"] = np.nan
    df.loc[df.index[5:10], "y"] = np.nan
    diag = diagnose("y ~ x", data=df, verbose=False)
    assert diag["missing"]["pattern"] == "spread (likely MCAR)"


# --- Outliers (Cook's distance) ----------------------------------------------


def test_diagnose_outliers_clean_data_few():
    df = _make_clean_linear(n=200)
    diag = diagnose("y ~ x", data=df, verbose=False, outlier_threshold=1.0)
    # Cook's D > 1.0 is a very strict cutoff; clean linear data should
    # have at most a handful of points above it (often zero).
    assert diag["outliers"]["n_outliers"] <= 3


def test_diagnose_outliers_detects_added_outliers():
    df = _make_with_outliers(n=200, n_outliers=5)
    diag = diagnose("y ~ x", data=df, verbose=False, outlier_threshold=0.1)
    assert diag["outliers"]["n_outliers"] >= 4
    assert diag["outliers"]["max_cooks_d"] > 0.1


def test_diagnose_outliers_returns_indices():
    df = _make_with_outliers(n=200, n_outliers=3)
    diag = diagnose("y ~ x", data=df, verbose=False, outlier_threshold=0.1)
    assert isinstance(diag["outliers"]["indices"], list)
    assert len(diag["outliers"]["indices"]) >= 1


# --- Linearity (Ramsey RESET) -------------------------------------------------


def test_diagnose_linearity_passes_for_linear_data():
    df = _make_clean_linear(n=300)
    diag = diagnose("y ~ x", data=df, verbose=False)
    assert diag["linearity"]["test"] == "Ramsey RESET"
    assert diag["linearity"]["reject_linearity"] is False


def test_diagnose_linearity_detects_quadratic():
    df = _make_nonlinear(n=300)
    diag = diagnose("y ~ x", data=df, verbose=False)
    assert diag["linearity"]["reject_linearity"] is True


# --- Heteroscedasticity (Breusch-Pagan) ---------------------------------------


def test_diagnose_heteroscedasticity_passes_for_homoscedastic():
    df = _make_clean_linear(n=400)
    diag = diagnose("y ~ x", data=df, verbose=False)
    assert diag["heteroscedasticity"]["test"] == "Breusch-Pagan"
    assert diag["heteroscedasticity"]["reject_homoscedasticity"] is False


def test_diagnose_heteroscedasticity_detects_nonconstant_variance():
    df = _make_heteroscedastic(n=800)
    diag = diagnose("y ~ x", data=df, verbose=False)
    assert diag["heteroscedasticity"]["reject_homoscedasticity"] is True


# --- Verbose output -----------------------------------------------------------


def test_diagnose_verbose_prints_summary(capsys):
    df = _make_clean_linear(n=100)
    diagnose("y ~ x", data=df, verbose=True)
    captured = capsys.readouterr()
    assert "Diagnostic for" in captured.out
    assert "Missingness" in captured.out
    assert "Outliers" in captured.out
    assert "Linearity" in captured.out
    assert "Heteroscedasticity" in captured.out


def test_diagnose_quiet_does_not_print(capsys):
    df = _make_clean_linear(n=100)
    diagnose("y ~ x", data=df, verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ""


# --- format_summary -----------------------------------------------------------


def test_format_summary_returns_multiline_string():
    df = _make_clean_linear(n=50)
    diag = diagnose("y ~ x", data=df, verbose=False)
    summary = format_summary(diag)
    assert isinstance(summary, str)
    assert "\n" in summary
    assert "Diagnostic for" in summary
    assert "Missingness" in summary


def test_format_summary_handles_minimal_dict():
    summary = format_summary({"n_obs": 100, "n_complete": 100, "columns": ["y", "x"]})
    assert "n=100" in summary
    assert "complete cases=100" in summary


# --- Multi-predictor formulas -------------------------------------------------


def test_diagnose_with_two_predictors():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = x1 + 2.0 * x2 + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    diag = diagnose("y ~ x1 + x2", data=df, verbose=False)
    assert diag["n_complete"] == n
    assert set(diag["columns"]) == {"y", "x1", "x2"}
    assert diag["linearity"]["reject_linearity"] is False


# --- Color/group variable ignored --------------------------------------------


def test_diagnose_formula_with_color_group():
    """Color and given variables in the formula should not break the diagnostic."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    z = rng.choice(["a", "b"], size=n)
    y = x + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    diag = diagnose("y ~ x + z | z", data=df, verbose=False)
    # Should still work; "z" is non-numeric so will be skipped from the design
    # matrix but the diagnostic should not crash.
    assert diag["n_complete"] > 0