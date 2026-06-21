"""Edge-case tests for py_flexplot (added 2026-06-21, Day 2 Lane 3 of grade recovery).

Targets:
- parse_flexplot_formula (dict-returning parser)
- _validate_data_for_plot (validates variable dict against DataFrame)
"""
import pytest
import pandas as pd
import numpy as np

from pyflexplot.core import parse_flexplot_formula, _validate_data_for_plot


# ---------- parse_flexplot_formula edge cases --------------------------

class TestFormulaEdgeCases:
    """Formula parser edge cases that should raise cleanly, not crash."""

    def test_non_string_formula_raises_type_error(self):
        with pytest.raises(TypeError, match="formula must be a string"):
            parse_flexplot_formula(42)

    def test_missing_tilde_raises(self):
        with pytest.raises(ValueError, match="exactly one '~'"):
            parse_flexplot_formula("y x")

    def test_double_tilde_raises(self):
        with pytest.raises(ValueError, match="exactly one '~'"):
            parse_flexplot_formula("y ~ x ~ z")

    def test_multiple_pipes_raises(self):
        with pytest.raises(ValueError, match="at most one '|'"):
            parse_flexplot_formula("y ~ x | a | b")

    def test_intercept_only_formula_accepted(self):
        """`y ~ 1` should parse cleanly with x=None."""
        result = parse_flexplot_formula("y ~ 1")
        assert result["y"] == "y"
        assert result["x"] is None
        assert result["intercept_only"] is True

    def test_strips_whitespace_from_tokens(self):
        result = parse_flexplot_formula("  y   ~   x  +  z  ")
        assert result["y"] == "y"
        assert result["all_x"] == ["x", "z"]

    def test_single_predictor_with_givens(self):
        result = parse_flexplot_formula("y ~ x | color")
        assert result["y"] == "y"
        assert result["all_x"] == ["x"]
        assert result["given"] == ["color"]

    def test_multiple_givens(self):
        result = parse_flexplot_formula("y ~ x | color + shape")
        assert result["y"] == "y"
        assert result["all_x"] == ["x"]
        assert result["given"] == ["color", "shape"]

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="exactly one '~'"):
            parse_flexplot_formula("")

    def test_none_formula_raises_type_error(self):
        with pytest.raises(TypeError, match="formula must be a string"):
            parse_flexplot_formula(None)

    def test_returns_dict(self):
        """Sanity: parser returns dict with documented keys."""
        result = parse_flexplot_formula("y ~ x")
        assert isinstance(result, dict)
        assert {"y", "x", "given", "all_x", "intercept_only"} <= set(result.keys())


# ---------- _validate_data_for_plot edge cases --------------------------

class TestValidateDataForPlot:
    """Data validator edge cases."""

    def test_non_dataframe_raises_type_error(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            _validate_data_for_plot(
                "y ~ x",
                {"y": [1, 2, 3]},  # a dict, not a DataFrame
                {"y": "y", "x": "x", "given": [], "color": None},
            )

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame({"y": [], "x": []})
        with pytest.raises(ValueError, match="non-empty"):
            _validate_data_for_plot(
                "y ~ x",
                df,
                {"y": "y", "x": "x", "given": [], "color": None},
            )

    def test_missing_predictor_column_raises(self):
        df = pd.DataFrame({"y": [1, 2, 3], "x": [4, 5, 6]})
        with pytest.raises(ValueError, match="missing columns"):
            _validate_data_for_plot(
                "y ~ z",
                df,
                {"y": "y", "x": "z", "given": [], "color": None},
            )

    def test_non_numeric_predictor_raises(self):
        df = pd.DataFrame({"y": [1, 2, 3], "x": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="numeric"):
            _validate_data_for_plot(
                "y ~ x",
                df,
                {"y": "y", "x": "x", "given": [], "color": None},
            )

    def test_clean_data_passes(self):
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [4.0, 5.0, 6.0]})
        # Should not raise
        _validate_data_for_plot(
            "y ~ x",
            df,
            {"y": "y", "x": "x", "given": [], "color": None},
        )


# ---------- Smoke: parse + validate pipeline ---------------------------

class TestParseValidatePipeline:
    """End-to-end: parse a formula, then validate its output dict."""

    def test_parse_then_validate_clean(self):
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": [5.0, 6.0, 7.0, 8.0]})
        parsed = parse_flexplot_formula("y ~ x")
        _validate_data_for_plot(
            "y ~ x",
            df,
            {"y": parsed["y"], "x": parsed["x"], "given": parsed.get("given", []), "color": parsed.get("color")},
        )

    def test_parse_then_validate_missing_column(self):
        df = pd.DataFrame({"y": [1.0, 2.0], "real_x": [3.0, 4.0]})
        parsed = parse_flexplot_formula("y ~ fake_x")
        with pytest.raises(ValueError, match="missing columns"):
            _validate_data_for_plot(
                "y ~ fake_x",
                df,
                {"y": parsed["y"], "x": parsed["x"], "given": parsed.get("given", []), "color": parsed.get("color")},
            )