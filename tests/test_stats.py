import pandas as pd
import numpy as np
import pytest
import statsmodels.formula.api as smf

from pyflexplot import model_comparison, estimates, p_format, eliminated_columns


def test_model_comparison_basic():
    np.random.seed(0)
    df = pd.DataFrame({
        "y": np.random.normal(size=100),
        "x": np.random.normal(size=100),
        "z": np.random.normal(size=100),
    })
    small = smf.ols("y ~ x", data=df).fit()
    large = smf.ols("y ~ x + z", data=df).fit()

    res, p = model_comparison(small, large)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 2
    assert 0 <= p <= 1


def test_model_comparison_wrong_order_still_valid():
    np.random.seed(0)
    df = pd.DataFrame({
        "y": np.random.normal(size=100),
        "x": np.random.normal(size=100),
        "z": np.random.normal(size=100),
    })
    small = smf.ols("y ~ x", data=df).fit()
    large = smf.ols("y ~ x + z", data=df).fit()

    # Either order should return a valid p-value because we reorder by LLF.
    res1, p1 = model_comparison(small, large)
    res2, p2 = model_comparison(large, small)
    assert p1 == pytest.approx(p2)


def test_model_comparison_missing_attributes():
    class Dummy:
        pass
    with pytest.raises(ValueError, match="missing required attributes"):
        model_comparison(Dummy(), Dummy())


def test_p_format():
    assert p_format(0.05) == ".050"
    assert p_format(0.0005) == "<.001"


def test_eliminated_columns():
    df = pd.DataFrame({
        "keep": [1, 2, 3],
        "drop": [np.nan, np.nan, 3.0],
    })
    out = eliminated_columns(df, threshold=0.5)
    assert "keep" in out.columns
    assert "drop" not in out.columns
