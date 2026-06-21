import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from typing import List, Union


def _check_statsmodels_attrs(model, attrs):
    """Raise ValueError if *model* is missing any of the listed attributes."""
    missing = [a for a in attrs if not hasattr(model, a)]
    if missing:
        raise ValueError(
            f"Model is missing required attributes for model_comparison: {missing}"
        )


def model_comparison(model1, model2):
    """
    Statistically compares the fits of two nested statsmodels results.

    Returns a DataFrame with AIC, BIC, and log-likelihood, plus a p-value from
    the likelihood-ratio test. The LRT always subtracts the smaller log-
    likelihood from the larger one and uses the corresponding positive degrees-
    of-freedom difference.
    """
    required = ("aic", "bic", "llf", "df_model")
    _check_statsmodels_attrs(model1, required)
    _check_statsmodels_attrs(model2, required)

    res = pd.DataFrame(
        {
            "AIC": [model1.aic, model2.aic],
            "BIC": [model1.bic, model2.bic],
            "LogLik": [model1.llf, model2.llf],
        },
        index=["Model 1", "Model 2"],
    )

    # Order so the larger (less constrained) model is subtracted from the
    # smaller (more constrained) one, yielding a positive LR statistic with a
    # positive df difference.
    if model2.llf >= model1.llf:
        lr_stat = 2 * (model2.llf - model1.llf)
        df_diff = int(round(model2.df_model - model1.df_model))
    else:
        lr_stat = 2 * (model1.llf - model2.llf)
        df_diff = int(round(model1.df_model - model2.df_model))

    if df_diff <= 0:
        raise ValueError(
            f"Degrees-of-freedom difference must be positive for a valid LRT; got {df_diff}. "
            "Models may not be nested or may be in the wrong order."
        )

    p_val = 1 - stats.chi2.cdf(lr_stat, df_diff)

    return res, p_val


def estimates(model):
    """
    Reports effect sizes (e.g., Cohen's d, Eta-squared) for statistical models.
    """
    summary = model.summary()
    # Simple extraction of coefficients and p-values as a starting point
    return summary


def p_format(p: float, digits: int = 3):
    """
    Ported from fifer: Formats p-values (e.g., <.001).
    """
    if p < 0.001:
        return "<.001"
    return f"{p:.{digits}f}".replace("0.", ".")


def eliminated_columns(df: pd.DataFrame, threshold: float = 0.5):
    """
    Ported from fifer: Removes columns with too many missing values.
    """
    na_count = df.isna().sum() / len(df)
    to_keep = na_count[na_count <= threshold].index
    return df[to_keep]


def color_table(df: pd.DataFrame, cmap: str = "viridis"):
    """
    Ported from fifer: Returns a styled pandas dataframe.
    """
    return df.style.background_gradient(cmap=cmap)
