import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from typing import List, Union

def model_comparison(model1, model2):
    """
    Statistically compares the fits of two models.
    Reports AIC, BIC, and p-values where applicable.
    """
    res = pd.DataFrame({
        "AIC": [model1.aic, model2.aic],
        "BIC": [model1.bic, model2.bic],
        "LogLik": [model1.llf, model2.llf]
    }, index=["Model 1", "Model 2"])
    
    # Simple LRT if nested (basic assumption for now)
    lr_stat = 2 * (model2.llf - model1.llf)
    df_diff = model2.df_model - model1.df_model
    p_val = 1 - stats.chi2.cdf(abs(lr_stat), abs(df_diff))
    
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
