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

    Returns a tuple ``(DataFrame, p_value)`` where the DataFrame carries
    per-model AIC, BIC, LogLik, R-squared, adjusted R-squared, and Bayes
    factor (computed from BIC via the Kass & Raftery 1995 approximation).
    The second element is the p-value from the likelihood-ratio test.

    The Bayes factor is attached to the more likely model (BIC-wise):
    the model with the lower BIC gets a BF ≥ 1 in its row, the other
    model gets 1/BF. This mirrors R's ``flexplot::model.comparison()``
    behavior.

    The LRT always subtracts the smaller log-likelihood from the larger
    one and uses the corresponding positive degrees-of-freedom difference.
    """
    required = ("aic", "bic", "llf", "df_model")
    _check_statsmodels_attrs(model1, required)
    _check_statsmodels_attrs(model2, required)

    # Bayes factor for model1 over model2 (Kass & Raftery 1995 approximation
    # from BIC): BF_{1,2} = exp((BIC_2 - BIC_1) / 2). Values > 1 favor model1.
    bf_raw = float(np.exp((model2.bic - model1.bic) / 2.0))

    # Attach the larger BF to the model with the lower BIC. The convention
    # here matches R's model_comparison_table(): the better model gets
    # BF >= 1; the worse model gets 1/BF.
    if model1.bic <= model2.bic:
        bf_col = [bf_raw, 1.0 / bf_raw]
    else:
        bf_col = [1.0 / bf_raw, bf_raw]

    res = pd.DataFrame(
        {
            "AIC": [model1.aic, model2.aic],
            "BIC": [model1.bic, model2.bic],
            "LogLik": [model1.llf, model2.llf],
        },
        index=["Model 1", "Model 2"],
    )

    # R-squared and adjusted R-squared columns when available (OLS / GLM).
    extras = {}
    if hasattr(model1, "rsquared") and hasattr(model2, "rsquared"):
        extras["R.squared"] = [float(model1.rsquared), float(model2.rsquared)]
    if hasattr(model1, "rsquared_adj") and hasattr(model2, "rsquared_adj"):
        extras["Adj.R.squared"] = [
            float(model1.rsquared_adj),
            float(model2.rsquared_adj),
        ]
    extras["BayesFactor"] = bf_col
    if extras:
        res = pd.concat([res, pd.DataFrame(extras, index=res.index)], axis=1)

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
    Extract a coefficient summary from a fitted model.

    NOTE: This is a thin pass-through to ``model.summary()``.  The R
    ``fifer`` package computes Cohen's d, eta-squared, and other effect
    sizes that this Python port does not yet implement.  Calling
    ``estimates()`` currently returns the model's statsmodels summary
    object; it does NOT compute effect sizes.

    Status: experimental, not yet a real effect-size reporter.  See the
    py-flexplot roadmap for the planned implementation.
    """
    return model.summary()


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
