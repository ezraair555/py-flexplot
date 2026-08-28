import pandas as pd
import numpy as np
from typing import List, Optional, Union
from plotnine import (
    aes,
    element_text,
    geom_hline,
    geom_point,
    geom_smooth,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradient,
    scale_fill_gradient2,
    theme,
    theme_bw,
    theme_minimal,
)


def hopper_plot(model, **kwargs):
    """
    Ported from flexplavaan: Visualize residuals from the variance/covariance matrix.
    Shows the discrepancy between observed and model-implied correlations.
    """
    try:
        from semopy import Model
    except ImportError as exc:
        raise ImportError("semopy not installed. Please install it to use SEM visualization.") from exc

    # Observed covariance matrix (numpy array in current semopy).
    obs_cov = getattr(model, "mx_cov", None)
    if obs_cov is None:
        raise AttributeError(
            "Model does not expose mx_cov (observed covariance matrix)."
        )
    obs_cov = np.asarray(obs_cov)

    # Model-implied covariance matrix.
    if hasattr(model, "calc_sigma"):
        sigma = model.calc_sigma()
        # In semopy >= 2.3 calc_sigma returns a tuple; take the first array.
        if isinstance(sigma, tuple):
            sigma = sigma[0]
        imp_cov = np.asarray(sigma)
    else:
        raise AttributeError(
            "Model does not have a calc_sigma() method for implied covariance."
        )

    if obs_cov.shape != imp_cov.shape:
        raise ValueError(
            f"Observed and implied covariance matrices have different shapes: "
            f"{obs_cov.shape} vs {imp_cov.shape}"
        )

    # Variable names: semopy stores observed variable names in model.vars['observed'].
    vars_obs = getattr(model, "vars", {}).get("observed")
    if vars_obs is None or len(vars_obs) != obs_cov.shape[0]:
        vars_obs = [f"var{i}" for i in range(obs_cov.shape[0])]

    # Calculate residuals (Observed - Implied)
    res_cov = obs_cov - imp_cov

    # Flatten lower triangle for plotting.
    data_list = []
    for i, row in enumerate(vars_obs):
        for j, col in enumerate(vars_obs):
            if i >= j:  # lower triangle
                data_list.append({
                    "var1": row,
                    "var2": col,
                    "residual": res_cov[i, j],
                })

    df_res = pd.DataFrame(data_list)

    p = (
        ggplot(df_res, aes(x="var1", y="var2", fill="residual"))
        + geom_tile()
        + scale_fill_gradient2(low="red", mid="white", high="blue")
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
        + labs(title="Hopper Plot (Covariance Residuals)")
    )

    return p


def disturbance_plot(model, var1: str, var2: str, data: pd.DataFrame):
    """
    Ported from flexplavaan: Visualize association between two variables
    after removing model-implied fit.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    if data.empty:
        raise ValueError("data must be non-empty for disturbance_plot.")

    for var in (var1, var2):
        if var not in data.columns:
            raise ValueError(f"Variable {var!r} not found in data.")

    if not hasattr(model, "predict"):
        raise AttributeError("Model has no predict method.")

    preds = model.predict(data)
    if not isinstance(preds, pd.DataFrame):
        raise TypeError(
            f"model.predict must return a DataFrame, got {type(preds).__name__}"
        )

    missing_cols = {var1, var2} - set(preds.columns)
    if missing_cols:
        raise ValueError(
            f"Predictions missing columns required for disturbance_plot: {sorted(missing_cols)}"
        )

    # Align predictions to the input data and validate length.
    preds = preds.reindex(data.index)
    valid = preds[[var1, var2]].notna().all(axis=1) & data[[var1, var2]].notna().all(axis=1)
    if not valid.any():
        raise ValueError(
            "No observations remain after aligning predictions with data."
        )

    res1 = data.loc[valid, var1] - preds.loc[valid, var1]
    res2 = data.loc[valid, var2] - preds.loc[valid, var2]

    df_res = pd.DataFrame({
        "res1": res1,
        "res2": res2,
    })

    p = (
        ggplot(df_res, aes(x="res1", y="res2"))
        + geom_point(alpha=0.4)
        + geom_smooth(method="loess", color="blue")
        + geom_hline(yintercept=0, color="red", linetype="dashed")
        + theme_bw()
        + labs(
            x=f"Residual {var1}",
            y=f"Residual {var2}",
            title=f"Disturbance Dependence: {var1} & {var2}",
        )
    )

    return p


def measurement_plot(model, latent_var: str, indicator: str, data: pd.DataFrame):
    """
    Visualize relationship between a latent variable and one of its indicators.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    if data.empty:
        raise ValueError("data must be non-empty for measurement_plot.")
    if indicator not in data.columns:
        raise ValueError(f"Indicator {indicator!r} not found in data.")

    if not hasattr(model, "predict_factors"):
        raise AttributeError("Model has no predict_factors method.")

    factors = model.predict_factors(data)
    if not isinstance(factors, pd.DataFrame):
        raise TypeError(
            f"model.predict_factors must return a DataFrame, got {type(factors).__name__}"
        )
    if latent_var not in factors.columns:
        raise ValueError(
            f"Latent variable {latent_var!r} not found in model factor predictions."
        )

    # Align factor scores with the input data.
    factors = factors.reindex(data.index)
    valid = factors[latent_var].notna() & data[indicator].notna()
    if not valid.any():
        raise ValueError(
            "No observations remain after aligning factor scores with data."
        )

    df_merged = pd.concat(
        [data.loc[valid, [indicator]], factors.loc[valid, [latent_var]]],
        join="inner",
        axis=1,
    )

    p = (
        ggplot(df_merged, aes(x=latent_var, y=indicator))
        + geom_point(alpha=0.5)
        + geom_smooth(method="lm", color="blue")
        + theme_bw()
        + labs(title=f"Measurement Plot: {latent_var} -> {indicator}")
    )

    return p
