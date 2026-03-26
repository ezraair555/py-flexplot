import pandas as pd
import numpy as np
from plotnine import *
from typing import Optional, List, Union

def hopper_plot(model, **kwargs):
    """
    Ported from flexplavaan: Visualize residuals from the variance/covariance matrix.
    Shows the discrepancy between observed and model-implied correlations.
    """
    try:
        from semopy import Model
        # Get covariance matrices
        obs_cov = model.mx_cov
        imp_cov = model.mx_exp_cov
        
        # Calculate residuals (Observed - Implied)
        res_cov = obs_cov - imp_cov
        
        # Flatten and convert to correlation scale for easier viewing
        # Note: This is a simplified version of the R hopper plot logic
        vars = list(obs_cov.index)
        data_list = []
        for i, row in enumerate(vars):
            for j, col in enumerate(vars):
                if i >= j: # lower triangle
                    data_list.append({
                        "var1": row,
                        "var2": col,
                        "residual": res_cov.iloc[i, j]
                    })
        
        df_res = pd.DataFrame(data_list)
        
        p = (ggplot(df_res, aes(x="var1", y="var2", fill="residual"))
             + geom_tile()
             + scale_fill_gradient2(low="red", mid="white", high="blue")
             + theme_minimal()
             + theme(axis_text_x=element_text(rotation=45, hjust=1))
             + labs(title="Hopper Plot (Covariance Residuals)"))
        
        return p
    except ImportError:
        return "semopy not installed. Please install it to use SEM visualization."

def disturbance_plot(model, var1: str, var2: str, data: pd.DataFrame):
    """
    Ported from flexplavaan: Visualize association between two variables 
    after removing model-implied fit.
    """
    # 1. Get residuals for var1 and var2
    # In SEM, this involves subtracting the predicted value from the observed
    # semopy can predict observed variables
    try:
        preds = model.predict(data)
        res1 = data[var1] - preds[var1]
        res2 = data[var2] - preds[var2]
        
        df_res = pd.DataFrame({
            "res1": res1,
            "res2": res2
        })
        
        p = (ggplot(df_res, aes(x="res1", y="res2"))
             + geom_point(alpha=0.4)
             + geom_smooth(method="loess", color="blue")
             + geom_hline(yintercept=0, color="red", linetype="dashed")
             + theme_bw()
             + labs(x=f"Residual {var1}", y=f"Residual {var2}", 
                    title=f"Disturbance Dependence: {var1} & {var2}"))
        
        return p
    except Exception as e:
        return f"Error creating disturbance plot: {e}"

def measurement_plot(model, latent_var: str, indicator: str, data: pd.DataFrame):
    """
    Visualize relationship between a latent variable and one of its indicators.
    """
    try:
        # Predict factor scores
        factors = model.predict_factors(data)
        df_merged = pd.concat([data[[indicator]], factors[[latent_var]]], axis=1)
        
        p = (ggplot(df_merged, aes(x=latent_var, y=indicator))
             + geom_point(alpha=0.5)
             + geom_smooth(method="lm", color="blue")
             + theme_bw()
             + labs(title=f"Measurement Plot: {latent_var} -> {indicator}"))
        
        return p
    except Exception as e:
        return f"Error creating measurement plot: {e}"
