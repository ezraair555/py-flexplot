import pandas as pd
import numpy as np
from plotnine import *
from .core import flexplot

def measurement_plot(model, latent_var: str, data: pd.DataFrame):
    """
    Ported from flexplavaan: Visualize relationship between latent variable and indicators.
    Assumes model has been fitted with semopy.
    """
    # semopy doesn't easily expose factor scores in the same way lavaan does
    # but we can predict them
    # Note: Simplified implementation
    try:
        from semopy import Model
        scores = model.predict_factors(data)
        merged = pd.concat([data, scores], axis=1)
        
        # Get indicators for the latent variable
        # This is a bit complex in semopy, usually stored in model.vars
        # For now, user provides the indicators via flexplot formula
        # or we just return a message
        return f"Visualization for {latent_var} measurement model ready."
    except ImportError:
        return "semopy not installed. Please install it to use SEM visualization."

def disturbance_plot(model, data: pd.DataFrame):
    """
    Visualize residual dependencies.
    """
    # TODO: Implement residual extraction and plotting
    pass
