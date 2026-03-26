import pandas as pd
import numpy as np
from plotnine import *
import patsy
from typing import Union, List, Optional
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

def parse_flexplot_formula(formula: str):
    """
    Parses a flexplot formula of the form:
    outcome ~ predictor1 + predictor2 | given1 + given2
    """
    if "|" in formula:
        main_part, given_part = formula.split("|")
    else:
        main_part = formula
        given_part = None
    
    y_name = main_part.split("~")[0].strip()
    x_formula = main_part.split("~")[1].strip()
    
    # Handle multiple predictors on X (predictor1 + predictor2)
    x_parts = [p.strip() for p in x_formula.split("+")]
    x_name = x_parts[0]
    color_name = x_parts[1] if len(x_parts) > 1 else None
    
    given_names = [g.strip() for g in given_part.split("+")] if given_part else []
    
    return {
        "y": y_name,
        "x": x_name,
        "color": color_name,
        "given": given_names,
        "all_x": x_parts
    }

def flexplot(formula: str, data: pd.DataFrame, method: str = "auto", **kwargs):
    """
    Intelligent multivariate graphics via formulas.
    """
    variables = parse_flexplot_formula(formula)
    y = variables["y"]
    x = variables["x"]
    color = variables["color"]
    given = variables["given"]
    
    # Determine variable types
    is_y_numeric = pd.api.types.is_numeric_dtype(data[y])
    is_x_numeric = pd.api.types.is_numeric_dtype(data[x])
    
    # Base plot
    p = ggplot(data, aes(x=x, y=y))
    
    # Determine plot type
    if is_y_numeric and is_x_numeric:
        p += geom_point(alpha=0.5)
        if method == "auto" or method == "lm":
            p += geom_smooth(method="lm", color="blue")
        elif method == "loess":
            p += geom_smooth(method="loess", color="blue")
            
    elif is_y_numeric and not is_x_numeric:
        p += geom_jitter(width=0.2, alpha=0.5)
        p += stat_summary(fun_data="mean_cl_boot", color="red", size=1)
        
    elif not is_y_numeric and is_x_numeric:
        p += geom_point(alpha=0.3)
        p += geom_smooth(method="glm", method_args={'family': 'binomial'})
        
    else:
        p += geom_jitter(width=0.2, height=0.2, alpha=0.5)
        
    if color:
        p += aes(color=color, group=color)
        
    if len(given) == 1:
        p += facet_wrap(f"~{given[0]}")
    elif len(given) >= 2:
        p += facet_grid(f"{given[1]} ~ {given[0]}")
        
    p += theme_bw()
    return p

def visualize(model, **kwargs):
    """
    Provides a visual representation of a fitted statistical object.
    Supports statsmodels and sklearn models.
    """
    # TODO: Implement prediction-based visualization
    return f"Visualization for {type(model).__name__} not yet implemented."

def compare_fits(formula: str, data: pd.DataFrame, model1, model2, **kwargs):
    """
    Visually compare the fit of two different models.
    """
    # TODO: Implement comparison plot
    return "Comparison plot not yet implemented."

def added_plot(formula: str, data: pd.DataFrame, **kwargs):
    """
    Generates an added variable plot (partial regression plot).
    """
    variables = parse_flexplot_formula(formula)
    y_var = variables["y"]
    x_var = variables["x"]
    other_vars = [v for v in variables["all_x"] if v != x_var]
    
    if not other_vars:
        return flexplot(formula, data, **kwargs)
    
    # Residuals of Y on other vars
    y_res_model = OLS.from_formula(f"{y_var} ~ {' + '.join(other_vars)}", data=data).fit()
    y_residuals = y_res_model.resid
    
    # Residuals of X on other vars
    x_res_model = OLS.from_formula(f"{x_var} ~ {' + '.join(other_vars)}", data=data).fit()
    x_residuals = x_res_model.resid
    
    res_df = pd.DataFrame({
        f"res_{y_var}": y_residuals,
        f"res_{x_var}": x_residuals
    })
    
    p = (ggplot(res_df, aes(x=f"res_{x_var}", y=f"res_{y_var}"))
         + geom_point(alpha=0.5)
         + geom_smooth(method="lm", color="blue")
         + labs(x=f"{x_var} | others", y=f"{y_var} | others", title="Added Variable Plot")
         + theme_bw())
    
    return p
