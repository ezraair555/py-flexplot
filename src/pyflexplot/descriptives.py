"""
descriptives: Descriptive-statistics visualizations.

A port of R's ``fifer::meansplot()``: takes a numeric ``y ~ group`` formula
and shows the mean (with an error bar) per group, optionally connecting
the means with a line.

Usage::

    from pyflexplot.descriptives import meansplot

    p = meansplot("weight ~ diet", data=df)
    p = meansplot("weight ~ diet", data=df, error="sd")     # SD instead of SE
    p = meansplot("weight ~ diet", data=df, error="ci")      # 95% CI on the mean
    p = meansplot("weight ~ diet", data=df, connect=True)    # line connecting means

This module is intentionally separate from ``core.py`` so the descriptive-
stats surface doesn't bloat the formula-dispatch logic in ``flexplot()``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotnine
from plotnine import (
    aes,
    geom_errorbar,
    geom_line,
    geom_point,
    ggplot,
    labs,
    theme_bw,
)

from .core import parse_flexplot_formula, _validate_data_for_plot


_VALID_ERROR = {"se", "sd", "ci", "range", "iqr", "no"}


def meansplot(
    formula: str,
    data: pd.DataFrame,
    error: str = "se",
    level: float = 0.95,
    connect: bool = True,
):
    """Plot the mean of y per level of x with an error bar.

    Parameters
    ----------
    formula : str
        Formula of the form ``y ~ group``. ``group`` may be a single
        categorical variable (string/object) or a numeric variable with
        few enough unique values to be treated as discrete.
    data : pd.DataFrame
        The dataset.
    error : {"se", "sd", "ci", "range", "iqr", "no"}, default "se"
        Kind of error bar to draw around each mean:
        - ``"se"``: standard error of the mean (default).
        - ``"sd"``: standard deviation.
        - ``"ci"``: ``level`` confidence interval on the mean.
        - ``"range"``: min-max range.
        - ``"iqr"``: Q1-Q3 IQR.
        - ``"no"``: no error bar.
    level : float, default 0.95
        Coverage probability for ``error="ci"``. Ignored otherwise.
    connect : bool, default True
        If ``True``, draw a line connecting the per-group means (useful
        for ordinal predictors where the trend matters).
    """
    if error not in _VALID_ERROR:
        raise ValueError(
            f"error must be one of {sorted(_VALID_ERROR)}; got {error!r}."
        )

    variables = parse_flexplot_formula(formula)
    _validate_data_for_plot(formula, data, variables)
    y = variables["y"]
    x = variables["x"]
    if variables.get("color"):
        raise ValueError(
            f"meansplot does not support a `color` term; got formula "
            f"{formula!r} with color={variables['color']!r}."
        )
    if variables.get("given"):
        raise ValueError(
            f"meansplot does not support `given` terms (faceting); got "
            f"formula {formula!r} with given={variables['given']!r}."
        )

    if not pd.api.types.is_numeric_dtype(data[y]):
        raise ValueError(
            f"meansplot requires a numeric y; got {y!r} with dtype "
            f"{data[y].dtype}."
        )

    # Group by x and compute summary statistics.
    grouped = data.groupby(x, observed=True, sort=True)[y]
    summary = grouped.agg(["count", "mean", "std"]).reset_index()

    if error == "se":
        summary["__lower"] = summary["mean"] - summary["std"] / np.sqrt(summary["count"])
        summary["__upper"] = summary["mean"] + summary["std"] / np.sqrt(summary["count"])
    elif error == "sd":
        summary["__lower"] = summary["mean"] - summary["std"]
        summary["__upper"] = summary["mean"] + summary["std"]
    elif error == "ci":
        from scipy import stats as _scipy_stats
        # 95% CI on the mean using a t-distribution (n-1 df).
        se = summary["std"] / np.sqrt(summary["count"])
        df = summary["count"] - 1
        t_crit = _scipy_stats.t.ppf(0.5 + level / 2, df)
        summary["__lower"] = summary["mean"] - t_crit * se
        summary["__upper"] = summary["mean"] + t_crit * se
    elif error == "range":
        summary["__lower"] = grouped.min().to_numpy()
        summary["__upper"] = grouped.max().to_numpy()
    elif error == "iqr":
        summary["__lower"] = grouped.quantile(0.25).to_numpy()
        summary["__upper"] = grouped.quantile(0.75).to_numpy()
    elif error == "no":
        # No error bars; we'll skip the geom_errorbar layer below.
        summary["__lower"] = summary["mean"]
        summary["__upper"] = summary["mean"]

    # If x is numeric with few unique values, coerce to categorical so
    # plotnine treats it as discrete levels (matches R's behavior).
    plot_df = summary.copy()
    if pd.api.types.is_numeric_dtype(plot_df[x]):
        plot_df[x] = plot_df[x].astype(str)

    p = (
        ggplot(plot_df, aes(x=x, y="mean"))
        + geom_point(size=3, color="black")
        + labs(
            x=x,
            y=f"mean({y})",
            title=f"Means plot: {y} by {x}",
        )
        + theme_bw()
    )
    if error != "no":
        p += geom_errorbar(
            aes(ymin="__lower", ymax="__upper"),
            width=0.2,
            color="black",
        )
    if connect:
        p += geom_line(
            mapping=aes(x=x, y="mean", group=1),
            color="gray",
            linetype="dashed",
        )
    return p