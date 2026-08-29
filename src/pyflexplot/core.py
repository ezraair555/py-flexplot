import pandas as pd
import numpy as np
from typing import List, Optional
from plotnine import (
    ggplot,
    aes,
    geom_histogram,
    geom_point,
    geom_smooth,
    geom_jitter,
    geom_line,
    stat_summary,
    facet_wrap,
    facet_grid,
    scale_color_identity,
    labs,
    theme_bw,
)
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


def parse_flexplot_formula(formula: str):
    """
    Parses a flexplot formula of the form:
        outcome ~ predictor1 + predictor2 | given1 + given2

    Validates the formula syntax, strips whitespace from tokens, rejects empty
    outcome/predictor, handles the intercept-only form ``y ~ 1`` explicitly,
    and allows at most one ``|``.
    """
    if not isinstance(formula, str):
        raise TypeError(f"formula must be a string, got {type(formula).__name__}")

    if formula.count("~") != 1:
        raise ValueError(
            f"Formula must contain exactly one '~': {formula!r}"
        )

    if formula.count("|") > 1:
        raise ValueError(
            f"Formula may contain at most one '|': {formula!r}"
        )

    if "|" in formula:
        main_part, given_part = formula.split("|", 1)
    else:
        main_part = formula
        given_part = None

    main_part = main_part.strip()
    given_part = given_part.strip() if given_part is not None else None

    y_name, sep, x_formula = main_part.partition("~")
    y_name = y_name.strip()
    x_formula = x_formula.strip()

    if not y_name:
        raise ValueError(f"Formula must have a non-empty outcome: {formula!r}")
    if not x_formula:
        raise ValueError(
            f"Formula must have predictors after '~' (use 'y ~ 1' for intercept-only): {formula!r}"
        )

    # Intercept-only formula: y ~ 1
    if x_formula == "1":
        return {
            "y": y_name,
            "x": None,
            "color": None,
            "given": [g.strip() for g in given_part.split("+")] if given_part else [],
            "all_x": [],
            "intercept_only": True,
        }

    x_parts = [p.strip() for p in x_formula.split("+")]
    x_parts = [p for p in x_parts if p]

    if not x_parts:
        raise ValueError(
            f"Formula must have at least one predictor after '~': {formula!r}"
        )

    x_name = x_parts[0]
    color_name = x_parts[1] if len(x_parts) > 1 else None

    given_names = [g.strip() for g in given_part.split("+")] if given_part else []
    given_names = [g for g in given_names if g]

    return {
        "y": y_name,
        "x": x_name,
        "color": color_name,
        "given": given_names,
        "all_x": x_parts,
        "intercept_only": False,
    }


def _validate_data_for_plot(formula: str, data: pd.DataFrame, variables: dict, require_numeric_x: bool = False):
    """Shared validation for flexplot and added_plot."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data must be a pandas DataFrame, got {type(data).__name__}"
        )
    if data.empty:
        raise ValueError(
            f"data must be a non-empty DataFrame for formula {formula!r}"
        )

    y = variables["y"]
    x = variables["x"]
    given = variables.get("given", [])
    color = variables.get("color")

    required = {y}
    if x is not None:
        required.add(x)
    if color is not None:
        required.add(color)
    for g in given:
        required.add(g)

    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(
            f"Formula {formula!r} references missing columns in data: {missing}"
        )

    # Validate outcome column y is numeric (or numeric-convertible).
    if y is not None and not pd.api.types.is_numeric_dtype(data[y]):
        try:
            pd.to_numeric(data[y].dropna())
        except (ValueError, TypeError):
            raise ValueError(
                f"Column {y!r} must be numeric for formula {formula!r}, "
                f"got dtype {data[y].dtype}"
            )

    if require_numeric_x and x is not None and not pd.api.types.is_numeric_dtype(data[x]):
        raise ValueError(
            f"Column {x!r} must be numeric for formula {formula!r}, "
            f"got dtype {data[x].dtype}"
        )


def _is_discrete(series: pd.Series) -> bool:
    """
    Returns True if the series is non-numeric (string, object, categorical, bool)
    or is numeric with 10 or fewer unique non-null values.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return True
    return series.dropna().nunique() <= 10


_VALID_FLEXPLOT_METHODS = frozenset({"auto", "lm", "loess"})


def flexplot(formula: str, data: pd.DataFrame, method: str = "auto", **kwargs):
    """
    Intelligent multivariate graphics via formulas.
    """
    if method not in _VALID_FLEXPLOT_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_VALID_FLEXPLOT_METHODS)}; got {method!r}. "
            "Pass 'auto' for the default behaviour (LM for numeric-vs-numeric, "
            "binomial GLM for numeric-vs-binary)."
        )
    variables = parse_flexplot_formula(formula)
    _validate_data_for_plot(formula, data, variables)

    y = variables["y"]
    x = variables["x"]
    color = variables["color"]
    given = variables["given"]
    intercept_only = variables.get("intercept_only", False)

    if intercept_only:
        # Intercept-only: show a univariate distribution of y.
        p = ggplot(data, aes(x=y)) + geom_histogram(bins=30)
        p += labs(title=f"Distribution of {y}")
        p += theme_bw()
        return p

    # Reject 3+ given variables: the formula parser accepts them but only
    # two are actually used (facet_grid takes at most two).  Better to fail
    # loudly than silently drop the third.
    if len(given) > 2:
        raise ValueError(
            f"Formula {formula!r} has {len(given)} given variables after '|': "
            f"{given}.  flexplot supports at most 2 given variables "
            "(a row facet and a column facet)."
        )

    # Determine variable types
    is_y_numeric = pd.api.types.is_numeric_dtype(data[y])
    is_x_discrete = _is_discrete(data[x])

    plot_df = data.copy()
    # Convert numeric discrete X to string/categorical so plotnine treats x-axis as discrete levels
    if is_x_discrete and pd.api.types.is_numeric_dtype(plot_df[x]):
        plot_df[x] = plot_df[x].astype(str)

    # Build the base aesthetic with color/group when needed so all geoms pick it up.
    aes_kwargs = {"x": x, "y": y}
    if color:
        aes_kwargs["color"] = color
        aes_kwargs["group"] = color
    p = ggplot(plot_df, aes(**aes_kwargs))

    # Determine plot type
    if is_y_numeric and not is_x_discrete:
        p += geom_point(alpha=0.5)
        if method == "auto" or method == "lm":
            p += geom_smooth(method="lm") if color else geom_smooth(method="lm", color="blue")
        elif method == "loess":
            p += geom_smooth(method="loess") if color else geom_smooth(method="loess", color="blue")

    elif is_y_numeric and is_x_discrete:
        p += geom_jitter(width=0.2, alpha=0.5)
        p += stat_summary(fun_data="mean_cl_boot", color="red", size=1)

    elif not is_y_numeric and not is_x_discrete:
        # Validate binary 0/1 for binomial smoothing.
        try:
            unique_y = pd.Series(plot_df[y].dropna().astype(float)).unique()
        except (ValueError, TypeError):
            raise ValueError(
                f"Binomial smoothing requires a numeric binary 0/1 outcome; {y!r} "
                f"could not be converted to numeric"
            )
        if len(unique_y) != 2 or not set(unique_y).issubset({0.0, 1.0}):
            raise ValueError(
                f"Binomial smoothing requires a binary 0/1 outcome; {y!r} has "
                f"unique values: {sorted(unique_y)}"
            )
        p += geom_point(alpha=0.3)
        p += geom_smooth(method="glm", method_args={"family": "binomial"})

    else:
        p += geom_jitter(width=0.2, height=0.2, alpha=0.5)

    if len(given) == 1:
        p += facet_wrap(f"~{given[0]}")
    elif len(given) >= 2:
        p += facet_grid(f"{given[1]} ~ {given[0]}")

    p += theme_bw()
    return p


def _first_non_intercept_name(model, fallback="x"):
    """Return the first non-intercept exog name or None if only intercept."""
    names = getattr(getattr(model, "model", None), "exog_names", None) or []
    non_intercept = [n for n in names if n not in ("Intercept", "const")]
    return non_intercept[0] if non_intercept else None


def _is_neural_net_fit(model) -> bool:
    """Duck-type detection of a :class:`pyflexplot.flex_nn.NeuralNetFit`.

    Avoids importing :mod:`flex_nn` at module load time so the core module
    stays cheap to import when neural-net support isn't needed.  A
    ``NeuralNetFit`` exposes ``.predict(data)`` returning an indexed
    Series plus the response-var metadata the wrapper carries.
    """
    cls = type(model)
    cls_name = f"{cls.__module__}.{cls.__qualname__}"
    return cls_name == "pyflexplot.flex_nn.NeuralNetFit"


def _visualize_neural_net(fit, data=None, **kwargs):
    """Visualization path for ``NeuralNetFit`` wrappers.

    Mirrors the statsmodels ``visualize()`` output (predicted-vs-actual
    line on top of a scatter) so a user can drop a fitted network into
    the same plot they would have used for an OLS fit.
    """
    if data is None:
        raise ValueError(
            "visualize() requires data= when called on a NeuralNetFit "
            "(the wrapper does not carry the training data)."
        )
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data must be a pandas DataFrame, got {type(data).__name__}"
        )
    if data.empty:
        raise ValueError("data must be non-empty for visualization.")

    response_var = fit.response_var
    if response_var not in data.columns:
        raise ValueError(
            f"Response column {response_var!r} (declared on the NeuralNetFit) "
            f"not found in data. Available columns: {list(data.columns)}"
        )

    # Determine the x predictor.  Honour explicit x=, otherwise use the
    # first declared predictor (mirrors the statsmodels fallback in
    # visualize()).
    x_name = kwargs.get("x")
    if x_name is None:
        if not fit.predictor_names:
            raise ValueError(
                "NeuralNetFit has no declared predictor_names; "
                "pass an explicit x= argument."
            )
        x_name = fit.predictor_names[0]
    if x_name not in data.columns:
        raise ValueError(
            f"Predictor column {x_name!r} not found in data for visualization."
        )

    pred = fit.predict(data)

    plot_df = data.copy()
    plot_df["__predicted"] = pred.reindex(plot_df.index)

    p = (
        ggplot(plot_df, aes(x=x_name, y=response_var))
        + geom_point(alpha=0.4)
        + geom_line(aes(y="__predicted"), color="red", size=1)
        + labs(
            title=f"Visualization: NeuralNetFit ({fit.backend})",
            subtitle=f"Predicted {response_var} vs {x_name}",
        )
        + theme_bw()
    )
    return p


def visualize(model, data: Optional[pd.DataFrame] = None, **kwargs):
    """
    Provides a visual representation of a fitted statistical object.
    Supports statsmodels (OLS, GLM), scikit-learn models, and
    :class:`pyflexplot.flex_nn.NeuralNetFit` wrappers.
    """
    # NeuralNetFit path: duck-typed dispatch so core.py doesn't have to
    # import flex_nn at module load time.
    if _is_neural_net_fit(model):
        return _visualize_neural_net(model, data=data, **kwargs)

    if data is None:
        if hasattr(model, "model") and hasattr(model.model, "data"):
            data = pd.DataFrame(model.model.data.orig_endog).join(
                pd.DataFrame(model.model.data.orig_exog)
            )
        else:
            raise ValueError("No data provided for visualization.")

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    if data.empty:
        raise ValueError("data must be non-empty for visualization.")

    if not hasattr(model, "predict"):
        raise NotImplementedError(
            f"Visualization for {type(model).__name__} not yet implemented "
            "(model has no predict method)."
        )

    endog_names = getattr(getattr(model, "model", None), "endog_names", None)
    if endog_names is None:
        raise ValueError("Cannot determine outcome variable name from model.")

    # Statsmodels predict returns a Series aligned to the input index.
    y_pred = model.predict(data)
    if not hasattr(y_pred, "index"):
        y_pred = pd.Series(y_pred, index=data.index)

    plot_df = data.copy()
    plot_df["__predicted"] = y_pred.reindex(plot_df.index)

    # Determine the first predictor robustly.
    x_name = kwargs.get("x")
    if x_name is None:
        x_name = _first_non_intercept_name(model)
    if x_name is None:
        raise ValueError(
            "visualize() requires a non-intercept model or explicit x= argument."
        )
    if x_name not in plot_df.columns:
        raise ValueError(
            f"Predictor column {x_name!r} not found in data for visualization."
        )

    p = (
        ggplot(plot_df, aes(x=x_name, y=endog_names))
        + geom_point(alpha=0.4)
        + geom_line(aes(y="__predicted"), color="red", size=1)
        + labs(
            title=f"Visualization: {type(model).__name__}",
            subtitle=f"Predicted {endog_names} vs {x_name}",
        )
        + theme_bw()
    )
    return p


def compare_fits(
    formula: str,
    data: pd.DataFrame,
    model1,
    model2,
    labels: List[str] = ["Model 1", "Model 2"],
    **kwargs,
):
    """
    Visually compare the fit of two different models (statsmodels/sklearn).
    """
    variables = parse_flexplot_formula(formula)
    _validate_data_for_plot(formula, data, variables)

    y_name = variables["y"]
    x_name = variables["x"]

    pred1 = _get_model_predictions(model1, data)
    pred2 = _get_model_predictions(model2, data)

    if len(pred1) != len(data) or len(pred2) != len(data):
        raise ValueError(
            f"Model predictions must match data length ({len(data)}): "
            f"got {len(pred1)} and {len(pred2)}."
        )

    plot_df = data.copy()
    plot_df["__m1"] = pred1
    plot_df["__m2"] = pred2

    p = (
        ggplot(plot_df, aes(x=x_name, y=y_name))
        + geom_point(alpha=0.3)
        + geom_line(aes(y="__m1", color='"#3498db"'), size=1)
        + geom_line(aes(y="__m2", color='"#e74c3c"'), size=1)
        + scale_color_identity(
            guide="legend",
            name="Model",
            labels=labels,
            breaks=["#3498db", "#e74c3c"],
        )
        + labs(title="Visual Model Comparison", x=x_name, y=y_name)
        + theme_bw()
    )

    return p


def _get_model_predictions(model, data: pd.DataFrame) -> pd.Series:
    """Return a pandas Series of predictions aligned to data.index."""
    if hasattr(model, "predict"):
        try:
            # Statsmodels fitted models accept a DataFrame and return a Series
            # indexed by the (possibly reduced) observation index.
            pred = model.predict(data)
        except Exception:
            # scikit-learn style: needs a 2-D array-like input.
            pred = model.predict(data.values)
    else:
        raise ValueError(
            f"Model of type {type(model).__name__} has no predict method."
        )

    if isinstance(pred, pd.Series):
        return pred.reindex(data.index)

    if isinstance(pred, pd.DataFrame):
        if pred.shape[1] == 1:
            pred = pred.iloc[:, 0]
        else:
            raise ValueError(
                f"Model predictions must be 1-D, got shape {pred.shape}"
            )
        return pred.reindex(data.index)

    pred = np.asarray(pred)
    if pred.ndim != 1:
        if pred.shape[1] == 1:
            pred = pred.ravel()
        else:
            raise ValueError(
                f"Model predictions must be 1-D, got shape {pred.shape}"
            )
    return pd.Series(pred, index=data.index)


def added_plot(formula: str, data: pd.DataFrame, **kwargs):
    """
    Generates an added variable plot (partial regression plot).
    """
    variables = parse_flexplot_formula(formula)
    _validate_data_for_plot(formula, data, variables, require_numeric_x=True)

    y_var = variables["y"]
    x_var = variables["x"]
    other_vars = [v for v in variables["all_x"] if v != x_var]

    if not other_vars:
        return flexplot(formula, data, **kwargs)

    # Residuals of Y on other vars
    y_res_model = OLS.from_formula(
        f"{y_var} ~ {' + '.join(other_vars)}", data=data
    ).fit()
    y_residuals = y_res_model.resid

    # Residuals of X on other vars
    x_res_model = OLS.from_formula(
        f"{x_var} ~ {' + '.join(other_vars)}", data=data
    ).fit()
    x_residuals = x_res_model.resid

    # Align residuals on the shared original index to avoid positional mismatch.
    res_df = pd.concat(
        {
            f"res_{y_var}": y_residuals,
            f"res_{x_var}": x_residuals,
        },
        join="inner",
        axis=1,
    )

    if len(res_df) != len(data):
        raise ValueError(
            f"Residual lengths do not match original data length: "
            f"{len(res_df)} vs {len(data)}. Check for missing data in "
            f"the variables used by the formula."
        )

    p = (
        ggplot(res_df, aes(x=f"res_{x_var}", y=f"res_{y_var}"))
        + geom_point(alpha=0.5)
        + geom_smooth(method="lm", color="blue")
        + labs(
            x=f"{x_var} | others",
            y=f"{y_var} | others",
            title="Added Variable Plot",
        )
        + theme_bw()
    )

    return p
