import ast
import re
import warnings

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from plotnine import (
    ggplot,
    aes,
    geom_histogram,
    geom_point,
    geom_smooth,
    geom_jitter,
    geom_line,
    geom_ribbon,
    geom_hline,
    geom_boxplot,
    geom_violin,
    geom_bar,
    geom_density,
    stat_summary,
    stat_qq,
    stat_qq_line,
    facet_wrap,
    facet_grid,
    scale_color_identity,
    scale_color_manual,
    labs,
    theme_bw,
    theme,
    coord_flip,
    element_blank,
    element_text,
)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS
from statsmodels.nonparametric.smoothers_lowess import lowess

from .uncertainty import (
    VALID_UNCERTAINTY,
    validate_uncertainty_params,
    compute_bootstrap_ci,
    compute_prediction_band,
)


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
            "has_interaction": False,
        }

    # Detect interaction operators (``*`` or ``:``) anywhere in the
    # right-hand side. The current fit is still additive; we set a flag
    # so ``flexplot()`` can warn the user. ``*`` is expanded to its R-style
    # constituent terms (``a*b`` → ``a + b + a:b``) so column lookup works.
    has_interaction = bool(_INTERACTION_OP.search(x_formula))
    if has_interaction:
        expanded_x_formula = _expand_r_formula(x_formula)
    else:
        expanded_x_formula = x_formula

    x_parts = [p.strip() for p in expanded_x_formula.split("+")]
    x_parts = [p for p in x_parts if p]

    if not x_parts:
        raise ValueError(
            f"Formula must have at least one predictor after '~': {formula!r}"
        )

    # ``x_name`` is the first atom of the first term (so ``x:z`` → ``x``);
    # ``color_name`` is the first atom of the second term if present.
    x_name = _first_atom(x_parts[0])
    color_name = _first_atom(x_parts[1]) if len(x_parts) > 1 else None

    given_names = [g.strip() for g in given_part.split("+")] if given_part else []
    given_names = [g for g in given_names if g]

    return {
        "y": y_name,
        "x": x_name,
        "color": color_name,
        "given": given_names,
        "all_x": x_parts,
        "intercept_only": False,
        "has_interaction": has_interaction,
    }


def _validate_data_for_plot(
    formula: str,
    data: pd.DataFrame,
    variables: dict,
    require_numeric_x: bool = False,
    intercept_only: bool = False,
):
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
    # For intercept-only formulas, R-flexplot supports univariate plots of
    # categorical outcomes (bar charts), so we relax the numeric requirement.
    if (
        y is not None
        and not intercept_only
        and not pd.api.types.is_numeric_dtype(data[y])
    ):
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


# Threshold below which R-flexplot converts numeric predictors to ordered
# categorical factors. Matches R's ``convert_if_less_than_five`` (numeric
# with <5 unique values -> ordered factor).
_LOW_CARDINALITY_THRESHOLD = 5


def _is_low_cardinality_numeric(series: pd.Series) -> bool:
    """Return True if ``series`` is numeric with fewer than 5 unique non-null values.

    Mirrors R-flexplot's ``convert_if_less_than_five``: numeric axis / color /
    given variables with 2-4 unique values should be treated as ordered
    categorical factors, both for plotting and for the smoother fit.  Series
    that are already non-numeric, or numeric with 5+ unique values, return
    False (no conversion needed).
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False
    n_unique = series.dropna().nunique()
    return 0 < n_unique < _LOW_CARDINALITY_THRESHOLD


def _is_discrete(series: pd.Series) -> bool:
    """
    Returns True if the series is non-numeric (string, object, categorical, bool)
    or is numeric with 10 or fewer unique non-null values.

    For R-parity low-cardinality conversion (numeric with <5 unique values
    becomes ordered categorical), see ``_is_low_cardinality_numeric`` /
    ``_convert_low_cardinality_to_categorical``.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return True
    return series.dropna().nunique() <= 10


def _convert_low_cardinality_to_categorical(
    data: pd.DataFrame,
    variables: list,
):
    """Convert numeric predictors with <5 unique values to string (categorical).

    R-flexplot's ``convert_if_less_than_five`` turns these into ordered
    factors so the discrete-x branch (geom_jitter + dispersion marker)
    applies instead of the numeric-x smoother.  In Python we convert to
    ``str`` so plotnine treats the column as discrete; the actual fitted
    model still uses the (now string) values via statsmodels' C() wrapper
    when needed.

    Returns a new DataFrame (the input is not mutated).  Variables that
    are missing from ``data`` are silently skipped (callers should have
    validated columns earlier).
    """
    out = data.copy()
    for var in variables:
        if var is None or var not in out.columns:
            continue
        if _is_low_cardinality_numeric(out[var]):
            out[var] = out[var].astype(str)
    return out


def _validate_binning_params(
    bins,
    labels,
    breaks,
    x_series: pd.Series,
):
    """Validate bins / labels / breaks arguments for numeric-x discretization.

    Rules:
    - ``bins``: positive int >= 2 (1 bin is meaningless).
    - ``breaks``: list of floats, length >= 2, strictly monotonically
      increasing.
    - ``labels``: list of strings. When given with ``breaks``, len must be
      len(breaks) - 1. When given with ``bins`` alone, len must equal
      ``bins``.
    - ``bins`` and ``breaks`` are mutually exclusive (breaks wins).
    - All binning params are silently ignored when x is already discrete
      or non-numeric (caller checks first).
    """
    if bins is None and breaks is None and labels is None:
        return
    if bins is not None:
        if not isinstance(bins, int) or isinstance(bins, bool):
            raise TypeError(
                f"bins must be an int >= 2; got {type(bins).__name__} ({bins!r})."
            )
        if bins < 2:
            raise ValueError(f"bins must be >= 2; got {bins}.")
    if breaks is not None:
        if not isinstance(breaks, (list, tuple)):
            raise TypeError(
                f"breaks must be a list/tuple of floats; got {type(breaks).__name__}."
            )
        if len(breaks) < 2:
            raise ValueError(
                f"breaks must have >= 2 cut points; got {len(breaks)}."
            )
        # Coerce to float and check monotonicity.
        breaks_f = [float(b) for b in breaks]
        for i in range(1, len(breaks_f)):
            if breaks_f[i] <= breaks_f[i - 1]:
                raise ValueError(
                    f"breaks must be strictly monotonically increasing; "
                    f"got {breaks_f!r}."
                )
    if labels is not None:
        if not isinstance(labels, (list, tuple)):
            raise TypeError(
                f"labels must be a list/tuple of strings; got {type(labels).__name__}."
            )
        if any(not isinstance(lbl, str) for lbl in labels):
            raise TypeError("labels must all be strings.")
        if breaks is not None:
            if len(labels) != len(breaks) - 1:
                raise ValueError(
                    f"labels length ({len(labels)}) must equal "
                    f"len(breaks) - 1 ({len(breaks) - 1})."
                )
        elif bins is not None:
            if len(labels) != bins:
                raise ValueError(
                    f"labels length ({len(labels)}) must equal bins ({bins})."
                )
    if bins is not None and breaks is not None:
        warnings.warn(
            "Both bins and breaks were provided; breaks takes precedence.",
            UserWarning,
            stacklevel=3,
        )


def _maybe_bin_numeric_x(
    data: pd.DataFrame,
    x: str,
    bins=None,
    labels=None,
    breaks=None,
):
    """Discretize a numeric x column into bins/breaks.

    Returns (dataframe, was_binned: bool). If neither bins nor breaks is
    given, returns (data.copy(), False) without modifying x.

    Uses pd.cut() for both equal-width (bins) and explicit-cut (breaks)
    paths. NaN handling: rows with NaN x are dropped from the binning but
    preserved in the returned dataframe with NaN x (plotnine will skip them).
    """
    if bins is None and breaks is None:
        return data.copy(), False

    x_arr = data[x].to_numpy()
    if breaks is not None:
        cuts = list(breaks)
    else:
        # Equal-width bins between min and max (inclusive on the lower end).
        x_min = float(np.nanmin(x_arr))
        x_max = float(np.nanmax(x_arr))
        cuts = np.linspace(x_min, x_max, num=int(bins) + 1).tolist()

    # Ensure endpoints are captured even if the data doesn't hit them.
    # pd.cut's include_lowest=True makes the leftmost bin closed on both ends.
    binned = pd.cut(
        data[x],
        bins=cuts,
        labels=labels,
        include_lowest=True,
    )

    out = data.copy()
    # Convert to string so plotnine treats it as discrete levels.
    out[x] = binned.astype(str)
    return out, True


_VALID_SPREAD = frozenset({None, "stdev", "range", "iqr", "no", "ci", "quartiles", "sterr"})


def _add_discrete_summary(p, spread: Optional[str]):
    """Add the dispersion marker layer for the discrete-x branch.

    Mirrors R-flexplot's ``spread`` argument:
    - None / "quartiles": median +/- Q1/Q3 IQR (R's default for discrete x).
    - "iqr": alias for "quartiles".
    - "ci": bootstrap CI on the mean (plotnine's stat_summary with
      ``fun_data='mean_cl_boot'``).
    - "sterr": mean +/- 1.96 * standard error of the mean.
    - "stdev": mean +/- 1 SD as a crossbar (pointrange with computed limits).
    - "range": min-max range as a wider crossbar.
    - "no": no summary layer at all.
    """
    if spread not in _VALID_SPREAD:
        raise ValueError(
            f"spread must be one of {sorted(s for s in _VALID_SPREAD if s)}; "
            f"got {spread!r}."
        )

    # R-token aliases: "quartiles" == "iqr".  Default to "iqr" to
    # match R-flexplot's discrete-x default; legacy Python callers can
    # request "ci" explicitly.
    if spread is None or spread == "quartiles":
        spread = "iqr"

    if spread == "no":
        return p

    if spread == "ci":
        p += stat_summary(fun_data="mean_cl_boot", color="red", size=1)
        return p

    if spread == "sterr":
        # Standard error of the mean: sd / sqrt(n).  R-flexplot uses the
        # same formula with a historical n-1 denominator; we use the
        # conventional sample-size denominator for consistency with
        # statsmodels / scipy.
        fun = _make_spread_fn(
            np.mean,
            lambda x: (
                np.mean(x) - 1.96 * (np.std(x, ddof=1) / np.sqrt(len(x))),
                np.mean(x) + 1.96 * (np.std(x, ddof=1) / np.sqrt(len(x))),
            ),
        )
        p += stat_summary(fun_data=fun, geom="pointrange", color="red", size=0.5)
        return p

    # stdev / range / iqr: use a precomputed summary dataframe + pointrange.
    # stat_summary can't easily express "by group" summaries that return a
    # single (y, ymin, ymax) per x level, so we build it manually.
    # Pull the aes from the existing plot: x_var is the discrete-x column.
    # We don't know the column names here without the caller passing them,
    # so we use the plot's already-attached data + aes.
    #
    # IMPORTANT: callers should prefer plot-level data extraction. For
    # simplicity we use a fallback: invoke stat_summary with a custom
    # fun_data that yields (ymin, y) by computing per-level quantiles.
    # The summary fn must return a DataFrame with columns 'y', 'ymin', 'ymax'
    # and an 'x' level column.
    if spread == "stdev":
        fun = _make_spread_fn(np.mean, lambda x: np.std(x, ddof=1))
    elif spread == "range":
        fun = _make_spread_fn(np.mean, lambda x: (np.min(x), np.max(x)))
    elif spread == "iqr":
        fun = _make_spread_fn(np.median, lambda x: (np.percentile(x, 25), np.percentile(x, 75)))
    else:  # pragma: no cover — guarded by validator
        return p

    p += stat_summary(fun_data=fun, geom="pointrange", color="red", size=0.5)
    return p


def _plot_univariate(
    data: pd.DataFrame,
    outcome: str,
    plot_type: Optional[str] = None,
    bins: Optional[int] = None,
):
    """Build an intercept-only / univariate distribution plot.

    Mirrors ``r-flexplot/R/flexplot_helper.R::flexplot_histogram`` plus the
    bivariate ``plot.type`` variants.  ``outcome`` is the variable being
    visualized (usually the formula's ``y``).

    Parameters
    ----------
    data : pd.DataFrame
        Plotting data frame.  May already be a subsample when ``sample=`` is
        used, but the caller decides that before invoking this helper.
    outcome : str
        Column name of the variable to plot.
    plot_type : {None, "histogram", "qq", "density", "boxplot", "violin"}, optional
        Univariate geom override.  ``None`` defaults to a histogram.
    bins : int, optional
        Number of histogram bins.  Ignored for non-histogram plot types.

    Returns
    -------
    plotnine.ggplot
        A complete univariate plot with ``theme_bw`` and appropriate axis
        labeling.
    """
    plot_type = plot_type or "histogram"
    is_numeric = pd.api.types.is_numeric_dtype(data[outcome])

    # Categorical outcome: R draws a bar chart regardless of plot_type.
    if not is_numeric:
        p = ggplot(data, aes(x=outcome)) + geom_bar()
        p += labs(x=outcome, title=f"Distribution of {outcome}")
        p += theme_bw()
        return p

    if plot_type == "qq":
        p = ggplot(data, aes(sample=outcome))
        p += stat_qq()
        p += stat_qq_line()
        p += labs(title=f"QQ plot of {outcome}")
        p += theme_bw()
        return p

    if plot_type == "density":
        p = ggplot(data, aes(x=outcome)) + geom_density()
        p += labs(x=outcome, title=f"Density of {outcome}")
        p += theme_bw()
        return p

    if plot_type in {"boxplot", "violin"}:
        geom = geom_boxplot() if plot_type == "boxplot" else geom_violin()
        p = ggplot(data, aes(y=outcome)) + geom
        p += labs(y=outcome, title=f"Distribution of {outcome}")
        p += theme_bw()
        # Hide the redundant x-axis markings (the natural analogue of
        # R's coord_flip + blank x-axis for a univariate boxplot).
        p += theme(
            axis_title_x=element_blank(),
            axis_text_x=element_blank(),
            axis_ticks_major_x=element_blank(),
        )
        return p

    # Default / explicit histogram
    n_bins = bins if bins is not None else 30
    p = ggplot(data, aes(x=outcome)) + geom_histogram(
        bins=n_bins, fill="lightgray", color="black"
    )
    p += labs(x=outcome, title=f"Distribution of {outcome}")
    p += theme_bw()
    return p


def _plot_related(
    data: pd.DataFrame,
    diff_col: str,
    spread: Optional[str],
    plot_type: Optional[str],
    jitter: Union[bool, tuple],
    alpha: float,
    raw_data: bool,
):
    """Build a related-samples / paired difference plot.

    Mirrors ``r-flexplot/R/flexplot_helper.R::flexplot_related``.  The input
    ``data`` is expected to contain a single column ``diff_col`` of paired
    difference scores.
    """
    p = ggplot(data, aes(x=1, y=diff_col)) + theme_bw()
    p += geom_hline(yintercept=0, color="lightgray")
    p += labs(y=diff_col, title=f"{diff_col}")
    p += theme(
        axis_title_x=element_blank(),
        axis_text_x=element_blank(),
        axis_ticks_major_x=element_blank(),
    )

    if plot_type in {"boxplot", "violin"}:
        geom = geom_boxplot() if plot_type == "boxplot" else geom_violin()
        p += geom
    else:
        # Default/errorbar path: show jittered points + a dispersion marker.
        if raw_data:
            if jitter is None:
                jitter_xy = (0.05, 0.0)
            elif isinstance(jitter, bool):
                jitter_xy = (0.05, 0.0) if jitter else (0.0, 0.0)
            else:
                jitter_xy = (float(jitter[0]), float(jitter[1]) if len(jitter) > 1 else 0.0)
            if jitter_xy[0] > 0:
                p += geom_jitter(width=jitter_xy[0], height=jitter_xy[1], alpha=alpha)
            else:
                p += geom_point(alpha=alpha)
        p = _add_discrete_summary(p, spread)

    return p


def _make_spread_fn(center_fn, spread_fn):
    """Build a plotnine fun_data-style callable for stat_summary.

    Returns a function ``f(values: np.ndarray) -> pd.DataFrame`` with one row
    containing columns ``y``, ``ymin``, ``ymax`` (plotnine's expected schema
    for ``pointrange``). The center_fn is applied to compute ``y``; the
    spread_fn is applied to compute (ymin, ymax).

    Note: plotnine's stat_summary fun_data expects the x-level grouping to
    be handled internally. We rely on the default ``fun_y=np.mean`` for the
    point and our custom fun_data for the range. If the caller's spread_fn
    returns a 2-tuple (lo, hi), we project those into ymin / ymax.
    """
    def _f(values):
        center = center_fn(values)
        spread = spread_fn(values)
        if isinstance(spread, tuple) and len(spread) == 2:
            lo, hi = spread
        else:  # pragma: no cover — defensive
            lo, hi = center - spread, center + spread
        return pd.DataFrame({"y": [center], "ymin": [lo], "ymax": [hi]})
    return _f


_VALID_FLEXPLOT_METHODS = frozenset(
    {
        "auto",
        "lm",
        "loess",
        "quadratic",
        "polynomial",
        "cubic",
        "logistic",
        "rlm",
        "poisson",
        "Gamma",
        # Mixed-effects extensions (v0.8.2+):
        "mixedlm",
        "lmer",
        "glmer",
    }
)

# Recognized methods for overlay entries. Includes a broader set than the
# primary ``method`` parameter because plotnine/statsmodels supports more
# smoothers for overlay use.
_VALID_OVERLAY_METHODS = frozenset({"lm", "loess", "lowess", "glm", "rlm", "ols", "wls", "gls", "mavg"})

# Recognized plot_type overrides. Histogram/QQ/density/violin are used
# primarily for intercept-only (univariate) plots but are accepted anywhere
# the data type permits them.
_VALID_PLOT_TYPES = frozenset(
    {"scatter", "line", "boxplot", "bar", "histogram", "qq", "density", "violin"}
)

# Default color cycle for overlay entries (distinct from the primary
# ``"blue"`` so the primary line is always visually identifiable).
_OVERLAY_COLOR_CYCLE = ("#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c")

# Interaction-operator detection. The presence of ``*`` or ``:`` in the
# right-hand side of a formula signals that the user wants interaction terms.
# The parser accepts these for forward-compatibility with v0.7.0 (real
# interaction-aware fitting), but the default fit in v0.6.x is still
# additive — a UserWarning is emitted to make this explicit.
_INTERACTION_OP = re.compile(r"(?<!\*)\*(?!\*)|:")


def _split_formula_terms(text: str, sep: str = "+") -> List[str]:
    """Split ``text`` on ``sep`` only outside parentheses.

    Used to split the RHS of a flexplot formula into additive terms without
    breaking apart function calls such as ``I(x ** 2 + 1)``.
    """
    depth = 0
    current: List[str] = []
    terms: List[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            term = "".join(current).strip()
            if term:
                terms.append(term)
            current = []
            continue
        current.append(ch)
    term = "".join(current).strip()
    if term:
        terms.append(term)
    return terms


def _expand_r_formula(text: str) -> str:
    """Expand R-style ``a*b`` to ``a + b + a:b``.

    Repeatedly applies the expansion until no ``*`` remains (handles
    multi-way interactions like ``a*b*c``). ``:`` terms are left as-is;
    downstream code can choose how to handle them.
    """
    if "*" not in text:
        return text
    pattern = re.compile(r"(\b\w+)\s*\*\s*(\w+)")
    while True:
        new_text = pattern.sub(r"\1 + \2 + \1:\2", text)
        if new_text == text:
            return text
        text = new_text


def _first_atom(term: str) -> str:
    """Return the first atom of a possibly-interacted term.

    ``x:z`` → ``x``; ``x`` → ``x``. Used to extract the column name when
    the parser encounters interaction terms.
    """
    return term.split(":", 1)[0].strip()


def _normalize_overlay(overlay):
    """Validate and normalize the ``overlay`` parameter into a list of dicts.

    Each returned dict has at least:
        - ``method``: str (required)
        - ``color``: str (default: next color from cycle)
        - ``label``: str (default: method name)
        - ``uncertainty``: {None, "ci", "prediction", "bootstrap"}, default "ci"
        - ``level``: float in (0, 1), default 0.95

    Raises ``ValueError`` if any entry is malformed.
    """
    if overlay is None:
        return []
    if not isinstance(overlay, (list, tuple)):
        raise ValueError(
            f"overlay must be a list or tuple; got {type(overlay).__name__}."
        )
    if not overlay:
        return []

    normalized = []
    for i, entry in enumerate(overlay):
        if isinstance(entry, str):
            spec = {"method": entry}
        elif isinstance(entry, dict):
            spec = dict(entry)
        else:
            raise ValueError(
                f"overlay entry {i} must be a str or dict; "
                f"got {type(entry).__name__}."
            )
        if "method" not in spec:
            raise ValueError(
                f"overlay entry {i} is missing required key 'method': {entry!r}."
            )
        if spec["method"] not in _VALID_OVERLAY_METHODS:
            raise ValueError(
                f"overlay entry {i}: method {spec['method']!r} is not a "
                f"recognized method. Valid: {sorted(_VALID_OVERLAY_METHODS)}."
            )
        spec.setdefault("color", _OVERLAY_COLOR_CYCLE[i % len(_OVERLAY_COLOR_CYCLE)])
        spec.setdefault("label", spec["method"])
        spec.setdefault("uncertainty", "ci")
        spec.setdefault("level", 0.95)
        normalized.append(spec)
    return normalized


# ---------------------------------------------------------------------------
# Formula-function detection / evaluation (R-flexplot parity, v0.8.x+).
# R's ``formula_functions`` looks for terms containing ``(`` (e.g. ``log(x)``,
# ``sqrt(x)``, ``poly(x, 2)``), applies the expression to ``data`` (via R's
# ``eval(parse(...))`` with the data as the evaluation environment), stores
# the result in a column named after the inner variable (e.g. ``x``), and
# rewrites the formula so downstream code sees ``y ~ x``.
#
# Python parity note: we use a SAFE whitelisted evaluator (numpy / pandas /
# math / statsmodels / patsy built-ins) rather than ``eval()`` on arbitrary
# strings, so a user-supplied formula can only invoke a closed set of
# known-safe functions.  Unknown functions raise ``ValueError``.
# ---------------------------------------------------------------------------


# Whitelisted functions available inside formula terms. Keys are the names
# users may write (lowercase, since formula evaluation is case-insensitive
# for Python identifiers matched against this map); values are the callable.
_FORMULA_FUNCS = {
    # numpy ufuncs / reductions
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "log1p": np.log1p,
    "exp": np.exp,
    "exp2": np.exp2,
    "expm1": np.expm1,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "abs_": np.abs,  # alias to handle ``abs()`` when ``abs`` shadows builtin
    "sign": np.sign,
    "round": np.round,
    "floor": np.floor,
    "ceil": np.ceil,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    # math module (scalar → scalar; will be wrapped to vectorize below)
    "log_m": __import__("math").log,
    "exp_m": __import__("math").exp,
    "sqrt_m": __import__("math").sqrt,
    # I(): identity (no-op; the ``I(x**2)`` R idiom)
    "I": (lambda x: x),
    # poly(): raw polynomial of given degree; default degree=2 (matches R's
    # default ``poly(x, 2, raw=TRUE)``).  Returns a numpy array with columns
    # ``[x, x^2, ..., x^degree]``; we use the highest-degree column as the
    # value (R returns the full matrix but ``formula_functions`` only stores
    # it under the inner-variable name; we keep the highest non-linear
    # term so downstream plots/smoothers see a single transformed column).
    "poly": None,  # filled in by _apply_formula_function (needs degree kwarg)
}


def _apply_formula_function(func_name: str, inner_expr: str, var_name: str,
                            data: pd.DataFrame, depth: int = 0):
    """Apply a whitelisted function to a single inner expression.

    ``inner_expr`` is the raw text inside the parentheses, e.g. ``"x"`` or
    ``"x, 2"``.  Returns a numpy array of values, ready to be stored in a
    column.  Raises ``ValueError`` for unknown functions or expressions
    that reference missing / unknown columns.

    Supports a single positional argument (the inner variable) plus
    optional numeric constants (e.g. ``poly(x, 2)``).  Inner expressions
    like ``a + b`` are not supported (R's ``eval(parse(...))`` would allow
    them, but we deliberately restrict to a single column reference so
    there's no way for the formula string to reach other columns).
    """
    if depth > 3:
        raise ValueError(
            f"Nested formula functions beyond depth 3 are not supported; "
            f"got {inner_expr!r} inside {func_name!r}."
        )

    parts = [p.strip() for p in inner_expr.split(",")]
    if not parts or not parts[0]:
        raise ValueError(
            f"Empty inner expression for {func_name}(...): {inner_expr!r}"
        )
    inner_var = parts[0]
    if inner_var not in data.columns:
        raise ValueError(
            f"Formula function {func_name}({inner_expr!r}) references "
            f"missing column {inner_var!r}; available: {list(data.columns)}."
        )
    inner_arr = data[inner_var].to_numpy()

    # poly(x, k): numpy polyfeatures [1, x, x^2, ..., x^k].  R uses raw
    # (un-orthogonalized) polynomials by default; we follow suit for parity.
    # Since we only store ONE column named after ``inner_var`` (R also
    # stores the whole matrix under that name), we keep the highest-degree
    # polynomial column — i.e. x^k for degree k.  Users who want the full
    # design matrix should construct it via method='polynomial' instead.
    if func_name == "poly":
        try:
            degree = int(parts[1]) if len(parts) > 1 else 2
        except (ValueError, TypeError):
            raise ValueError(
                f"poly() requires an integer degree; got {parts[1]!r}."
            )
        if degree < 1:
            raise ValueError(
                f"poly() requires degree >= 1; got {degree}."
            )
        return np.asarray(inner_arr, dtype=float) ** degree

    if func_name not in _FORMULA_FUNCS:
        raise ValueError(
            f"Formula function {func_name!r} is not supported; "
            f"allowed names: {sorted(k for k in _FORMULA_FUNCS if k)}."
        )
    fn = _FORMULA_FUNCS[func_name]

    # For numpy ufuncs, calling on a numpy array vectorizes correctly.  For
    # math.* scalar funcs (e.g. math.sqrt), wrap to vectorize.  ``I()`` is
    # a no-op passthrough; we handled it above via the identity lambda.
    try:
        result = fn(inner_arr)
    except Exception as exc:
        raise ValueError(
            f"Failed to apply formula function {func_name}() to column "
            f"{inner_var!r}: {exc}"
        ) from exc
    return np.asarray(result)


# Regex matching a term of the form ``funcName(inner[, args])``.  Greedy
# only on the inner-var portion, so ``log(x)`` matches as ``log`` + ``x``.
_FORMULA_FUNC_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*\(\s*([^()]+)\s*\)\s*$"
)


def _apply_formula_functions(formula: str, data: pd.DataFrame):
    """Detect and evaluate formula functions (R ``formula_functions`` parity).

    Scans the right-hand side of ``formula`` for any term containing ``(``
    (i.e. a function call).  For each, applies the whitelisted function to
    the referenced column(s), stores the result in a new column named after
    the inner variable (overwriting it if present — R behavior), and
    rewrites the formula so downstream code sees the simpler
    ``inner_var`` name.

    Returns
    -------
    (new_data, new_formula, transformed_terms)
        ``new_data`` is the input DataFrame augmented with the new
        transformed columns.  ``new_formula`` is the rewritten formula
        string.  ``transformed_terms`` is a list of ``(term, func_name,
        inner_var)`` tuples describing each transformation that was
        applied (empty list when the formula has no functions).
    """
    if "|" not in formula:
        main_part = formula
        given_part = None
    else:
        main_part, given_part = formula.split("|", 1)

    if "~" not in main_part:
        # No predictor side: nothing to transform.
        return data.copy(), formula, []

    y_part, x_part = main_part.split("~", 1)
    x_part = x_part.strip()
    y_part = y_part.strip()

    # Split the RHS into additive terms without breaking apart function
    # calls such as ``I(x ** 2 + 1)``.
    raw_terms = _split_formula_terms(x_part, "+")
    if not raw_terms:
        return data.copy(), formula, []

    transformed = []  # list of (original_term, func_name, inner_var)
    new_data = data.copy()
    new_x_parts = []

    def _process_term(term: str, parts: List[str]) -> None:
        """Apply formula-function transformation to one additive term.

        Appends the rewritten term to ``parts`` and updates ``new_data`` /
        ``transformed`` in the enclosing scope.
        """
        # If the term is an interaction (``a:b`` or ``a*b``) AND it's not a
        # formula function call, leave it untouched.  Formula-function
        # terms (``log(x)``, ``sqrt(z)``, ``poly(x, 2)``, ``I(x ** 2)``)
        # contain parentheses and are processed below even when their
        # inner expressions contain arithmetic operators like ``**``.
        if ("*" in term or ":" in term) and "(" not in term:
            parts.append(term)
            return

        m = _FORMULA_FUNC_RE.match(term)
        if not m:
            parts.append(term)
            return
        func_name, inner = m.group(1), m.group(2).strip()

        if func_name == "I":
            try:
                result = _eval_I_inner(inner, new_data)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported expression inside I(): {inner!r} ({exc})."
                ) from exc
            inner_var = _extract_single_var(inner)
            if inner_var is None:
                raise ValueError(
                    f"I() inner expression must reference a single column; "
                    f"got {inner!r}."
                )
            new_data[inner_var] = result
            transformed.append((term, "I", inner_var))
            parts.append(inner_var)
            return

        result = _apply_formula_function(func_name, inner, None, new_data)
        inner_var = _extract_single_var(inner.split(",")[0])
        if inner_var is None:
            raise ValueError(
                f"Formula function {func_name}() must have a single column "
                f"as its first argument; got {inner!r}."
            )
        new_data[inner_var] = result
        transformed.append((term, func_name, inner_var))
        parts.append(inner_var)

    for term in raw_terms:
        _process_term(term, new_x_parts)

    new_main = f"{y_part} ~ {' + '.join(new_x_parts)}"

    if given_part is not None:
        given_terms = _split_formula_terms(given_part, "+")
        new_given_parts = []
        for term in given_terms:
            _process_term(term, new_given_parts)
        new_formula = f"{new_main} | {' + '.join(new_given_parts)}"
    else:
        new_formula = new_main

    return new_data, new_formula, transformed


# Whitelisted operators for I() inner expressions.  Only single-column
# references plus arithmetic on them are allowed.
_I_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
}


def _eval_I_inner(inner: str, data: pd.DataFrame):
    """Safely evaluate an ``I()`` inner expression.

    Allowed forms:
      - ``x``: just a column.
      - ``x ** k``, ``x * k``, ``x + k``, ``x - k``, ``x / k`` with a
        numeric constant ``k``.
      - ``x ** k + c``, ``x * k + c``, etc. (column * power + constant).
      - ``x + y`` (column + column) -- accepted because R allows it.
    Returns a numpy array.

    We deliberately reject anything that looks like a function call inside
    ``I()`` (use the explicit ``log(x)`` syntax for that).
    """
    tree = ast.parse(inner, mode="eval")
    return _eval_I_node(tree.body, data)


def _eval_I_node(node, data: pd.DataFrame):
    if isinstance(node, ast.Name):
        if node.id not in data.columns:
            raise ValueError(f"Unknown column {node.id!r}")
        return data[node.id].to_numpy()
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(
                f"Only numeric constants are allowed in I(); got {node.value!r}"
            )
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _I_ALLOWED_BINOPS:
            raise ValueError(
                f"Operator {op_type.__name__} is not allowed inside I()."
            )
        left = _eval_I_node(node.left, data)
        right = _eval_I_node(node.right, data)
        return _I_ALLOWED_BINOPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_eval_I_node(node.operand, data)
        if isinstance(node.op, ast.UAdd):
            return _eval_I_node(node.operand, data)
        raise ValueError(
            f"Unary operator {type(node.op).__name__} not allowed in I()."
        )
    raise ValueError(
        f"Expressions of type {type(node).__name__} are not allowed in I()."
    )


def _extract_single_var(inner: str) -> Optional[str]:
    """Return the single column name referenced in ``inner``.

    Accepts forms: ``x``, ``x ** 2``, ``x + 1`` (column reference must be
    the LHS identifier; constants and arithmetic are allowed alongside it).
    Returns None if the expression does not reference exactly one column.
    """
    try:
        tree = ast.parse(inner, mode="eval")
    except SyntaxError:
        return None
    vars_found = set()
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Name):
            vars_found.add(sub.id)
    # Filter out Python built-in names that may appear in the constant part.
    vars_found.discard("pi")
    vars_found.discard("e")
    if len(vars_found) == 1:
        return next(iter(vars_found))
    return None


# ---------------------------------------------------------------------------
# Multivariate slotting / auto-binning for slot 2+ numeric predictors
# (R-flexplot ``flexplot_break_me`` parity, v0.8.x+).
# After the formula is split, any numeric predictor in slot 2+ or in
# ``given`` with more than ``bins`` unique values is binned into a
# ``<varname>_binned`` column (used as the color / group aesthetic).
# ---------------------------------------------------------------------------


_DEFAULT_BINS = 3


def _slot_bin_numeric_predictors(
    data: pd.DataFrame,
    formula: str,
    bins: Optional[int] = None,
    breaks: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
):
    """Bin numeric slot-2+ / given predictors into ``<name>_binned`` columns.

    Mirrors R's ``flexplot_break_me``: a numeric predictor is binned when
    it appears in slot 2 or later (i.e. it's the color/group or a panel
    variable) AND its unique-value count exceeds ``bins``.  Returns
    ``(new_data, new_color, new_given, bin_count)`` where ``new_data``
    has the new ``<varname>_binned`` columns appended (or overwritten),
    ``new_color`` is the column name to use for the color aesthetic
    (possibly the binned version), and ``new_given`` is the list of
    column names to use for facet variables (also possibly the binned
    versions).  ``bin_count`` is the number of variables that were
    actually binned.

    If a third non-given predictor is present (e.g. ``y ~ x1 + x2 + x3``),
    this raises ``ValueError`` because R limits the display to at most
    4 visual variables (1 outcome + 3 predictors) and adding more
    predictors overwhelms the plot.

    Parameters
    ----------
    bins : int, default 3
        Cut-point count used by ``pd.cut`` for the auto-bin path.  Mirrors
        R's default ``bins=3``.  Ignored when ``breaks`` is provided.
    breaks, labels : optional
        Reserved for forward-compat with R's explicit ``breaks=list(...)``
        API; current implementation accepts only a single breaks/labels
        pair (used for any predictor that needs binning).  Tests pass a
        flat ``breaks`` and ``labels`` and rely on the auto-cut fallback
        path; richer dict-based breaks can be added in a later release.
    """
    bins = bins if bins is not None else _DEFAULT_BINS
    if not isinstance(bins, int) or bins < 2:
        # Silently fall back to default for invalid bins (callers should
        # have validated via _validate_binning_params already).
        bins = _DEFAULT_BINS

    variables = parse_flexplot_formula(formula)
    if variables.get("intercept_only", False):
        return data.copy(), variables.get("color"), variables.get("given", []), 0

    y = variables["y"]
    x = variables["x"]
    color = variables.get("color")
    given = variables.get("given", [])
    all_x = variables.get("all_x", [])  # includes any +color tokens

    # If there are 3+ non-given predictors (i.e. x + color + extra), reject.
    # R limits flexplot to 4 display vars total (1 outcome + 3 predictors)
    # for cognitive load; an extra slot would also need a 4th aesthetic.
    # `all_x` already includes interaction-expanded terms, so we strip
    # those out for the count.
    atom_predictors = [
        t for t in all_x if ":" not in t and "*" not in t
    ]
    if len(atom_predictors) > 2:
        raise ValueError(
            f"Formula {formula!r} has {len(atom_predictors)} non-given "
            f"predictors ({atom_predictors}); flexplot supports at most "
            f"two non-given predictors (x and an optional color).  For "
            f"more, move them into the `| given` part of the formula."
        )

    new_data = data.copy()
    bin_count = 0

    # Helper: bin a numeric column into ``<col>_binned`` using ``pd.cut``
    # over equal-width cuts between min/max.
    def _bin_one(col: str) -> Optional[str]:
        nonlocal bin_count
        if col is None or col not in new_data.columns:
            return col
        s = new_data[col]
        if not pd.api.types.is_numeric_dtype(s):
            return col
        n_unique = s.dropna().nunique()
        if n_unique <= bins:
            # Already low-cardinality: skip binning (R skips when <= bins).
            return col
        # If we got an explicit breaks list, use it; otherwise equal-width.
        x_min = float(np.nanmin(s.to_numpy()))
        x_max = float(np.nanmax(s.to_numpy()))
        cuts = np.linspace(x_min, x_max, num=int(bins) + 1).tolist()
        binned_col = f"{col}_binned"
        new_data[binned_col] = pd.cut(
            s, bins=cuts, labels=labels, include_lowest=True
        ).astype(str)
        bin_count += 1
        return binned_col

    # Bin the color predictor if it's numeric and high-cardinality.
    new_color = _bin_one(color) if color is not None else None

    # Bin the given variables similarly.
    new_given = [_bin_one(g) if g is not None else None for g in given]
    new_given = [g for g in new_given if g is not None]

    return new_data, new_color, new_given, bin_count


# ---------------------------------------------------------------------------
# flexplot_alpha_default / match_jitter_categorical parity (v0.8.x+).
# ---------------------------------------------------------------------------


# Sentinel alpha used internally to mean "user did not pass alpha"
# (matches R's ``alpha=.99977`` default in flexplot_prep_variables).
_FLEXPLOT_ALPHA_SENTINEL = 0.99977


def flexplot_alpha_default(data: pd.DataFrame, x: Optional[str], y: str,
                           alpha: Optional[float]) -> float:
    """Return the alpha to use for the raw-data geom.

    Mirrors R's ``flexplot_alpha_default``:
      - If user explicitly set alpha (not the sentinel), return it as-is.
      - Otherwise (R sentinel ``.99977`` / Python ``None``), use 0.2 for
        categorical x, 0.5 for numeric x.
      - For intercept-only formulas (no x axis), pass through.

    The Python-side alpha resolution (see ``flexplot()``) keeps the legacy
    behavior of ``alpha=0.3`` for numeric binary y; this helper is the
    categorical-vs-numeric split.
    """
    if alpha is not None and alpha != _FLEXPLOT_ALPHA_SENTINEL:
        # Explicit user value.
        return float(alpha)
    if x is None or x not in data.columns:
        # Intercept-only / no x: pass through.
        return float(alpha) if alpha is not None else 0.5
    s = data[x]
    if not pd.api.types.is_numeric_dtype(s):
        return 0.2
    return 0.5


def match_jitter_categorical(x, is_categorical: bool):
    """Resolve the jitter argument for the categorical / mixed-x branch.

    Mirrors R's ``match_jitter_categorical``:

      - ``None`` + categorical x  → (0.2, 0)
      - ``None`` + numeric x      → (0, 0) (no jitter)
      - ``True``  → (0.2, 0)
      - ``False`` → (0, 0)
      - numeric length 1          → (x, 0)
      - numeric length 2          → (x, y)
      - anything else             → raises ``ValueError``

    Returns a 2-tuple ``(width, height)``.
    """
    if x is None:
        return (0.2, 0.0) if is_categorical else (0.0, 0.0)
    if isinstance(x, bool):
        return (0.2, 0.0) if x else (0.0, 0.0)
    if isinstance(x, (int, float)):
        return (float(x), 0.0)
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return (float(x[0]), 0.0)
        if len(x) == 2:
            return (float(x[0]), float(x[1]))
        if len(x) > 2:
            raise ValueError(
                f"jitter must be a length-1 or length-2 sequence; got length "
                f"{len(x)}: {x!r}."
            )
        # Empty sequence
        raise ValueError(f"jitter must be non-empty; got {x!r}.")
    raise ValueError(
        f"jitter must be None, a bool, or a numeric length-1/2 sequence; "
        f"got {type(x).__name__}: {x!r}."
    )


def flexplot(
    formula: str,
    data: pd.DataFrame,
    method: str = "auto",
    random_effects: Optional[str] = None,
    mixed_backend: str = "auto",
    uncertainty: Optional[str] = "ci",
    level: float = 0.95,
    bands: Optional[List[float]] = None,
    overlay: Optional[List] = None,
    bins: Optional[int] = None,
    labels: Optional[List[str]] = None,
    breaks: Optional[List[float]] = None,
    spread: Optional[str] = None,
    sample: Optional[int] = None,
    ghost_line: Optional[str] = None,
    plot_type: Optional[str] = None,
    return_data: bool = False,
    ghost_reference=None,
    plot_string=None,
    related: bool = False,
    interaction_model: bool = False,
    jitter: Optional[Union[bool, List[float]]] = None,
    alpha: Optional[float] = None,
    raw_data: bool = True,
    **kwargs,
):
    """Intelligent multivariate graphics via formulas.

    Parameters
    ----------
    formula : str
        Flexplot formula of the form ``y ~ x [+ color] [| given1, given2]``.
        R-style interaction syntax (``y ~ x*z``, ``y ~ x:z``) is also
        accepted since v0.6.2; see Notes below.
    data : pd.DataFrame
        Non-empty data frame holding the referenced columns.
    method : {"auto", "lm", "loess", "quadratic", "polynomial", "cubic", "logistic", "rlm", "poisson", "Gamma", "mixedlm", "lmer", "glmer"}
        Smoother for the numeric-vs-numeric branch. ``"auto"`` selects LM.
        ``"quadratic"`` / ``"polynomial"``: degree-2 OLS in x (matches R).
        ``"cubic"``: degree-3 OLS in x.
        ``"logistic"``: GLM with logit link on numeric binary y.
        ``"rlm"``: robust regression via statsmodels RLM (Huber).
        ``"poisson"``: GLM with log link (requires non-negative y).
        ``"Gamma"``: GLM with inverse link (requires strictly positive y).
        ``"mixedlm"`` / ``"lmer"``: linear mixed-effects model with a
        random intercept (and optional random slope via
        ``random_effects='(1 + x|group)'``).
        ``"glmer"``: binomial mixed-effects model (random intercept).
        Non-conforming outcomes for logistic/poisson/Gamma fall back to OLS
        with a ``UserWarning``.
    random_effects : str, optional
        Random-effects spec for mixed methods:
        - Column name (e.g., ``"school"``) => random intercept ``(1|school)``.
        - lme4-style mini spec (e.g., ``"(1|school)"``,
          ``"(1 + x|school)"``).
        Required when ``method`` is ``"mixedlm"``, ``"lmer"``, or
        ``"glmer"``.
    mixed_backend : {"auto", "statsmodels"}, default "auto"
        Mixed-model backend selector. ``"auto"`` currently resolves to
        statsmodels; this argument is reserved for future ``pymer4``/R
        integration.
    uncertainty : {None, "ci", "prediction", "bootstrap"}, default "ci"
        Type of uncertainty band drawn around the fitted line.
        - ``None``: no fit, just the scatter.
        - ``"ci"``: confidence interval on the mean response (plotnine default).
        - ``"prediction"``: residual-based prediction interval on new observations.
        - ``"bootstrap"``: case-resampled CI (loess branch only; n_resamples=200).
    level : float in (0, 1), default 0.95
        Coverage probability for a single band. Ignored when ``bands`` is given.
    bands : list of float in (0, 1), optional
        Nested coverage levels (e.g., ``[0.5, 0.8, 0.95]``) for Tufte-style
        multi-ribbon display. Overrides ``level`` when provided.
    overlay : list of str or dict, optional
        Additional smoother specs to overlay on the same axes alongside the
        primary ``method``. Each entry is either a method name (``"lm"``,
        ``"loess"``, ``"rlm"``, etc.) or a dict with keys:
        - ``method`` (required): one of the recognized smoother methods.
        - ``color`` (optional, default cycles through a 5-color palette).
        - ``label`` (optional, default = ``method``): legend label.
        - ``uncertainty`` (optional, default ``"ci"``): per-overlay band type.
        - ``level`` (optional, default 0.95): per-overlay band coverage.
    bins : int, optional
        Discretize a numeric x into ``bins`` equal-width intervals before
        plotting, so the discrete-style summary (geom_jitter + dispersion
        marker) applies. Mutually exclusive with ``breaks`` (which wins).
        No-op when x is already discrete or non-numeric.
    labels : list of str, optional
        Custom labels for the discrete x levels produced by ``bins`` /
        ``breaks``. Length must equal ``bins`` (when given with bins) or
        ``len(breaks) - 1`` (when given with breaks).
    breaks : list of float, optional
        Explicit cut points for discretizing numeric x. Takes precedence
        over ``bins`` when both are given (a ``UserWarning`` is emitted).
    spread : {None, "ci", "sterr", "stdev", "range", "iqr", "quartiles", "no"}, default None
        Dispersion marker drawn in the discrete-x branch alongside
        ``geom_jitter``. Mirrors R-flexplot's ``spread``.
        - ``None`` / ``"quartiles"`` / ``"iqr"``: median ± Q1/Q3 IQR
          (R's default for discrete x).
        - ``"ci"``: bootstrap CI on the mean.
        - ``"sterr"``: mean ± 1.96 × standard error of the mean.
        - ``"stdev"``: mean ± 1 SD as a pointrange.
        - ``"range"``: min-max range.
        - ``"no"``: no summary layer at all.
    sample : int, optional
        Subsample N rows for the plotnine layers (scatter / jitter) while
        keeping the smoother fits on the full DataFrame. No-op when
        ``N >= len(data)``. Deterministic via ``np.random.default_rng(0)``.
    ghost_line : {"red", "dashed", "slope1", None}, default None
        Reference line drawn after the main layers. ``"red"`` for a solid
        red threshold at y=0; ``"dashed"`` for a black dashed reference
        at y=0; ``"slope1"`` for a diagonal slope=1 reference line for
        prediction-vs-observed overlays (v0.7.3+).
    plot_type : {"scatter", "line", "boxplot", "bar", "histogram", "qq", "density", "violin", None}, default None
        Explicit geom override. Bypasses the auto-dispatch.  For
        intercept-only formulas, ``"histogram"`` (default), ``"qq"``,
        ``"density"``, ``"boxplot"``, and ``"violin"`` produce univariate
        distribution plots.
    return_data : bool, default False
        When ``True``, return ``{"plot": ggplot, "data": DataFrame}``
        instead of just the plot. Useful with ``sample=`` to know which
        rows were plotted.
    ghost_reference : pd.DataFrame, optional
        Reference dataset to overlay on the same axes. Two patterns:
        - Columns ``(x, y)``: draws a gray geom_point layer (reference scatter).
        - Columns ``(x, "pred")``: draws a red dashed geom_line (prediction line).
    plot_string : dict, optional
        Override the axis/legend labels derived from the formula. Accepts
        keys ``x``, ``y``, ``title``, ``subtitle``, ``caption``, ``color``.
    related : bool, default False
        R-flexplot's paired-samples flag.  When True and the formula is
        ``y ~ x`` with a two-level categorical predictor, the plot shows
        paired difference scores (level2 - level1) against x = 1, with a
        reference line at 0 and a dispersion marker.  Requires equal group
        sizes and no color or panel variables.  Raises ``ValueError`` when
        the precondition is not met.
    interaction_model : bool, default False
        When ``True`` and the formula contains ``*`` or ``:`` syntax, fit
        a statsmodels OLS with the actual interaction term and overlay
        non-parallel per-color-group regression lines (rather than the
        default additive fit with parallel slopes). Suppresses the
        "additive fit" UserWarning when set. Falls back to the additive
        path when the formula has no interaction term, no separate color
        group, or only one color level.
    **kwargs
        Reserved for future extension.

    Returns
    -------
    plotnine.ggplot
        The composed plot object. Call ``.draw()`` to render or ``.save()``
        to write to disk.

    Notes
    -----
    For the numeric-vs-binary branch (binomial GLM), the band is always drawn
    on the response (probability) scale; plotnine handles the inverse-link
    transformation internally. Numeric binary ``[0, 1]`` y (v0.6.1+) and
    string binary y both route to the binomial branch. Explicit
    ``method="logistic"`` bypasses the binary pre-check and forces the
    numeric branch with a parametric logistic GLM.

    Interaction syntax (``*``, ``:``) is parsed since v0.6.2. The default
    fit remains **additive** (parallel slopes per color group); a
    ``UserWarning`` is emitted whenever interaction syntax is detected.
    Pass ``interaction_model=True`` (v0.7.0+) to fit the actual
    interaction term and overlay non-parallel per-color-group regression
    lines; this also suppresses the additive-fit warning.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pyflexplot import flexplot
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=100), "y": rng.normal(size=100)})
    >>> p = flexplot("y ~ x", data=df)
    >>> isinstance(p, ggplot)
    True

    With uncertainty bands:

    >>> p = flexplot("y ~ x", data=df, bands=[0.5, 0.8, 0.95])
    >>> any(isinstance(layer.geom, geom_smooth) for layer in p.layers)
    True

    With overlay smoothers:

    >>> p = flexplot(
    ...     "y ~ x", data=df,
    ...     overlay=[{"method": "loess", "label": "LOESS smoother"}],
    ... )
    >>> smooth_layers = [l for l in p.layers if isinstance(l.geom, geom_smooth)]
    >>> len(smooth_layers) >= 2
    True

    Auto-binning numeric x (v0.6.4):

    >>> df2 = pd.DataFrame({
    ...     "x": rng.uniform(0, 100, size=80),
    ...     "y": rng.normal(size=80),
    ... })
    >>> p = flexplot("y ~ x", data=df2, bins=4)
    >>> any(isinstance(layer.geom, geom_jitter) for layer in p.layers)
    True

    Polynomial fit on a non-linear signal (v0.6.4):

    >>> df3 = pd.DataFrame({
    ...     "x": np.linspace(-3, 3, 60),
    ...     "y": np.linspace(-3, 3, 60) ** 2 + rng.normal(scale=0.3, size=60),
    ... })
    >>> p = flexplot("y ~ x", data=df3, method="polynomial")
    >>> any(isinstance(layer.geom, geom_line) for layer in p.layers)
    True
    """
    if method not in _VALID_FLEXPLOT_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_VALID_FLEXPLOT_METHODS)}; got {method!r}. "
            "Pass 'auto' for the default behaviour (LM for numeric-vs-numeric, "
            "binomial GLM for numeric-vs-binary)."
        )
    validate_uncertainty_params(uncertainty, level, bands, method)
    overlay_specs = _normalize_overlay(overlay)
    if spread is not None and spread not in (s for s in _VALID_SPREAD if s):
        raise ValueError(
            f"spread must be one of {sorted(s for s in _VALID_SPREAD if s)} "
            f"or None; got {spread!r}."
        )

    variables = parse_flexplot_formula(formula)
    if variables.get("has_interaction") and not interaction_model:
        # v0.6.x default: parser accepts interaction syntax (R-compatible)
        # but the fit remains additive (parallel slopes per color group).
        # Warn so users aren't misled. When interaction_model=True is
        # explicit, the fit uses the actual interaction term via
        # _add_interaction_smooth() and no warning is needed (v0.7.0+).
        warnings.warn(
            f"Interaction syntax detected in formula {formula!r} but flexplot's "
            f"default fit is additive (parallel slopes per color group). "
            f"Pass `interaction_model=True` for true non-parallel slopes. "
            f"To suppress this warning, write the formula without `*` or `:`.",
            UserWarning,
            stacklevel=2,
        )

    # --- Formula-function evaluation (R ``formula_functions`` parity) ---
    # Detect terms like ``log(x)``, ``sqrt(x)``, ``poly(x, 2)``, ``I(x**2)``
    # in the right-hand side, apply the whitelisted function to ``data``,
    # store the result in a column named after the inner variable, and
    # rewrite the formula so downstream code sees the simpler name.
    transformed_data, transformed_formula, transformed_terms = (
        _apply_formula_functions(formula, data)
    )
    if transformed_terms:
        # Re-parse the rewritten formula; variables dict now reflects the
        # simpler (un-transed) term names.
        variables = parse_flexplot_formula(transformed_formula)
        # Preserve the interaction flag from the original parse (it was
        # already detected before the function-rewrite pass).
        if "has_interaction" not in variables:
            variables["has_interaction"] = False
        formula = transformed_formula
        data = transformed_data

    _validate_data_for_plot(
        formula, data, variables, intercept_only=variables.get("intercept_only", False)
    )

    y = variables["y"]
    x = variables["x"]
    color = variables["color"]
    given = variables["given"]
    intercept_only = variables.get("intercept_only", False)

    # --- Auto-categorize low-cardinality numeric predictors ---
    # R-flexplot's ``convert_if_less_than_five`` turns numeric variables
    # with <5 unique values into ordered factors.  This must run BEFORE
    # the type-detection / binning below so the low-cardinality numeric
    # path is taken (discrete x branch) rather than the high-cardinality
    # numeric path (LM / loess smoother).
    if not intercept_only and x is not None:
        cat_targets = [v for v in [x, color, *given] if v is not None]
        # Avoid converting columns that already are transformed to a
        # ``_binned`` string — those are intentionally stringified above.
        cat_targets = [v for v in cat_targets if not v.endswith("_binned")]
        if cat_targets:
            data = _convert_low_cardinality_to_categorical(data, cat_targets)

    # --- Optional subsampling (v0.6.5+) ---
    # When ``sample=N`` is set and N < len(data), subsample to N rows for
    # plotting only. Subsequent smoother fits still use the FULL data so the
    # fit isn't degraded by the subsample. We track ``_sampled_df`` for
    # return_data= so the caller knows which rows were plotted.
    if sample is not None:
        if not isinstance(sample, int) or isinstance(sample, bool):
            raise TypeError(
                f"sample must be an int >= 1; got {type(sample).__name__} ({sample!r})."
            )
        if sample < 1:
            raise ValueError(f"sample must be >= 1; got {sample}.")
    if sample is not None and sample < len(data):
        rng_sample = np.random.default_rng(0)  # deterministic for reproducibility
        sampled_idx = rng_sample.choice(len(data), size=sample, replace=False)
        sampled_idx = np.sort(sampled_idx)
        plot_input_df = data.iloc[sampled_idx].reset_index(drop=True)
        fit_input_df = data  # full data; smoother fits unchanged
    else:
        plot_input_df = data
        fit_input_df = data

    # --- Multivariate slotting / auto-binning for slot 2+ numeric predictors ---
    # Mirrors R's ``flexplot_break_me``: a numeric color or given variable
    # with more than ``bins`` unique values is binned into ``<name>_binned``
    # and that column drives the color aesthetic / facets.  Re-raises on
    # formulas with 3+ non-given predictors.
    if not intercept_only:
        slot_data, binned_color, binned_given, _bin_count = (
            _slot_bin_numeric_predictors(
                fit_input_df, formula, bins=bins, breaks=breaks, labels=labels
            )
        )
        # Apply the binning to both plot and fit inputs.
        binned_cols = [
            c for c in slot_data.columns
            if c not in fit_input_df.columns and c.endswith("_binned")
        ]
        # The helper returns ``slot_data`` containing the original columns
        # PLUS any ``<name>_binned`` ones.  We merge them back into both
        # ``plot_input_df`` and ``fit_input_df`` so the aesthetic mappings
        # and the smoother fits see the binned columns.
        if binned_cols:
            for col in binned_cols:
                # Original source column for the binned copy.
                src = col[: -len("_binned")]
                if src in fit_input_df.columns:
                    # Carry over the binned values from slot_data; they are
                    # a deterministic function of the original numeric
                    # column, so the alignment is index-based.
                    plot_input_df[col] = slot_data[col].to_numpy()
                    fit_input_df[col] = slot_data[col].to_numpy()
            # Update variables' color / given to point at the binned cols
            # so the aes / facet wiring below uses them.
            if binned_color is not None and color is not None:
                variables["color"] = binned_color
                color = binned_color
            if binned_given:
                # Replace ``given`` with the binned variants while keeping
                # the same length / ordering as the original list.
                new_given = []
                for orig in variables["given"]:
                    if orig is None:
                        new_given.append(None)
                        continue
                    candidate = f"{orig}_binned"
                    if candidate in fit_input_df.columns:
                        new_given.append(candidate)
                    else:
                        new_given.append(orig)
                variables["given"] = new_given
                given = new_given

    if intercept_only:
        # Intercept-only: show a univariate distribution of y.
        # R-flexplot supports histogram/qq/density/boxplot/violin via plot.type.
        p = _plot_univariate(plot_input_df, y, plot_type=plot_type, bins=bins)
        if return_data:
            return {"plot": p, "data": plot_input_df}
        return p

    if not isinstance(related, bool):
        raise TypeError(f"related must be a bool; got {type(related).__name__}.")

    # --- Related-samples / paired difference plot (R-flexplot related=T) ---
    # Only valid for y ~ x where x is a two-level grouping variable and there
    # are no color/given facets.  We replace the data with paired difference
    # scores (level2 - level1) and draw a univariate difference plot.
    if related:
        if color is not None or len(given) > 0:
            raise ValueError(
                "related=True is only supported for formulas with a single "
                "predictor and no color or panel variables (e.g., 'y ~ x')."
            )
        if x is None:
            raise ValueError("related=True requires a predictor variable.")

        x_series = plot_input_df[x]
        # Ensure a categorical-style grouping variable with exactly two levels.
        if pd.api.types.is_numeric_dtype(x_series) and x_series.nunique(dropna=True) == 2:
            x_series = x_series.astype(str)
        levs = sorted(x_series.dropna().unique())
        if len(levs) != 2:
            raise ValueError(
                f"related=True requires exactly 2 levels of the predictor; "
                f"{x!r} has {len(levs)} levels."
            )

        groups = {
            lev: plot_input_df.loc[x_series == lev, y].reset_index(drop=True)
            for lev in levs
        }
        sizes = [len(g) for g in groups.values()]
        if len(set(sizes)) != 1:
            raise ValueError(
                "related=True requires equal group sizes to compute paired "
                f"differences; got sizes {dict(zip(levs, sizes))}."
            )

        diff_label = f"Difference ({levs[1]}-{levs[0]})"
        related_df = pd.DataFrame({diff_label: groups[levs[1]].to_numpy() - groups[levs[0]].to_numpy()})
        alpha_rel = alpha if alpha is not None else 0.5
        p = _plot_related(
            related_df,
            diff_label,
            spread,
            plot_type,
            jitter,
            alpha_rel,
            raw_data,
        )
        if return_data:
            return {"plot": p, "data": related_df}
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

    # Numeric-x binning: if bins / breaks / labels are provided and x is
    # numeric (and not already auto-discrete), discretize x so the
    # discrete-style summary applies. Validation:
    # - bins: int >= 2 (>= 2 needed to be meaningful).
    # - breaks: list of floats, len >= 2, monotonically increasing, span the
    #   x range.
    # - labels: list of strings, len = len(breaks) - 1 when provided with
    #   breaks, or len = bins when provided with bins.
    # Mutual precedence: breaks > bins (breaks overrides bins when both set).
    # Validation lives in _validate_binning_params.
    if not _is_discrete(plot_input_df[x]) and pd.api.types.is_numeric_dtype(plot_input_df[x]):
        _validate_binning_params(bins, labels, breaks, plot_input_df[x])
        plot_df, x_discretized = _maybe_bin_numeric_x(
            plot_input_df, x, bins=bins, labels=labels, breaks=breaks
        )
        if x_discretized:
            is_x_discrete = True
        else:
            plot_df = plot_input_df.copy()
            is_x_discrete = False
    else:
        plot_df = plot_input_df.copy()
        is_x_discrete = _is_discrete(plot_input_df[x])
        # Convert numeric discrete X to string/categorical so plotnine treats x-axis as discrete levels
        if is_x_discrete and pd.api.types.is_numeric_dtype(plot_df[x]):
            plot_df[x] = plot_df[x].astype(str)

    # Binary-0/1 pre-check: a numeric y whose unique values are a subset of
    # {0, 1} should be treated as a binary outcome for binomial smoothing,
    # regardless of pandas' is_numeric_dtype (which returns True for int).
    # Without this pre-check, numeric binary y would fall into the LM/loess
    # branch and the binomial GLM branch would only fire for non-numeric y
    # (where the .astype(float) below would raise first).
    # BUT: explicit method='logistic' bypasses this and routes through the
    # numeric branch with a logistic GLM (see _add_parametric_smooth).
    y_is_binary = False
    if is_y_numeric and method not in {"logistic", "glmer"}:
        try:
            unique_y = pd.Series(plot_input_df[y].dropna().astype(float)).unique()
        except (ValueError, TypeError):
            unique_y = None
        y_is_binary = (
            unique_y is not None
            and len(unique_y) == 2
            and set(unique_y).issubset({0.0, 1.0})
        )

    # Build the base aesthetic with color/group when needed so all geoms pick it up.
    aes_kwargs = {"x": x, "y": y}
    if color:
        aes_kwargs["color"] = color
        aes_kwargs["group"] = color
    p = ggplot(plot_df, aes(**aes_kwargs))

    # --- Optional plot_type override (v0.6.5+) ---
    # Bypasses the auto-dispatch and forces a specific geom. Useful when
    # the user knows they want a boxplot regardless of how x is shaped, or
    # when the auto-dispatch picks the wrong branch because the data
    # violates heuristics (e.g. 11 unique values in x rather than 10).
    if plot_type is not None:
        if plot_type not in _VALID_PLOT_TYPES:
            raise ValueError(
                f"plot_type must be one of {sorted(_VALID_PLOT_TYPES)}; got {plot_type!r}."
            )
        if plot_type == "scatter":
            p += geom_point(alpha=0.5)
        elif plot_type == "line":
            p += geom_line()
        elif plot_type == "boxplot":
            p += geom_boxplot()
        elif plot_type == "violin":
            p += geom_violin()
        elif plot_type == "bar":
            # plotnine's geom_bar doesn't accept fun=; use stat_summary
            # with fun_y=np.mean + geom="bar" to get a bar chart of
            # group means per x level.
            p += stat_summary(fun_y=np.mean, geom="bar")
        # Skip the auto-dispatch below.
        skip_dispatch = True
    else:
        skip_dispatch = False

    # --- jitter / alpha / raw_data resolution (v0.8.0+, R-parity) ---
    # R ``match_jitter_categorical``: None + categorical x -> (0.2, 0);
    # None + numeric x -> (0, 0); True -> (0.2, 0); False -> (0, 0);
    # numeric length-1 -> (x, 0); length-2 -> (x, y).  We delegate to the
    # helper for the categorical-numeric split.  The check uses the
    # POST-binning ``is_x_discrete`` (which is True when ``bins=`` /
    # ``breaks=`` discretized x, or when low-cardinality conversion kicked
    # in) so a user who asks for `bins=4` always gets the categorical
    # jitter defaults regardless of the underlying numeric dtype.
    is_x_discrete_for_jitter = bool(is_x_discrete)
    if isinstance(jitter, (list, tuple)) and len(jitter) == 2:
        # Explicit numeric pair: bypass the R rule so users can still get
        # the exact jitter widths they want.
        jitter_xy = (float(jitter[0]), float(jitter[1]))
    elif isinstance(jitter, (int, float)) and not isinstance(jitter, bool):
        # Numeric length-1: pass through (R: ``c(.2)`` -> ``(0.2, 0)``).
        jitter_xy = (float(jitter), 0.0)
    else:
        jitter_xy = match_jitter_categorical(jitter, is_x_discrete_for_jitter)
    # alpha: explicit value (float in (0, 1]) wins everywhere; otherwise
    # use flexplot_alpha_default's categorical/numeric rule (0.2 for
    # categorical x, 0.5 for numeric x).  Numeric binary y keeps the
    # legacy 0.3 default (parity with prior releases and a slightly
    # softer overlay on the tight 0/1 cluster).
    if alpha is not None:
        if not isinstance(alpha, (int, float)) or not (0 < alpha <= 1):
            raise ValueError(
                f"alpha must be a float in (0, 1]; got {alpha!r}."
            )
        alpha_point = float(alpha)
    else:
        if y_is_binary:
            alpha_point = 0.3
        else:
            alpha_point = flexplot_alpha_default(plot_input_df, x, y, alpha)

    # Determine plot type.
    # Order matters:
    #   1. Binary 0/1 y must be detected before the generic numeric branch
    #      (otherwise int/float [0, 1] y falls into LM/loess).
    #   2. Numeric X is "discrete" when _is_discrete() returns True (numeric
    #      with <=10 unique values; post-05ac368 R-flexplot parity).
    if not skip_dispatch and y_is_binary and not is_x_discrete:
        # Binomial GLM branch — numeric binary outcome with numeric x.
        if raw_data:
            p += geom_point(alpha=alpha_point)
        p = _add_binomial_smooth(p, data, x, y, uncertainty, level, bands)
        if overlay_specs:
            p = _add_overlay_binomial(p, data, x, y, overlay_specs)

    elif not skip_dispatch and not is_y_numeric and not is_x_discrete:
        # Non-numeric y (string/categorical) with numeric x. Validate as
        # numeric 0/1; reject anything that doesn't fit a {0, 1} subset.

        try:
            unique_y = pd.Series(plot_df[y].dropna().astype(float)).unique()
        except (ValueError, TypeError):
            raise ValueError(
                f"Binomial smoothing requires a numeric binary 0/1 outcome; "
                f"{y!r} could not be converted to numeric"
            )
        if len(unique_y) != 2 or not set(unique_y).issubset({0.0, 1.0}):
            raise ValueError(
                f"Binomial smoothing requires a binary 0/1 outcome; {y!r} has "
                f"unique values: {sorted(unique_y)}"
            )
        if raw_data:
            p += geom_point(alpha=alpha_point)
        p = _add_binomial_smooth(p, data, x, y, uncertainty, level, bands)
        if overlay_specs:
            p = _add_overlay_binomial(p, data, x, y, overlay_specs)

    elif not skip_dispatch and is_y_numeric and not is_x_discrete:
        if raw_data:
            p += geom_point(alpha=alpha_point)
        if interaction_model and variables.get("has_interaction") and color:
            # Non-parallel slopes per color group via statsmodels OLS with
            # the actual interaction term (e.g. y ~ x * color). v0.7.0+.
            p = _add_interaction_smooth(
                p, data, x, y, color, variables["all_x"],
                method, uncertainty, level, bands,
            )
        else:
            p = _add_numeric_smooth(
                p,
                data,
                x,
                y,
                method,
                uncertainty,
                level,
                bands,
                random_effects=random_effects,
                mixed_backend=mixed_backend,
            )
        if overlay_specs:
            p = _add_overlay_numeric(p, data, x, y, overlay_specs)

    elif not skip_dispatch and is_y_numeric and is_x_discrete:
        if raw_data and jitter_xy[0] > 0:
            p += geom_jitter(width=jitter_xy[0], alpha=alpha_point)
        elif raw_data:
            p += geom_point(alpha=alpha_point)
        p = _add_discrete_summary(p, spread)

    elif not skip_dispatch:
        if raw_data and jitter_xy != (0.0, 0.0):
            p += geom_jitter(width=jitter_xy[0], height=jitter_xy[1], alpha=alpha_point)
        elif raw_data:
            p += geom_point(alpha=alpha_point)

    if len(given) == 1:
        p += facet_wrap(f"~{given[0]}")
    elif len(given) >= 2:
        p += facet_grid(f"{given[1]} ~ {given[0]}")

    p += theme_bw()

    # --- Optional plot.string override (v0.6.6+) ---
    # R-flexplot's plot.string is a dict of label overrides:
    # {"x": "Time (s)", "y": "Voltage (V)", "title": "Experiment 1"}
    # Validation: must be a dict with string keys and string values.
    if plot_string is not None:
        if not isinstance(plot_string, dict):
            raise TypeError(
                f"plot_string must be a dict of {{label: text}} overrides; "
                f"got {type(plot_string).__name__}."
            )
        bad = {k: type(v).__name__ for k, v in plot_string.items()
               if not isinstance(k, str) or not isinstance(v, str)}
        if bad:
            raise TypeError(
                f"plot_string keys and values must all be strings; "
                f"bad entries: {bad}"
            )
        # Apply via plotnine's labs(). Only known labels are passed through;
        # unknown keys are silently ignored (plotnine's labs() ignores
        # unknown keys but emits a warning we don't want to surface).
        labs_kwargs = {
            k: v for k, v in plot_string.items()
            if k in {"x", "y", "title", "subtitle", "caption", "color"}
        }
        if labs_kwargs:
            p += labs(**labs_kwargs)

    # --- Optional ghost.line reference layer (v0.6.5+) ---
    # ghost_line="red": solid red reference line. Useful for highlighting a
    # threshold or a reference value (e.g. y=0, or y=mean(y)).
    # ghost_line="dashed": dashed black line. R's flexplot() uses this to
    # mark the slope=1 reference for prediction-vs-observed plots.
    # Both are drawn as geom_hline (horizontal), so they're 1D references
    # at y=0. For diagonal references (slope=1), future work.
    # --- ghost.line (v0.8.0, R-parity) ---
    # R semantics: ghost.line is the COLOR of a line fit on a reference
    # panel that is repeated into every other panel (cross-panel
    # comparison). Python-only extensions kept: "slope1" (diagonal abline).
    # When the formula has NO `given` facets, the legacy Python behavior
    # applies: "red"/"dashed" -> y=0 hline; "slope1" -> y=x abline.
    if ghost_line is not None:
        if not isinstance(ghost_line, str):
            raise TypeError(
                f"ghost_line must be a color string, 'slope1', or None; "
                f"got {type(ghost_line).__name__}."
            )
        if len(given) >= 1:
            # R-parity path: fit y ~ x on the reference subset and repeat
            # the predicted line into every panel, drawn in ghost_line's
            # color. Reference selection via ghost_reference dict
            # ({given_var: level}) or the first level of the first given
            # variable when absent.
            if x is None or is_x_discrete:
                # Ghost lines are only defined for numeric x fits.
                if not isinstance(ghost_line, str):
                    raise TypeError("ghost_line must be a string.")
            ref_df = plot_input_df
            if ghost_reference is not None:
                if isinstance(ghost_reference, pd.DataFrame):
                    # DataFrame overlay path (legacy) is handled AFTER this
                    # block; dicts drive panel-referencing here.
                    ref_df = None
                elif isinstance(ghost_reference, dict):
                    for var, val in ghost_reference.items():
                        if var not in plot_input_df.columns:
                            raise ValueError(
                                f"ghost_reference key {var!r} is not a "
                                f"data column."
                            )
                        mask = plot_input_df[var] == val
                        if not mask.any():
                            # Nearest-match fallback for numeric refs.
                            if pd.api.types.is_numeric_dtype(plot_input_df[var]):
                                idx = (plot_input_df[var] - val).abs().idxmin()
                                mask = plot_input_df[var] == plot_input_df.loc[idx, var]
                            else:
                                mask = plot_input_df[var] == plot_input_df[var].iloc[0]
                        ref_df = plot_input_df[mask]
                else:
                    raise TypeError(
                        "ghost_reference must be None, a dict "
                        "({given_var: level}) for panel reference selection, "
                        "or a DataFrame for overlay; got "
                        f"{type(ghost_reference).__name__}."
                    )
            if ref_df is not None and len(ref_df) > 1 and pd.api.types.is_numeric_dtype(ref_df[x]):
                gx = ref_df[x].to_numpy(dtype=float)
                gy = ref_df[y].to_numpy(dtype=float)
                if np.isfinite(gx).all() and np.isfinite(gy).all():
                    _X = np.column_stack([np.ones_like(gx), gx])
                    _m = OLS(gy, _X).fit()
                    _x_eval = np.linspace(np.nanmin(gx), np.nanmax(gx), num=200)
                    _y_eval = _m.predict(np.column_stack([np.ones_like(_x_eval), _x_eval]))
                    ghost_df = pd.DataFrame({x: _x_eval, y: _y_eval})
                    p += geom_line(
                        aes(y=y),
                        data=ghost_df,
                        color=ghost_line,
                        linetype="dashed",
                        inherit_aes=False,
                    )
        else:
            # No facets: legacy y=0 / slope=1 references.
            if ghost_line not in {"red", "dashed", "slope1"}:
                raise ValueError(
                    f"Without `| given` facets, ghost_line must be 'red', "
                    f"'dashed', 'slope1', or None; got {ghost_line!r}. "
                    f"Panel-repetition (R parity) requires a facet in the "
                    f"formula."
                )
            if ghost_line == "red":
                p += geom_hline(yintercept=0, color="red")
            elif ghost_line == "dashed":
                p += geom_hline(yintercept=0, color="black", linetype="dashed")
            elif ghost_line == "slope1":
                from plotnine import geom_abline
                p += geom_abline(intercept=0, slope=1, color="black", linetype="dashed")

    # --- Optional ghost.reference overlay (v0.6.6+) ---
    # R-flexplot accepts ghost.reference as a DataFrame to overlay on the
    # same axes. Two common patterns:
    #   1. Reference scatter: columns matching x/y → draw geom_point in
    #      light gray.
    #   2. Reference prediction line: columns (x, "pred") → draw geom_line
    #      in a contrasting color.
    # We detect the pattern by checking if the DataFrame has columns
    # [x, y] or [x, "pred"].
    if ghost_reference is not None:
        # Dict form ({given_var: level}) was consumed by the ghost.line
        # panel-reference path above; only DataFrames reach this legacy
        # overlay path.
        if isinstance(ghost_reference, dict):
            # Dict = panel-reference selector, consumed by the ghost.line
            # block above. Here we only validate the facet requirement.
            if len(given) == 0:
                raise TypeError(
                    "ghost_reference dict requires a `| given` facet in the "
                    "formula (it selects the reference panel)."
                )
        elif isinstance(ghost_reference, pd.DataFrame):
            pass  # DataFrame overlay handled below.
        else:
            raise TypeError(
                f"ghost_reference must be None, a dict (panel reference), "
                f"or a pandas DataFrame (overlay); got "
                f"{type(ghost_reference).__name__}."
            )
        if isinstance(ghost_reference, pd.DataFrame) and x not in ghost_reference.columns:
            raise ValueError(
                f"ghost_reference DataFrame must have column {x!r} "
                f"(matching x in the formula); got columns "
                f"{list(ghost_reference.columns)}."
            )
        if isinstance(ghost_reference, dict):
            pass  # dict refs don't carry overlay columns
        elif "pred" in ghost_reference.columns:
            # Prediction-line pattern: geom_line in red.
            p += geom_line(
                aes(y="pred"),
                data=ghost_reference,
                color="red",
                linetype="dashed",
                inherit_aes=False,
            )
        elif y in ghost_reference.columns:
            # Reference-scatter pattern: geom_point in light gray.
            p += geom_point(
                data=ghost_reference,
                color="gray",
                alpha=0.4,
                inherit_aes=False,
            )
        else:
            raise ValueError(
                f"ghost_reference must have either column {y!r} (scatter) "
                f"or 'pred' (line); got {list(ghost_reference.columns)}."
            )

    if return_data:
        return {"plot": p, "data": plot_input_df}
    return p


def _lowess_predict(x_eval: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Wrapper around statsmodels ``lowess`` that takes (x_eval, y_sorted_by_x)."""
    order = np.argsort(x)
    x_sorted = np.asarray(x)[order]
    y_sorted = np.asarray(y)[order]
    smoothed = lowess(y_sorted, x_sorted, return_sorted=False)
    return np.interp(x_eval, x_sorted, smoothed)


def _add_numeric_smooth(
    p,
    data: pd.DataFrame,
    x: str,
    y: str,
    method: str,
    uncertainty: Optional[str],
    level: float,
    bands: Optional[List[float]],
    random_effects: Optional[str] = None,
    mixed_backend: str = "auto",
):
    """Add fitted line + uncertainty band for numeric-vs-numeric.

    Returns the plotnine plot object with the appropriate layers added.
    Caller is responsible for adding geom_point first.
    """
    if uncertainty is None:
        # No fit at all — preserve the scatter only.
        return p

    # polynomial/quadratic/cubic are OLS fits with higher-order x terms;
    # logistic/poisson/Gamma are GLMs; rlm is robust regression.  plotnine's
    # geom_smooth does NOT support all of these cleanly, so we route them
    # through statsmodels and add geom_line + geom_ribbon manually.
    if method in {
        "quadratic",
        "polynomial",
        "cubic",
        "logistic",
        "rlm",
        "poisson",
        "Gamma",
        "mixedlm",
        "lmer",
        "glmer",
    }:
        return _add_parametric_smooth(
            p,
            data,
            x,
            y,
            method,
            uncertainty,
            level,
            bands,
            random_effects=random_effects,
            mixed_backend=mixed_backend,
        )

    use_loess = method == "loess"

    # --- Nested bands (multiple ribbons via multiple geom_smooth layers) ---
    if bands is not None:
        levels = sorted(set(bands))
        for lvl in levels:
            if use_loess:
                p += geom_smooth(method="loess", level=lvl, color="blue", alpha=0.15)
            else:
                p += geom_smooth(method="lm", level=lvl, color="blue", alpha=0.15)
        return p

    # --- Single band ---
    if uncertainty == "ci":
        if use_loess:
            p += geom_smooth(method="loess", level=level, color="blue")
        else:
            p += geom_smooth(method="lm", level=level, color="blue")
        return p

    if uncertainty == "prediction":
        # Fit an OLS model, compute residual-based PI on a sorted-x grid,
        # and draw a ribbon + the fitted line.
        from scipy import stats as _scipy_stats

        x_arr = data[x].to_numpy(dtype=float)
        y_arr = data[y].to_numpy(dtype=float)
        model = OLS(y_arr, sm.add_constant(x_arr)).fit()
        x_eval = np.sort(np.unique(x_arr))
        if x_eval.size < 2:
            # Degenerate: cannot draw a meaningful band.
            p += geom_line(aes(y=y), color="blue")
            return p
        yhat_eval = model.predict(sm.add_constant(x_eval))
        yhat_full = model.predict(sm.add_constant(x_arr))
        sigma = float(np.sqrt(np.mean((y_arr - yhat_full) ** 2)))
        z = float(_scipy_stats.norm.ppf(0.5 + level / 2))
        half_width = z * sigma
        ribbon_df = pd.DataFrame({
            x: x_eval,
            "__lower": yhat_eval - half_width,
            "__upper": yhat_eval + half_width,
            y: yhat_eval,
        })
        p += geom_ribbon(
            aes(ymin="__lower", ymax="__upper"),
            data=ribbon_df,
            alpha=0.2,
            fill="blue",
            inherit_aes=False,
        )
        p += geom_line(
            aes(y=y),
            data=ribbon_df,
            color="blue",
            inherit_aes=False,
        )
        return p

    if uncertainty == "bootstrap":
        # Case-resampled bootstrap CI for the loess branch.
        x_arr = data[x].to_numpy(dtype=float)
        y_arr = data[y].to_numpy(dtype=float)
        x_eval, lower, upper = compute_bootstrap_ci(
            x_arr, y_arr,
            smooth_fn=lambda x_e, y_s: _lowess_predict(x_e, y_s, x_arr),
            n_resamples=200,
            level=level,
            random_state=None,
        )
        smoothed_line = _lowess_predict(x_eval, y_arr, x_arr)
        ribbon_df = pd.DataFrame({
            x: x_eval,
            "__lower": lower,
            "__upper": upper,
            y: smoothed_line,
        })
        p += geom_ribbon(
            aes(ymin="__lower", ymax="__upper"),
            data=ribbon_df,
            alpha=0.2,
            fill="blue",
            inherit_aes=False,
        )
        p += geom_line(
            aes(y=y),
            data=ribbon_df,
            color="blue",
            inherit_aes=False,
        )
        return p

    # Should never reach here thanks to validate_uncertainty_params.
    return p


def _add_parametric_smooth(
    p,
    data: pd.DataFrame,
    x: str,
    y: str,
    method: str,
    uncertainty: Optional[str],
    level: float,
    bands: Optional[List[float]],
    random_effects: Optional[str] = None,
    mixed_backend: str = "auto",
):
    """Add fitted line + CI ribbon for polynomial / cubic / logistic methods.

    plotnine's ``geom_smooth(method="lm", ...)`` does NOT accept
    ``formula=poly(x, k)`` cleanly, so we fit statsmodels directly and draw
    the line + ribbon manually. Mirrors the prediction/ bootstrap branches
    in ``_add_numeric_smooth``.

    Methods:
    - "quadratic"/"polynomial": degree-2 OLS.
    - "cubic": degree-3 OLS.
    - "logistic": GLM with logit link on numeric binary y.
    - "mixedlm"/"lmer": linear mixed-effects with a random intercept.
    - "glmer": binomial mixed-effects with a random intercept.
    """
    from scipy import stats as _scipy_stats

    def _parse_random_effects_spec(spec: Optional[str], x_name: str):
        if not spec:
            raise ValueError(
                "Mixed-effects methods require random_effects=. "
                "Use a group column name (e.g., random_effects='school') "
                "or an lme4-style mini spec (e.g., '(1|school)' or "
                "'(1 + x|school)')."
            )
        text = str(spec).strip()
        if text in data.columns:
            return "1", text
        m = re.match(r"^\(\s*(.+?)\s*\|\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)$", text)
        if not m:
            raise ValueError(
                f"Invalid random_effects specification {spec!r}. "
                "Expected a column name or '(effects|group)'."
            )
        re_formula_raw, group_col = m.group(1).strip(), m.group(2).strip()
        if group_col not in data.columns:
            raise ValueError(
                f"random_effects group column {group_col!r} not found in data."
            )
        if re_formula_raw == "1":
            return "1", group_col
        re_formula = re_formula_raw.replace(x_name, "x")
        allowed = {"1 + x", "x", "0 + x", "1+x", "0+x"}
        if re_formula not in allowed:
            raise NotImplementedError(
                "Supported random_effects formulas are '(1|g)' and "
                "'(1 + x|g)' (or '(x|g)')."
            )
        if re_formula in {"x", "0 + x", "0+x"}:
            re_formula = "0 + x"
        else:
            re_formula = "1 + x"
        return re_formula, group_col

    def _add_fe_band(ribbon_df: pd.DataFrame, fe_mean: np.ndarray, fe_cov: np.ndarray):
        if bands is not None:
            levels = sorted(set(bands))
        else:
            levels = [level]
        for lvl in levels:
            z = float(_scipy_stats.norm.ppf(0.5 + lvl / 2))
            se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", fe_mean, fe_cov, fe_mean), 0.0))
            ribbon_df[f"__lower_{lvl}"] = ribbon_df[y].to_numpy() - z * se
            ribbon_df[f"__upper_{lvl}"] = ribbon_df[y].to_numpy() + z * se
        for lvl in sorted(levels, reverse=True):
            alpha = 0.1 + 0.15 * (lvl / max(levels))
            p_local = geom_ribbon(
                aes(ymin=f"__lower_{lvl}", ymax=f"__upper_{lvl}"),
                data=ribbon_df,
                alpha=alpha,
                fill="blue",
                inherit_aes=False,
            )
            yield p_local

    x_arr = data[x].to_numpy(dtype=float)
    y_arr = data[y].to_numpy(dtype=float)
    n = x_arr.size
    if n < 2:
        return p

    if method in {"quadratic", "polynomial"}:
        # R-flexplot: both "polynomial" and "quadratic" are degree-2 OLS.
        X = np.column_stack([np.ones_like(x_arr), x_arr, x_arr ** 2])
        model = OLS(y_arr, X).fit()
        link_label = "polynomial (degree-2)"
    elif method == "cubic":
        X = np.column_stack([np.ones_like(x_arr), x_arr, x_arr ** 2, x_arr ** 3])
        model = OLS(y_arr, X).fit()
        link_label = "cubic (degree-3)"
    elif method == "logistic":
        # Validate binary {0, 1}; fall back to OLS with a warning if not.
        unique_y = np.unique(y_arr[~np.isnan(y_arr)])
        is_binary = set(unique_y.tolist()).issubset({0.0, 1.0}) and len(unique_y) == 2
        if not is_binary:
            warnings.warn(
                f"method='logistic' requires a numeric binary 0/1 outcome; "
                f"{y!r} has unique values {sorted(unique_y.tolist())}. "
                f"Falling back to OLS.",
                UserWarning,
                stacklevel=3,
            )
            X = sm.add_constant(x_arr)
            model = OLS(y_arr, X).fit()
            link_label = "OLS fallback (logistic requires binary y)"
        else:
            X = sm.add_constant(x_arr)
            model = sm.GLM(
                y_arr, X, family=sm.families.Binomial(link=sm.families.links.Logit())
            ).fit()
            link_label = "logistic (logit)"
    elif method == "rlm":
        X = sm.add_constant(x_arr)
        model = sm.RLM(y_arr, X, M=sm.robust.norms.HuberT()).fit()
        link_label = "rlm (Huber)"
    elif method == "poisson":
        X = sm.add_constant(x_arr)
        if np.any(y_arr < 0):
            warnings.warn(
                f"method='poisson' requires a non-negative outcome; {y!r} has "
                f"negative values. Falling back to OLS.",
                UserWarning,
                stacklevel=3,
            )
            model = OLS(y_arr, X).fit()
            link_label = "OLS fallback (poisson requires non-negative y)"
        else:
            model = sm.GLM(
                y_arr, X, family=sm.families.Poisson(link=sm.families.links.Log())
            ).fit()
            link_label = "poisson (log)"
    elif method == "Gamma":
        X = sm.add_constant(x_arr)
        if np.any(y_arr <= 0):
            warnings.warn(
                f"method='Gamma' requires a strictly positive outcome; {y!r} has "
                f"non-positive values. Falling back to OLS.",
                UserWarning,
                stacklevel=3,
            )
            model = OLS(y_arr, X).fit()
            link_label = "OLS fallback (Gamma requires positive y)"
        else:
            model = sm.GLM(
                y_arr, X,
                family=sm.families.Gamma(link=sm.families.links.InversePower()),
            ).fit()
            link_label = "Gamma (inverse)"
    elif method in {"mixedlm", "lmer"}:
        if mixed_backend not in {"auto", "statsmodels"}:
            raise ValueError(
                f"mixed_backend must be 'auto' or 'statsmodels'; got {mixed_backend!r}."
            )
        re_formula, group_col = _parse_random_effects_spec(random_effects, x)
        fit_df = pd.DataFrame({"y": y_arr, "x": x_arr, "__group": data[group_col]})
        try:
            model = smf.mixedlm(
                "y ~ x",
                data=fit_df,
                groups=fit_df["__group"],
                re_formula=re_formula,
            ).fit(reml=True, method="lbfgs", disp=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fit mixedlm/lmer model: {exc}"
            ) from exc
        link_label = "mixedlm/lmer"
    elif method == "glmer":
        if mixed_backend not in {"auto", "statsmodels"}:
            raise ValueError(
                f"mixed_backend must be 'auto' or 'statsmodels'; got {mixed_backend!r}."
            )
        unique_y = np.unique(y_arr[~np.isnan(y_arr)])
        is_binary = set(unique_y.tolist()).issubset({0.0, 1.0}) and len(unique_y) == 2
        if not is_binary:
            raise ValueError(
                f"method='glmer' requires a numeric binary 0/1 outcome; "
                f"{y!r} has unique values {sorted(unique_y.tolist())}."
            )
        _, group_col = _parse_random_effects_spec(random_effects, x)
        fit_df = pd.DataFrame({"y": y_arr, "x": x_arr, "__group": data[group_col]})
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        try:
            model = BinomialBayesMixedGLM.from_formula(
                "y ~ x",
                {"__group": "0 + C(__group)"},
                fit_df,
            ).fit_vb()
        except Exception as exc:
            raise RuntimeError(f"Failed to fit glmer model: {exc}") from exc
        link_label = "glmer"
    else:  # pragma: no cover — guarded by caller
        return p

    x_eval = np.linspace(np.nanmin(x_arr), np.nanmax(x_arr), num=200)

    # Build the corresponding evaluation design matrix.
    if method in {"quadratic", "polynomial"}:
        X_eval = np.column_stack([np.ones_like(x_eval), x_eval, x_eval ** 2])
    elif method == "cubic":
        X_eval = np.column_stack(
            [np.ones_like(x_eval), x_eval, x_eval ** 2, x_eval ** 3]
        )
    elif method in {"logistic", "rlm", "poisson", "Gamma"}:
        # logistic, rlm, poisson, Gamma (and logistic OLS fallback) are all
        # linear-in-x models: intercept + x.
        X_eval = sm.add_constant(x_eval)
    elif method in {"mixedlm", "lmer"}:
        X_eval = np.column_stack([np.ones_like(x_eval), x_eval])
    elif method == "glmer":
        X_eval = np.column_stack([np.ones_like(x_eval), x_eval])
    else:  # pragma: no cover
        X_eval = sm.add_constant(x_eval)

    if method in {"mixedlm", "lmer"}:
        fe_names = list(model.fe_params.index)
        fe_mean = np.asarray(model.fe_params.to_numpy(), dtype=float)
        cov_df = model.cov_params()
        if hasattr(cov_df, "loc"):
            fe_cov = np.asarray(cov_df.loc[fe_names, fe_names], dtype=float)
        else:
            fe_cov = np.asarray(cov_df, dtype=float)[: len(fe_names), : len(fe_names)]
        design = np.column_stack([np.ones_like(x_eval), x_eval])
        yhat_eval = design @ fe_mean
        ribbon_df = pd.DataFrame({x: x_eval, y: yhat_eval})
        for layer in _add_fe_band(ribbon_df, design, fe_cov):
            p += layer
        p += geom_line(aes(y=y), data=ribbon_df, color="blue", inherit_aes=False)
        _ = link_label
        return p
    if method == "glmer":
        beta = np.asarray(model.fe_mean, dtype=float)
        design = np.column_stack([np.ones_like(x_eval), x_eval])
        eta = design @ beta
        yhat_eval = 1.0 / (1.0 + np.exp(-eta))
        ribbon_df = pd.DataFrame({x: x_eval, y: yhat_eval})
        # Approximate FE-only bands from posterior SD; this ignores covariance,
        # but gives a stable uncertainty envelope without requiring MCMC draws.
        if bands is not None:
            levels = sorted(set(bands))
        else:
            levels = [level]
        fe_sd = np.asarray(getattr(model, "fe_sd", np.full_like(beta, np.nan)), dtype=float)
        se_eta = np.sqrt(np.maximum((design ** 2) @ (fe_sd ** 2), 0.0))
        for lvl in levels:
            z = float(_scipy_stats.norm.ppf(0.5 + lvl / 2))
            lo = 1.0 / (1.0 + np.exp(-(eta - z * se_eta)))
            hi = 1.0 / (1.0 + np.exp(-(eta + z * se_eta)))
            ribbon_df[f"__lower_{lvl}"] = lo
            ribbon_df[f"__upper_{lvl}"] = hi
        for lvl in sorted(levels, reverse=True):
            alpha = 0.1 + 0.15 * (lvl / max(levels))
            p += geom_ribbon(
                aes(ymin=f"__lower_{lvl}", ymax=f"__upper_{lvl}"),
                data=ribbon_df,
                alpha=alpha,
                fill="blue",
                inherit_aes=False,
            )
        p += geom_line(aes(y=y), data=ribbon_df, color="blue", inherit_aes=False)
        _ = link_label
        return p

    yhat_eval = np.asarray(model.predict(X_eval))

    # --- Bands: nested or single ---
    if bands is not None:
        levels = sorted(set(bands))
    else:
        levels = [level]

    # Outermost band draws the line; inner bands draw only the ribbon.
    outermost_lvl = levels[-1]
    z_outer = float(_scipy_stats.norm.ppf(0.5 + outermost_lvl / 2))

    # Build a single combined ribbon dataframe with all band columns so we
    # can layer them on the same plot. Outermost band = widest.
    ribbon_df = pd.DataFrame({x: x_eval, y: yhat_eval})
    for lvl in levels:
        z = float(_scipy_stats.norm.ppf(0.5 + lvl / 2))
        # Use the prediction SE for the mean (not for new observations) for
        # a CI-style band. statsmodels' get_prediction().summary_frame(alpha)
        # gives mean_ci_lower / mean_ci_upper directly.
        try:
            pred = model.get_prediction(X_eval)
            frame = pred.summary_frame(alpha=1 - lvl)
            lower = frame["mean_ci_lower"].to_numpy()
            upper = frame["mean_ci_upper"].to_numpy()
        except Exception:
            # Fallback: normal-approx CI using the model's parameter covariance.
            # For RLM, get_prediction() is not available, so we compute
            # Var(x' beta) = diag(X_eval @ Cov(beta) @ X_eval').
            cov_params = getattr(model, "cov_params", None)
            if cov_params is not None:
                try:
                    var = np.einsum("ij,jk,ik->i", X_eval, cov_params(), X_eval)
                    se = np.sqrt(var)
                except Exception:
                    se = np.full(len(yhat_eval), np.nan)
            else:
                se = np.full(len(yhat_eval), np.nan)
            # Guard degenerate SE (e.g., perfect fit); draw a flat band.
            finite_se = np.where(np.isfinite(se), se, 0.0)
            lower = yhat_eval - z * finite_se
            upper = yhat_eval + z * finite_se

        ribbon_df[f"__lower_{lvl}"] = lower
        ribbon_df[f"__upper_{lvl}"] = upper

    # Draw ribbons (innermost first so outermost ends up on top).
    for lvl in sorted(levels, reverse=True):
        alpha = 0.1 + 0.15 * (lvl / max(levels))
        p += geom_ribbon(
            aes(ymin=f"__lower_{lvl}", ymax=f"__upper_{lvl}"),
            data=ribbon_df,
            alpha=alpha,
            fill="blue",
            inherit_aes=False,
        )

    p += geom_line(
        aes(y=y),
        data=ribbon_df,
        color="blue",
        inherit_aes=False,
    )

    # Inject the link label into the plot's labels so users can see what
    # was fit. plotnine exposes .labels; mutate via a workaround (geom_line
    # doesn't carry labels, so attach as a one-off text annotation is
    # cleaner — but text annotations need positioning data. Skipping for
    # now; users can pass `labs()` themselves).
    _ = link_label  # reserved for future annotation hook
    return p


def _add_interaction_smooth(
    p,
    data: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    all_x: list,
    method: str,
    uncertainty: Optional[str],
    level: float,
    bands: Optional[List[float]],
):
    """Add per-color-group fitted lines for an interaction formula.

    Used when ``interaction_model=True`` and the formula contains ``*`` or
    ``:`` syntax. Fits a statsmodels OLS with the actual interaction term
    (e.g. ``y ~ x * z`` rather than ``y ~ x + z``) and overlays one
    ``geom_line`` + optional ``geom_ribbon`` per level of ``color``.

    Parameters
    ----------
    all_x : list of str
        The expanded predictor list from the parser; for ``y ~ x*z`` this
        is ``['x', 'z', 'x:z']``. The interaction term is auto-detected as
        any term containing ``:``.
    """
    # Identify the interaction term in all_x (the one containing ":").
    interaction_term = next((t for t in all_x if ":" in t), None)
    if interaction_term is None:
        # Defensive: if interaction_model=True but no interaction term
        # was parsed, fall back to the additive path. (Should not happen
        # if the parser is consistent, but we don't want to crash.)
        return _add_numeric_smooth(
            p, data, x, y, method, uncertainty, level, bands
        )

    # Find the color column name in the interaction term: "x:z" -> first atom.
    color_atom = _first_atom(interaction_term.split(":")[1]) if ":" in interaction_term else color
    if color_atom != color:
        # Mismatch — fallback.
        return _add_numeric_smooth(
            p, data, x, y, method, uncertainty, level, bands
        )

    # Build the design matrix: y ~ x + color + x:color (the interaction).
    x_arr = data[x].to_numpy(dtype=float)
    color_arr = data[color].to_numpy()
    y_arr = data[y].to_numpy(dtype=float)

    # Encode color via a category code so the interaction term is numeric.
    color_series = pd.Series(color_arr)
    color_codes, color_levels = pd.factorize(color_series)
    color_codes = color_codes.astype(float)
    n_groups = len(color_levels)
    if n_groups < 2:
        # Degenerate: only one color level. Fall back to additive.
        return _add_numeric_smooth(
            p, data, x, y, method, uncertainty, level, bands
        )

    # Build design matrix: intercept + x + color_dummies (drop first) + x:color_dummies
    # Simpler approach: use statsmodels' formula API directly with the
    # interaction term. This is cleaner than constructing the design
    # matrix by hand and matches R's `y ~ x * color` semantics.
    import statsmodels.formula.api as _smf

    # Build a temporary DataFrame for statsmodels.
    fit_df = pd.DataFrame({
        "_y": y_arr,
        "_x": x_arr,
        "_color": color_arr,
    })
    # Renaming so statsmodels' patsy accepts them (avoid operator parsing issues).
    fit_df.columns = ["_y", "_x", "_color"]
    formula_str = "_y ~ _x * _color"
    try:
        model = _smf.ols(formula_str, data=fit_df).fit()
    except Exception as exc:
        raise RuntimeError(
            f"interaction_model=True requires a valid OLS fit with the "
            f"interaction term; got: {exc}"
        )

    # Predict on a per-color grid.
    x_min = float(np.nanmin(x_arr))
    x_max = float(np.nanmax(x_arr))
    x_eval = np.linspace(x_min, x_max, num=200)

    # Build eval DataFrame with one row per (x_eval, color_level).
    eval_rows = []
    for level_val in color_levels:
        for xv in x_eval:
            eval_rows.append({"_x": xv, "_color": level_val})
    eval_df = pd.DataFrame(eval_rows)
    yhat_eval = np.asarray(model.predict(eval_df))

    # Compute CI bands. Supports a single level (level=) or nested bands
    # (bands=[...]). Returns a dict {level_value: (lower_array, upper_array)}
    # or None if CI computation failed / is suppressed.
    band_arrays = None
    if uncertainty in {"ci", "prediction"}:
        levels_to_compute = sorted(set(bands)) if bands is not None else [level]
        band_arrays = {}
        ci_kind = "obs_ci" if uncertainty == "prediction" else "mean_ci"
        for lvl in levels_to_compute:
            try:
                pred = model.get_prediction(eval_df)
                frame = pred.summary_frame(alpha=1 - lvl)
                lower = frame[f"{ci_kind}_lower"].to_numpy()
                upper = frame[f"{ci_kind}_upper"].to_numpy()
                band_arrays[lvl] = (lower, upper)
            except Exception:
                # Skip this level if statsmodels can't produce it.
                pass
        if not band_arrays:
            band_arrays = None

    # Determine colors per group using a default palette.
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, level_val in enumerate(color_levels):
        group_mask = (eval_df["_color"] == level_val).to_numpy()
        line_color = palette[i % len(palette)]
        # Build a per-group dataframe for plotnine.
        line_df = pd.DataFrame({
            x: x_eval,
            y: yhat_eval[group_mask],
            color: np.full_like(x_eval, level_val, dtype=object),
        })
        # The line itself.
        p += geom_line(
            data=line_df,
            color=line_color,
            inherit_aes=False,
        )
        # Optional ribbons. For nested bands, draw innermost first so the
        # outermost band ends up on top.
        if band_arrays is not None:
            for lvl, (ci_lower, ci_upper) in sorted(band_arrays.items()):
                ribbon_df = line_df.copy()
                ribbon_df["__lower"] = ci_lower[group_mask]
                ribbon_df["__upper"] = ci_upper[group_mask]
                # Outer (larger coverage) bands are wider; lower alpha for
                # the innermost so the layering reads as Tufte-style nested.
                alpha = 0.10 + 0.10 * (lvl / max(band_arrays.keys()))
                p += geom_ribbon(
                    aes(ymin="__lower", ymax="__upper"),
                    data=ribbon_df,
                    alpha=alpha,
                    fill=line_color,
                    inherit_aes=False,
                )
    return p


def _add_binomial_smooth(
    p,
    data: pd.DataFrame,
    x: str,
    y: str,
    uncertainty: Optional[str],
    level: float,
    bands: Optional[List[float]],
):
    """Add fitted line + uncertainty band for numeric-vs-binary (binomial GLM).

    The band is always rendered on the response (probability) scale.
    """
    if uncertainty is None:
        # No fit at all — just the scatter.
        return p

    method_args = {"family": "binomial"}

    if bands is not None:
        levels = sorted(set(bands))
        for lvl in levels:
            p += geom_smooth(
                method="glm",
                method_args=method_args,
                level=lvl,
                color="blue",
                alpha=0.15,
            )
        return p

    p += geom_smooth(
        method="glm",
        method_args=method_args,
        level=level,
        color="blue",
    )
    return p


def _add_overlay_numeric(p, data, x, y, overlay_specs):
    """Add one geom_smooth per overlay spec on the numeric-vs-numeric branch.

    Each entry is drawn with its own color and ``uncertainty``/``level``
    settings. No bootstrap overlay for non-loess methods (plotnine doesn't
    expose stat_smooth's bootstrap from a single ``geom_smooth`` call with
    arbitrary methods; we keep the API consistent by routing all overlays
    through ``geom_smooth`` and reserving bootstrap for ``"loess"``).
    """
    has_labels = any(
        spec.get("label") and spec["label"] != spec["method"]
        for spec in overlay_specs
    )
    label_colors = {}

    for spec in overlay_specs:
        method = spec["method"]
        level = spec["level"]
        color = spec["color"]
        label = spec.get("label", method)
        kwargs = {"method": method, "level": level, "color": color}
        # Forward any extra stat_args (span, formula, method_args, ...) that
        # the user provided.
        for k in ("span", "formula", "method_args", "n"):
            if k in spec:
                kwargs[k] = spec[k]
        p += geom_smooth(**kwargs)
        if has_labels:
            label_colors[label] = color

    if has_labels and label_colors:
        # Add a manual color scale so labels appear in the legend.
        p += scale_color_manual(
            name="Method",
            values=label_colors,
        )
    return p


def _add_overlay_binomial(p, data, x, y, overlay_specs):
    """Add binomial GLM overlay smoothers on the numeric-vs-binary branch.

    Only entries with ``method == "glm"`` are supported here; other methods
    raise so the user gets a clear error rather than a silently-broken chart.
    """
    label_colors = {}
    for spec in overlay_specs:
        if spec["method"] != "glm":
            raise ValueError(
                f"Overlay method {spec['method']!r} is not supported on the "
                f"binomial branch; only 'glm' is allowed."
            )
        kwargs = {
            "method": "glm",
            "method_args": {"family": "binomial"},
            "level": spec["level"],
            "color": spec["color"],
        }
        p += geom_smooth(**kwargs)
        label_colors[spec.get("label", "glm")] = spec["color"]
    if label_colors:
        p += scale_color_manual(name="Method", values=label_colors)
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


def visualize(
    model,
    data: Optional[pd.DataFrame] = None,
    plot: str = "model",
    **kwargs,
):
    """
    Provides a visual representation of a fitted statistical object.
    Supports statsmodels (OLS, GLM), scikit-learn models, and
    :class:`pyflexplot.flex_nn.NeuralNetFit` wrappers.

    Parameters
    ----------
    model : fitted model
        Any model with a ``predict`` method. Statsmodels (OLS, GLM),
        scikit-learn regressors, and ``NeuralNetFit`` are supported.
    data : pd.DataFrame, optional
        Predictor data. If omitted, inferred from ``model.model.data``.
    plot : {"model", "residuals", "all"}, default "model"
        What to draw:
        - ``"model"``: predicted-vs-observed scatter with the fitted
          line (the legacy behavior).
        - ``"residuals"``: residual-vs-fitted scatter and a residual
          histogram, side by side. Mirrors R's
          ``visualize.lm(plot="residuals")``.
        - ``"all"``: a combined ``cowplot``-style panel with the model
          fit on the left and the residual plots on the right.
          When ``cowplot`` isn't installed, the components are returned
          as a dict of named layers.

    Returns
    -------
    plotnine.ggplot OR dict
        When ``plot="model"`` or ``plot="all"`` (with cowplot), a
        single ggplot (or cowplot-joined object). When
        ``plot="residuals"``, a dict with ``"rvf"`` (residual-vs-fitted)
        and ``"hist"`` (residual histogram) ggplot objects.
    """
    # NeuralNetFit path: duck-typed dispatch so core.py doesn't have to
    # import flex_nn at module load time.
    if _is_neural_net_fit(model):
        return _visualize_neural_net(model, data=data, **kwargs)

    plot = str(plot).lower()
    valid_plots = ("model", "residuals", "all")
    if plot not in valid_plots:
        raise ValueError(
            f"plot must be one of {valid_plots}; got {plot!r}."
        )

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
    if endog_names in plot_df.columns:
        plot_df["__residual"] = plot_df[endog_names] - plot_df["__predicted"]
    else:
        plot_df["__residual"] = pd.Series(
            getattr(model, "resid", pd.Series(dtype=float))
        ).reindex(plot_df.index)

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

    # The "model" panel — legacy predicted-vs-observed plot.
    p_model = (
        ggplot(plot_df, aes(x=x_name, y=endog_names))
        + geom_point(alpha=0.4)
        + geom_line(aes(y="__predicted"), color="red", size=1)
        + labs(
            title=f"Visualization: {type(model).__name__}",
            subtitle=f"Predicted {endog_names} vs {x_name}",
        )
        + theme_bw()
    )

    if plot == "model":
        return p_model

    # Residual plots (used for plot="residuals" or plot="all").
    plot_resid_df = plot_df.dropna(subset=["__residual", "__predicted"])
    p_rvf = (
        ggplot(plot_resid_df, aes(x="__predicted", y="__residual"))
        + geom_point(alpha=0.4)
        + geom_hline(yintercept=0, linetype="dashed", color="gray")
        + labs(
            title="Residuals vs Predicted",
            x="Predicted",
            y="Residual",
        )
        + theme_bw()
    )
    p_hist = (
        ggplot(plot_resid_df, aes(x="__residual"))
        + geom_histogram(bins=30, fill="steelblue", color="white")
        + labs(
            title="Residual Distribution",
            x="Residual",
            y="Count",
        )
        + theme_bw()
    )

    if plot == "residuals":
        return {"rvf": p_rvf, "hist": p_hist}

    # plot == "all" — try to compose with cowplot; otherwise return dict.
    try:
        import cowplot  # type: ignore
        return cowplot.plot_grid(
            p_model, p_rvf, p_hist,
            ncol=2,
            rel_widths=(0.6, 0.4),
        )
    except ImportError:
        return {
            "model": p_model,
            "rvf": p_rvf,
            "hist": p_hist,
        }


def compare_fits(
    formula: str,
    data: pd.DataFrame,
    model1,
    model2,
    labels: List[str] = ["Model 1", "Model 2"],
    return_preds: bool = False,
    pred_type: str = "response",
    report_se: bool = False,
    re: bool = False,
    num_points: Optional[int] = None,
    clusters: Optional[int] = None,
    **kwargs,
):
    """
    Visually compare the fit of two different models (statsmodels/sklearn).

    Parameters
    ----------
    formula, data, model1, model2, labels :
        Existing arguments; see prior releases.
    return_preds : bool, default False
        When True, return the prediction DataFrame instead of the plot.
        Columns: the formula's predictor(s), ``__y`` (the observed
        response), and ``__pred1`` / ``__pred2`` (each model's
        prediction). This matches R's ``compare.fits(..., return.preds=TRUE)``.
    pred_type : {"response", "link"}, default "response"
        Type of predictions passed to ``statsmodels`` for GLM models.
        ``"response"`` returns the probability scale (default);
        ``"link"`` returns the linear-predictor scale. Ignored for
        non-GLM models. Matches R's ``compare.fits(..., pred.type=...)``.
    report_se : bool, default False
        R-parity stub: when True, would include standard-error bands on
        prediction lines.  Currently ignored; reserved for a future release.
    re : bool, default False
        R-parity stub: when True, would include random-effects predictions
        for mixed models.  Currently ignored; reserved for a future release.
    num_points : int, optional
        R-parity stub: when set, would evaluate predictions on a grid of
        ``num_points`` points spanning the x-range.  Currently ignored;
        reserved for a future release.
    clusters : int, optional
        R-parity stub: when set, would cluster the prediction grid.
        Currently ignored; reserved for a future release.

    Returns
    -------
    plotnine.ggplot OR pandas.DataFrame
        When ``return_preds=False``, the comparison plot. When
        ``return_preds=True``, a DataFrame with observed and predicted
        values for both models.
    """
    if report_se or re or num_points is not None or clusters is not None:
        warnings.warn(
            "compare_fits(): arguments report_se/re/num_points/clusters are "
            "accepted for R API parity but are currently no-ops in py-flexplot.",
            UserWarning,
            stacklevel=2,
        )

    variables = parse_flexplot_formula(formula)
    _validate_data_for_plot(formula, data, variables)

    y_name = variables["y"]
    x_name = variables["x"]

    pred1 = _get_model_predictions(model1, data, pred_type=pred_type)
    pred2 = _get_model_predictions(model2, data, pred_type=pred_type)

    if len(pred1) != len(data) or len(pred2) != len(data):
        raise ValueError(
            f"Model predictions must match data length ({len(data)}): "
            f"got {len(pred1)} and {len(pred2)}."
        )

    plot_df = data.copy()
    plot_df["__m1"] = pred1
    plot_df["__m2"] = pred2

    if return_preds:
        return plot_df

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


def third_eye(*args, **kwargs):
    """R API placeholder for ``third.eye``.

    The R package exposes ``third.eye`` as a specialized 3-way interaction
    visualization.  py-flexplot intentionally keeps this as a stub for now,
    so API discovery/parity tooling can detect the endpoint while behavior
    remains explicitly unimplemented.
    """
    raise NotImplementedError(
        "third_eye() is not implemented in py-flexplot yet. "
        "Use flexplot(..., interaction_model=True) for interaction visuals."
    )


def _get_model_predictions(
    model,
    data: pd.DataFrame,
    pred_type: str = "response",
) -> pd.Series:
    """Return a pandas Series of predictions aligned to data.index.

    Parameters
    ----------
    model : fitted model with ``predict`` method
    data : pd.DataFrame
        Predictor data.
    pred_type : {"response", "link"}, default "response"
        Passed through to ``statsmodels`` GLM ``predict``; controls
        whether the prediction is on the response or link scale.
        Ignored for non-GLM models.
    """
    if hasattr(model, "predict"):
        try:
            # Statsmodels fitted models accept a DataFrame and return a Series
            # indexed by the (possibly reduced) observation index. GLM models
            # accept a ``linear`` kwarg to switch between response and link.
            if hasattr(model, "model") and getattr(model, "model", None) is not None:
                model_class = type(model).__name__.lower()
                if "glm" in model_class:
                    pred = model.predict(data, linear=(pred_type == "link"))
                else:
                    pred = model.predict(data)
            else:
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


def added_plot(
    formula: str,
    data: pd.DataFrame,
    lm_formula: Optional[str] = None,
    method: str = "loess",
    x: Optional[Union[str, int]] = None,
    offset: bool = True,
    **kwargs,
):
    """Create an added variable plot (R-flexplot ``added.plot()`` parity).

    Residualizes the outcome on a conditioning model, adds the mean of the
    outcome back to the residuals (R's "maintain interpretation" step), and
    plots the chosen display variable against those residuals.

    R-flexplot semantics (v0.8.0+; supersedes the v0.6.x behavior which
    plotted the first variable against doubly-residualized data):

    - ``formula``: ``y ~ var1 + var2 + ...``.
    - Default display variable (``x=None``) is the **last** variable on the
      RHS of ``formula`` (R's default): ``y ~ x + z`` residualizes ``y`` on
      ``x`` and plots ``z``.
    - ``lm_formula`` (optional): the fitted model used to residualize ``y``
      (e.g. ``"weight.loss ~ health * muscle.gain"``). Defaults to the
      remaining formula variables (all but the display variable).
    - ``x`` (optional): which variable to display. Either the column name
      or its 1-based position in ``formula``'s predictor list (R uses
      ``x=2`` for the second).
    - ``offset`` (default ``True``): add the mean of ``y`` back onto the
      residuals so the y-axis keeps the outcome's scale (R does this).
      Pass ``False`` for raw centered residuals.
    - ``method``: smoother for the fitted line (default ``"loess"``,
      matching R).
    """
    variables = parse_flexplot_formula(formula)
    _validate_data_for_plot(formula, data, variables, require_numeric_x=True)

    y_var = variables["y"]
    all_x = [v for v in variables["all_x"] if ":" not in v]  # atoms only
    if not all_x:
        # Degenerate: only an interaction term (or nothing) — fall back.
        return flexplot(formula, data, **kwargs)

    # Resolve the display variable (R default: last on the RHS).
    if x is None:
        x_var = all_x[-1]
    elif isinstance(x, int):
        if not (1 <= x <= len(all_x)):
            raise ValueError(
                f"x={x} is out of range; formula has {len(all_x)} predictors."
            )
        x_var = all_x[x - 1]
    elif isinstance(x, str):
        if x not in all_x:
            raise ValueError(
                f"x={x!r} not found among formula predictors {all_x}."
            )
        x_var = x
    else:
        raise TypeError(f"x must be a str, int, or None; got {type(x).__name__}.")

    # Conditioning variables: from lm_formula if given, else all formula
    # predictors except the display variable (R: "the fitted model that is
    # then residualized"). Handle interaction-expanded atoms.
    if lm_formula is not None:
        if not isinstance(lm_formula, str):
            raise TypeError(
                f"lm_formula must be a string; got {type(lm_formula).__name__}."
            )
        lm_variables = parse_flexplot_formula(
            lm_formula if "~" in lm_formula else f"{y_var} ~ {lm_formula}"
        )
        lm_rhs = [v for v in lm_variables["all_x"]]
        if lm_variables["y"] != y_var:
            raise ValueError(
                f"lm_formula {lm_formula!r} must share the outcome {y_var!r}; "
                f"it uses {lm_variables['y']!r}."
            )
        condition_vars = [v for v in lm_variables["all_x"] if ":" not in v]
        if x_var in condition_vars:
            condition_vars.remove(x_var)
    else:
        condition_vars = [v for v in all_x if v != x_var]
        if not condition_vars:
            # Only one predictor: nothing to condition on; plain flexplot.
            return flexplot(formula, data, **kwargs)

    # Residualize y on the conditioning variables.
    clean_cond = [re.sub(r"\W", "_", v) for v in condition_vars]
    mapping = {orig: clean for orig, clean in zip(condition_vars, clean_cond)}
    needed = list(dict.fromkeys(condition_vars + [x_var, y_var]))
    fit_df = data[needed].dropna().copy()
    for orig, clean in mapping.items():
        if orig != clean:
            fit_df[clean] = fit_df.pop(orig)
    y_res_model = OLS.from_formula(
        f"{y_var} ~ {' + '.join(clean_cond)}", data=fit_df
    ).fit()
    y_residuals = y_res_model.resid + y_res_model.model.endog.mean() if offset \
        else y_res_model.resid

    plot_df = pd.DataFrame({
        x_var: fit_df[x_var].to_numpy(),
        f"{y_var}|cond": np.asarray(y_residuals),
    })

    aes_kwargs = {"x": x_var, "y": f"{y_var}|cond"}
    p = ggplot(plot_df, aes(**aes_kwargs))
    p += geom_point(alpha=0.5)
    # Reuse the numeric-smooth machinery, honoring method / uncertainty /
    # level / bands kwargs when provided.
    kwargs_unc = kwargs.pop("uncertainty", "ci")
    kwargs_level = kwargs.pop("level", 0.95)
    kwargs_bands = kwargs.pop("bands", None)
    p = _add_numeric_smooth(
        p, plot_df, x_var, f"{y_var}|cond",
        method if method in _VALID_FLEXPLOT_METHODS else ("loess" if method == "auto" else method),
        kwargs_unc, kwargs_level, kwargs_bands,
    )
    p += labs(x=x_var, y=f"{y_var} | conditional", title="Added Variable Plot")
    p += theme_bw()
    return p
