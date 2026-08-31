import re
import warnings

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
    geom_ribbon,
    geom_hline,
    geom_boxplot,
    geom_bar,
    stat_summary,
    facet_wrap,
    facet_grid,
    scale_color_identity,
    scale_color_manual,
    labs,
    theme_bw,
)
import statsmodels.api as sm
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


_VALID_SPREAD = frozenset({None, "stdev", "range", "iqr", "no", "ci"})


def _add_discrete_summary(p, spread: Optional[str]):
    """Add the dispersion marker layer for the discrete-x branch.

    Mirrors R-flexplot's ``spread`` argument:
    - None / "ci": bootstrap CI on the mean (plotnine's stat_summary with
      ``fun_data='mean_cl_boot'``). This is the legacy default.
    - "stdev": mean +/- 1 SD as a crossbar (pointrange with computed limits).
    - "range": min-max range as a wider crossbar.
    - "iqr": Q1-Q3 IQR as a boxplot-like crossbar.
    - "no": no summary layer at all.
    """
    if spread not in _VALID_SPREAD:
        raise ValueError(
            f"spread must be one of {sorted(s for s in _VALID_SPREAD if s)}; "
            f"got {spread!r}."
        )

    if spread == "no":
        return p

    if spread is None or spread == "ci":
        # Legacy default: bootstrap CI via stat_summary.
        p += stat_summary(fun_data="mean_cl_boot", color="red", size=1)
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

    p += stat_summary(fun_data=fun, fun_y=np.mean, geom="pointrange", color="red", size=0.5)
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


_VALID_FLEXPLOT_METHODS = frozenset({"auto", "lm", "loess", "polynomial", "cubic", "logistic"})

# Recognized methods for overlay entries. Includes a broader set than the
# primary ``method`` parameter because plotnine/statsmodels supports more
# smoothers for overlay use.
_VALID_OVERLAY_METHODS = frozenset({"lm", "loess", "lowess", "glm", "rlm", "ols", "wls", "gls", "mavg"})

# Default color cycle for overlay entries (distinct from the primary
# ``"blue"`` so the primary line is always visually identifiable).
_OVERLAY_COLOR_CYCLE = ("#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c")

# Interaction-operator detection. The presence of ``*`` or ``:`` in the
# right-hand side of a formula signals that the user wants interaction terms.
# The parser accepts these for forward-compatibility with v0.7.0 (real
# interaction-aware fitting), but the default fit in v0.6.x is still
# additive — a UserWarning is emitted to make this explicit.
_INTERACTION_OP = re.compile(r"[*:]")


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


def flexplot(
    formula: str,
    data: pd.DataFrame,
    method: str = "auto",
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
    method : {"auto", "lm", "loess", "polynomial", "cubic", "logistic"}
        Smoother for the numeric-vs-numeric branch. ``"auto"`` selects LM.
        ``"polynomial"`` / ``"cubic"``: degree-3 OLS in x (cubic is an alias).
        ``"logistic"``: GLM with logit link on numeric binary y (falls back
        to OLS with a warning if y is not in {0, 1}; the binary pre-check is
        bypassed so the parametric branch always fires).
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
    spread : {None, "ci", "stdev", "range", "iqr", "no"}, default None
        Dispersion marker drawn in the discrete-x branch alongside
        ``geom_jitter``. Mirrors R-flexplot's ``spread``.
        - ``None`` / ``"ci"``: bootstrap CI on the mean (legacy default).
        - ``"stdev"``: mean ± 1 SD as a pointrange.
        - ``"range"``: min-max range.
        - ``"iqr"``: Q1-Q3 IQR.
        - ``"no"``: no summary layer at all.
    sample : int, optional
        Subsample N rows for the plotnine layers (scatter / jitter) while
        keeping the smoother fits on the full DataFrame. No-op when
        ``N >= len(data)``. Deterministic via ``np.random.default_rng(0)``.
    ghost_line : {"red", "dashed", None}, default None
        Reference line drawn at y=0 after the main layers. ``"red"`` for a
        solid red threshold; ``"dashed"`` for a black dashed reference.
    plot_type : {"scatter", "line", "boxplot", "bar", None}, default None
        Explicit geom override. Bypasses the auto-dispatch.
    return_data : bool, default False
        When ``True``, return ``{"plot": ggplot, "data": DataFrame}``
        instead of just the plot. Useful with ``sample=`` to know which
        rows were plotted.
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

    Interaction syntax (``*``, ``:``) is parsed since v0.6.2 but the v0.6.x
    fit remains additive. A ``UserWarning`` is emitted whenever interaction
    syntax is detected; v0.7.0 will add ``interaction_model=True`` for true
    non-parallel slopes.

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
    if variables.get("has_interaction"):
        # v0.6.x: parser accepts interaction syntax (R-compatible) but the
        # fit remains additive (parallel slopes per color group). Warn so
        # users aren't misled. v0.7.0 will add `interaction_model=True` for
        # true non-parallel slopes.
        warnings.warn(
            f"Interaction syntax detected in formula {formula!r} but flexplot's "
            f"default fit is additive (parallel slopes per color group). "
            f"v0.7.0 will add `interaction_model=True` for non-parallel slopes. "
            f"To suppress this warning, write the formula without `*` or `:`.",
            UserWarning,
            stacklevel=2,
        )
    _validate_data_for_plot(formula, data, variables)

    y = variables["y"]
    x = variables["x"]
    color = variables["color"]
    given = variables["given"]
    intercept_only = variables.get("intercept_only", False)

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

    if intercept_only:
        # Intercept-only: show a univariate distribution of y.
        p = ggplot(plot_input_df, aes(x=y)) + geom_histogram(bins=30)
        p += labs(title=f"Distribution of {y}")
        p += theme_bw()
        if return_data:
            return {"plot": p, "data": plot_input_df}
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
    if is_y_numeric and method not in {"logistic"}:
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
        _VALID_PLOT_TYPES = {"scatter", "line", "boxplot", "bar"}
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
        elif plot_type == "bar":
            # plotnine's geom_bar doesn't accept fun=; use stat_summary
            # with fun_y=np.mean + geom="bar" to get a bar chart of
            # group means per x level.
            p += stat_summary(fun_y=np.mean, geom="bar")
        # Skip the auto-dispatch below.
        skip_dispatch = True
    else:
        skip_dispatch = False

    # Determine plot type.
    # Order matters:
    #   1. Binary 0/1 y must be detected before the generic numeric branch
    #      (otherwise int/float [0, 1] y falls into LM/loess).
    #   2. Numeric X is "discrete" when _is_discrete() returns True (numeric
    #      with <=10 unique values; post-05ac368 R-flexplot parity).
    if not skip_dispatch and y_is_binary and not is_x_discrete:
        # Binomial GLM branch — numeric binary outcome with numeric x.
        p += geom_point(alpha=0.3)
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
        p += geom_point(alpha=0.3)
        p = _add_binomial_smooth(p, data, x, y, uncertainty, level, bands)
        if overlay_specs:
            p = _add_overlay_binomial(p, data, x, y, overlay_specs)

    elif not skip_dispatch and is_y_numeric and not is_x_discrete:
        p += geom_point(alpha=0.5)
        p = _add_numeric_smooth(
            p, data, x, y, method, uncertainty, level, bands
        )
        if overlay_specs:
            p = _add_overlay_numeric(p, data, x, y, overlay_specs)

    elif not skip_dispatch and is_y_numeric and is_x_discrete:
        p += geom_jitter(width=0.2, alpha=0.5)
        p = _add_discrete_summary(p, spread)

    elif not skip_dispatch:
        p += geom_jitter(width=0.2, height=0.2, alpha=0.5)

    if len(given) == 1:
        p += facet_wrap(f"~{given[0]}")
    elif len(given) >= 2:
        p += facet_grid(f"{given[1]} ~ {given[0]}")

    p += theme_bw()

    # --- Optional ghost.line reference layer (v0.6.5+) ---
    # ghost_line="red": solid red reference line. Useful for highlighting a
    # threshold or a reference value (e.g. y=0, or y=mean(y)).
    # ghost_line="dashed": dashed black line. R's flexplot() uses this to
    # mark the slope=1 reference for prediction-vs-observed plots.
    # Both are drawn as geom_hline (horizontal), so they're 1D references
    # at y=0. For diagonal references (slope=1), future work.
    if ghost_line is not None:
        if ghost_line not in {"red", "dashed"}:
            raise ValueError(
                f"ghost_line must be 'red', 'dashed', or None; got {ghost_line!r}."
            )
        if ghost_line == "red":
            p += geom_hline(yintercept=0, color="red")
        elif ghost_line == "dashed":
            p += geom_hline(yintercept=0, color="black", linetype="dashed")

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
):
    """Add fitted line + uncertainty band for numeric-vs-numeric.

    Returns the plotnine plot object with the appropriate layers added.
    Caller is responsible for adding geom_point first.
    """
    if uncertainty is None:
        # No fit at all — preserve the scatter only.
        return p

    use_loess = method == "loess"
    # polynomial/cubic are OLS fits with higher-order x terms; logistic is
    # a GLM with the logit link. plotnine's geom_smooth(method="lm", ...) does
    # NOT support poly-of-x cleanly, so we route these through statsmodels
    # and add geom_line + geom_ribbon manually (mirroring the prediction /
    # bootstrap branches).
    if method in {"polynomial", "cubic", "logistic"}:
        return _add_parametric_smooth(
            p, data, x, y, method, uncertainty, level, bands
        )

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
):
    """Add fitted line + CI ribbon for polynomial / cubic / logistic methods.

    plotnine's ``geom_smooth(method="lm", ...)`` does NOT accept
    ``formula=poly(x, k)`` cleanly, so we fit statsmodels directly and draw
    the line + ribbon manually. Mirrors the prediction/ bootstrap branches
    in ``_add_numeric_smooth``.

    Methods:
    - "polynomial": OLS with degree-3 polynomial in x (default; same as
      ``cubic``). User can call with ``method="polynomial"``; degree is
      fixed at 3 for now — R-flexplot's default.
    - "cubic": alias of "polynomial".
    - "logistic": GLM with logit link on numeric binary y. Falls back to
      OLS if y is not in {0, 1} (and emits a UserWarning).
    """
    from scipy import stats as _scipy_stats

    x_arr = data[x].to_numpy(dtype=float)
    y_arr = data[y].to_numpy(dtype=float)
    n = x_arr.size
    if n < 2:
        return p

    if method in {"polynomial", "cubic"}:
        # Build the design matrix: intercept + x + x^2 + x^3.
        X = np.column_stack([np.ones_like(x_arr), x_arr, x_arr ** 2, x_arr ** 3])
        model = OLS(y_arr, X).fit()
        link_label = "polynomial (degree-3)"
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
            # Linear OLS fallback: degree-1 in x (intercept + x). The eval
            # matrix below uses the same shape.
            X = np.column_stack([np.ones_like(x_arr), x_arr])
            model = OLS(y_arr, X).fit()
            link_label = "OLS fallback (logistic requires binary y)"
        else:
            import statsmodels.api as _sm
            X = _sm.add_constant(x_arr)
            model = _sm.GLM(
                y_arr, X, family=_sm.families.Binomial(link=_sm.families.links.Logit())
            ).fit()
            link_label = "logistic (logit)"
    else:  # pragma: no cover — guarded by caller
        return p

    x_eval = np.linspace(np.nanmin(x_arr), np.nanmax(x_arr), num=200)

    if method in {"polynomial", "cubic"}:
        # OLS degree-3 design matrix: intercept + x + x^2 + x^3.
        X_eval = np.column_stack(
            [np.ones_like(x_eval), x_eval, x_eval ** 2, x_eval ** 3]
        )
    else:  # logistic (binary GLM) and logistic (OLS degree-1 fallback): both use intercept + x.
        import statsmodels.api as _sm
        X_eval = _sm.add_constant(x_eval)

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
            # Fallback: normal-approx CI on the linear predictor.
            se = getattr(model, "bse", None)
            residual_std = float(np.sqrt(model.mse_resid)) if hasattr(model, "mse_resid") else 1.0
            lower = yhat_eval - z * residual_std
            upper = yhat_eval + z * residual_std

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

    Returns
    -------
    plotnine.ggplot OR pandas.DataFrame
        When ``return_preds=False``, the comparison plot. When
        ``return_preds=True``, a DataFrame with observed and predicted
        values for both models.
    """
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
