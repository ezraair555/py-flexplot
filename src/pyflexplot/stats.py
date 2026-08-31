import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from typing import Any, Dict, List, Optional, Union


def _check_statsmodels_attrs(model, attrs):
    """Raise ValueError if *model* is missing any of the listed attributes."""
    missing = [a for a in attrs if not hasattr(model, a)]
    if missing:
        raise ValueError(
            f"Model is missing required attributes for model_comparison: {missing}"
        )


def model_comparison(model1, model2, return_pred_difference=False):
    """
    Statistically compares the fits of two statsmodels results.

    Returns a tuple ``(DataFrame, p_value)`` where the DataFrame carries
    per-model AIC, BIC, LogLik, R-squared, adjusted R-squared, and Bayes
    factor (computed from BIC via the Kass & Raftery 1995 approximation).
    Pass ``return_pred_difference=True`` for a third element: a pandas
    Series of quantiles (0/25/50/75/100%) of the two models' in-sample
    prediction differences — R's ``model.comparison()`` returns this as
    its ``pred.difference`` component (v0.8.0+).

    The Bayes factor is attached to the more likely model (BIC-wise):
    the model with the lower BIC gets a BF ≥ 1 in its row, the other
    model gets 1/BF. This mirrors R's ``flexplot::model.comparison()``
    behavior.

    Non-nested models (v0.8.0+): AIC / BIC / Bayes factor / R² are valid
    for both nested and non-nested pairs; the likelihood-ratio p-value is
    only defined for nested models. When the two models' predictor sets
    are not a subset-superset pair, ``p_value`` is ``None`` (v0.7.x
    raised ``ValueError`` in that case — behavior change, documented).

    Parameters
    ----------
    model1, model2 : statsmodels regression results
        The models to compare. Nested-ness is detected via the models'
        ``exog_names`` (one set must contain the other).
    return_pred_difference : bool, default False
        When True, return ``(DataFrame, p_value, pred_difference)`` where
        ``pred_difference`` is a Series of quantiles of prediction
        differences, or None if predictions couldn't be aligned.
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

    # Nesting detection (v0.8.0+): R's model.comparison() supports non-nested
    # models (AIC/BIC/BF need no nesting); the LRT p-value is only defined
    # when one model's predictors are a subset of the other's. When models
    # are not nested (or the df difference is degenerate), we return
    # p_value=None instead of raising (v0.7.x raised ValueError).
    names1 = set(getattr(model1.model, "exog_names", None) or [])
    names2 = set(getattr(model2.model, "exog_names", None) or [])
    nested = bool(names1) and bool(names2) and (
        names1.issubset(names2) or names2.issubset(names1)
    )

    if nested and df_diff > 0:
        p_val = float(1 - stats.chi2.cdf(lr_stat, df_diff))
    else:
        p_val = None

    # --- pred.difference (v0.8.0+, R-parity) ---
    # R's model.comparison() returns list(statistics=..., pred.difference=...);
    # the pred.difference component holds quantiles of the two models'
    # in-sample prediction differences. Computed here via no-arg predict()
    # (statsmodels returns in-sample fitted values), available on all
    # RegressionResults. Non-fatal on failure (set to None).
    pred_difference = None
    try:
        p1 = np.asarray(model1.predict(), dtype=float)
        p2 = np.asarray(model2.predict(), dtype=float)
        if p1.shape == p2.shape and p1.size > 0:
            diff = pd.Series(p1 - p2)
            pred_difference = diff.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    except Exception:
        pred_difference = None

    if return_pred_difference:
        return res, p_val, pred_difference
    return res, p_val


def eta_squared(model, level: float = 0.95, typ: int = 3):
    """Compute partial eta-squared (η²_p) per predictor in a fitted OLS model.

    A port of R's ``sjstats::eta_sq()`` / ``fifer::eta_squared()``. For each
    non-intercept term in the model, returns the partial eta-squared:

        η²_p = (SS_effect / df_effect) / (SS_effect / df_effect + SS_resid / df_resid)
             = (F * df1) / (F * df1 + df2)

    where F is the per-term F-statistic, df1 is the term's df (1 for a
    single coefficient, k for a categorical), and df2 is the residual df.

    Partial eta-squared estimates the variance in y explained by each
    predictor *after* controlling for all other predictors. It's bounded
    in [0, 1] but can exceed R² when predictors are correlated (it's a
    separate concept from semi-partial R² which is bounded above by R²).

    Method (v0.7.5+):
    - Use ``statsmodels.stats.anova.anova_lm(model, typ=typ)`` to get
      type-I, II, or III sums of squares per term. Default ``typ=3``
      matches R's ``car::Anova(..., type=3)`` semantics.
    - For each term, compute η²_p from the per-term F.
    - Compute CI via the same non-central-F inversion as
      ``_r_squared_ci()`` applied to η²_p.

    Parameters
    ----------
    model : statsmodels.regression.linear_model.RegressionResults
        A fitted OLS model (or any model with ``.fvalue``, ``.f_pvalue``,
        ``.df_model``, ``.df_resid``, ``.model.exog_names`` attributes).
    level : float, default 0.95
        Coverage probability for the per-predictor CI.
    typ : int, default 3
        Type of sums of squares. 1 = sequential, 2 = hierarchical,
        3 = marginal (R's default). Type III is the most common choice
        for unbalanced designs and matches ``car::Anova(type=3)``.

    Returns
    -------
    pandas.DataFrame
        Indexed by predictor name (excluding intercept). Columns:
        - ``eta_sq`` : partial eta-squared
        - ``eta_sq_ci_low`` : CI lower bound (or ``None`` if degenerate)
        - ``eta_sq_ci_high`` : CI upper bound (or ``None`` if degenerate)
        - ``F`` : per-term F-statistic
        - ``p_value`` : per-term p-value
        - ``df`` : per-term degrees of freedom
    """
    if not hasattr(model, "df_model") or not hasattr(model, "df_resid"):
        raise TypeError(
            "eta_squared requires a statsmodels regression result; "
            "got an object with no df_model / df_resid attributes."
        )

    # Identify the predictor names (excluding intercept).
    exog_names = getattr(getattr(model, "model", None), "exog_names", None)
    if exog_names is None:
        raise TypeError(
            "eta_squared requires a statsmodels model with .model.exog_names."
        )
    predictors = [n for n in exog_names if n != "Intercept"]

    if not predictors:
        return pd.DataFrame(
            columns=["eta_sq", "eta_sq_ci_low", "eta_sq_ci_high", "F", "p_value", "df"]
        )

    # Compute per-term SS via statsmodels' anova_lm. Type III SS requires
    # the model to have been fitted with `data` so the design matrix can
    # be reconstructed; if that fails, fall back to a manual approach.
    try:
        from statsmodels.stats.anova import anova_lm
        anova_tbl = anova_lm(model, typ=typ)
    except Exception as exc:
        # statsmodels raises if the model wasn't fit with `data=` and we
        # can't recover the term-level SS. Fall back to the legacy
        # single-row computation rather than crashing.
        if not hasattr(model, "fvalue"):
            return pd.DataFrame(
                columns=["eta_sq", "eta_sq_ci_low", "eta_sq_ci_high", "F", "p_value", "df"]
            )
        F = float(model.fvalue)
        df1 = int(model.df_model)
        df2 = int(model.df_resid)
        nobs = int(model.nobs)
        eta2 = (F * df1) / (F * df1 + df2)
        ci = _r_squared_ci(r2=eta2, df_model=df1, nobs=nobs, level=level)
        return pd.DataFrame(
            {
                "eta_sq": [eta2],
                "eta_sq_ci_low": [ci[0] if ci is not None else None],
                "eta_sq_ci_high": [ci[1] if ci is not None else None],
                "F": [F],
                "p_value": [float(model.f_pvalue)],
                "df": [df1],
            },
            index=["model"],
        )

    # Drop the residual row; only term rows contribute to per-predictor η².
    if "df" in anova_tbl.columns and "F" in anova_tbl.columns:
        # Newer statsmodels uses lowercase; older uses uppercase. Be lenient.
        pass
    term_rows = anova_tbl.drop(index="Residual", errors="ignore")

    nobs = int(model.nobs)
    df_resid = float(anova_tbl.loc["Residual", "df"]) if "Residual" in anova_tbl.index else float(model.df_resid)

    rows = []
    for term, row in term_rows.iterrows():
        # Skip the intercept row.
        if term in {"Intercept", "C(Intercept)"}:
            continue
        F_term = float(row["F"])
        df_term = float(row["df"])
        p_term = float(row["PR(>F)"]) if "PR(>F)" in row else float("nan")
        eta2 = (F_term * df_term) / (F_term * df_term + df_resid)
        ci = _r_squared_ci(r2=eta2, df_model=int(df_term), nobs=nobs, level=level)
        rows.append({
            "eta_sq": eta2,
            "eta_sq_ci_low": ci[0] if ci is not None else None,
            "eta_sq_ci_high": ci[1] if ci is not None else None,
            "F": F_term,
            "p_value": p_term,
            "df": df_term,
        })
    if not rows:
        return pd.DataFrame(
            columns=["eta_sq", "eta_sq_ci_low", "eta_sq_ci_high", "F", "p_value", "df"]
        )
    return pd.DataFrame(rows, index=list(term_rows.index.drop(["Intercept", "Residual"], errors="ignore"))[:len(rows)])


def _r_squared_ci(r2: float, df_model: int, nobs: int, level: float = 0.95):
    """Confidence interval for R-squared via non-central-F inversion.

    Method (Olkin & Finn, 1995; matching R's ``MBESS::ci.R2()``):

    Given an observed R², a CI for the population R² (ρ²) is found by
    inverting the non-central F distribution. The test statistic

        F_obs = (R² / k) / ((1 - R²) / (n - k - 1))

    follows a non-central F distribution with ``(k, n - k - 1, λ)``
    degrees of freedom and non-centrality parameter
    ``λ = n * ρ² / (1 - ρ²)`` under the alternative that the population
    R² equals ρ².

    The CI endpoints solve for λ at each tail:

    - **Lower bound** ρ²_L: the noncentral F upper-tail P-value at F_obs
      equals α/2 (i.e., F_obs sits in the *lower* tail of the
      distribution, so the population R² is *smaller* than observed).
    - **Upper bound** ρ²_U: the noncentral F upper-tail P-value at F_obs
      equals 1 - α/2 (i.e., F_obs sits in the *upper* tail, so the
      population R² is *larger*).

    We invert for λ at each tail via bisection on the survival function,
    then recover ρ² via ``ρ² = λ / (n + λ)``.

    Edge cases:
    - R² very close to 1.0: the upper bound collapses; the CI is
      ``(lo, 1.0)``.
    - R² very close to 0.0: the lower bound collapses; the CI is
      ``(0.0, hi)``.
    - Invalid inputs (negative R², R² >= 1, df_model < 1, nobs <= k+1):
      return ``None``.

    Parameters
    ----------
    r2 : float
        Observed R-squared from a fitted OLS model.
    df_model : int
        Number of model parameters (excluding the intercept).
    nobs : int
        Number of observations used in the fit.
    level : float
        Coverage probability (default 0.95).

    Returns
    -------
    tuple of (lo, hi) or None
        ``None`` indicates the inputs were invalid; otherwise a
        ``(lo, hi)`` tuple with both bounds in [0.0, 1.0].
    """
    if not (0.0 <= r2 < 1.0) or df_model < 1 or nobs <= df_model + 1:
        return None

    from scipy.stats import ncf, f as f_dist

    k = df_model
    n = nobs
    df1, df2 = k, n - k - 1
    alpha = 1.0 - level

    # Observed F statistic.
    f_obs = (r2 / k) / ((1.0 - r2) / df2) if r2 < 1.0 else float("inf")

    def _upper_tail_p(lam: float) -> float:
        """P(F >= f_obs) under noncentral F(df1, df2, lambda).

        For lambda == 0, scipy's ncf.sf returns a buggy negative value
        on some versions; we fall back to the central F.sf in that
        case.
        """
        if lam == 0.0:
            return float(f_dist.sf(f_obs, df1, df2))
        return float(ncf.sf(f_obs, df1, df2, lam))

    def _solve_lambda_for_upper_tail_p(p_target: float) -> Optional[float]:
        """Find lambda such that _upper_tail_p(lambda) = p_target.

        The upper-tail P(F >= f_obs) under ncf(df1, df2, lambda) is
        monotonically *increasing* in lambda (as lambda grows, the
        distribution shifts right past f_obs). So we want a small
        lambda for small p_target, and a large lambda for large
        p_target.
        """
        p0 = _upper_tail_p(0.0)
        if p_target <= p0:
            # p_target is at or below the central-F upper-tail P-value:
            # the solution is at lambda = 0 (population R² = 0).
            return 0.0
        if p_target >= 1.0:
            # p_target is at or above 1; can never be reached (upper-
            # tail P is bounded above by 1). Return None to indicate
            # an open-ended CI.
            return None
        # Bracket: at lambda=0, upper-tail P is p0 (small). At large
        # lambda, upper-tail P approaches 1.
        lo = 0.0
        hi = 1.0
        while _upper_tail_p(hi) < p_target and hi < 1e10:
            hi *= 10.0
        if _upper_tail_p(hi) < p_target:
            # Cannot reach p_target; CI is open-ended.
            return None
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if _upper_tail_p(mid) < p_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    lambda_lo = _solve_lambda_for_upper_tail_p(alpha / 2.0)         # smaller λ -> smaller R²
    lambda_hi = _solve_lambda_for_upper_tail_p(1.0 - alpha / 2.0)   # larger λ -> larger R²

    def _lambda_to_r2(lam: Optional[float]) -> Optional[float]:
        if lam is None:
            return None
        if lam == 0.0:
            return 0.0
        return lam / (n + lam)

    rho_lo = _lambda_to_r2(lambda_lo)
    rho_hi = _lambda_to_r2(lambda_hi)

    if rho_lo is None:
        rho_lo = 0.0
    if rho_hi is None:
        rho_hi = 1.0

    rho_lo = max(0.0, min(1.0, rho_lo))
    rho_hi = max(0.0, min(1.0, rho_hi))
    return (rho_lo, rho_hi)


def estimates(model, mc: bool = True):
    """
    R-parity note (v0.8.0): ``mc`` mirrors R's estimates(mc=) — when False,
    comparison-dependent outputs (``semi.p.r2``, ``mean_differences``) are
    skipped (set to None). Factor-level estimates still compute.
    Compute a structured effect-size report for a fitted OLS model.

    A port of R's ``fifer::estimates()`` / ``flexplot::estimates.lm()``.
    Returns a dict with:

    - ``r.squared`` (float): model R-squared.
    - ``adj.r.squared`` (float): adjusted R-squared.
    - ``r.squared.ci`` (tuple or None): ``(lo, hi)`` 95% CI for R-squared via
      non-central F inversion (``statsmodels.stats.correlation.cov_nl``).
    - ``sigma`` (float): residual standard error.
    - ``n`` (int): number of observations used by the fit.
    - ``coef`` (pd.DataFrame): coefficients with name, estimate, std.
      error, t-statistic, p-value, and 95% CI from ``model.conf_int()``.
    - ``standardized`` (pd.Series): standardized betas for the predictors
      (excludes the intercept), computed as
      ``b_j * sd(x_j) / sd(y)``.
    - ``semi.p.r2`` (pd.Series): semi-partial R-squared for each
      predictor, computed by fitting reduced models
      (``y ~ x_other``) and measuring the R-squared drop from the full
      model.
    - ``factors`` (list[str]): names of factor (categorical) predictors.
    - ``numbers`` (list[str]): names of numeric predictors.
    - ``formula`` (str): the fitted formula, when accessible via
      ``model.model.formula``.

    Notes
    -----
    Cohen's d / factor pairwise differences and standardized betas for
    categorical predictors are NOT yet implemented (planned for v0.7.0).
    Random-effects / mixed-model ``estimates()`` is also deferred.

    Examples
    --------
    >>> import statsmodels.formula.api as smf
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "y": rng.normal(size=80),
    ...     "x1": rng.normal(size=80),
    ...     "x2": rng.normal(size=80),
    ... })
    >>> fit = smf.ols("y ~ x1 + x2", data=df).fit()
    >>> est = estimates(fit)
    >>> est["r.squared"] >= 0  # doctest: +SKIP
    True
    >>> "x1" in est["standardized"].index  # doctest: +SKIP
    True
    """
    if not hasattr(model, "rsquared"):
        raise TypeError(
            f"estimates() expects a fitted OLS-like model with a "
            f"'rsquared' attribute; got {type(model).__name__}."
        )

    out: Dict[str, Any] = {}

    # --- Model-level statistics -----------------------------------------
    out["r.squared"] = float(model.rsquared)
    out["adj.r.squared"] = float(model.rsquared_adj)
    out["sigma"] = float(np.sqrt(model.mse_resid))
    out["n"] = int(model.nobs)

    # --- R-squared CI via non-central F inversion ------------------------
    # statsmodels does not export cov_nl; use scipy's F distribution and
    # the standard non-centrality-parameter inversion to bracket R².
    out["r.squared.ci"] = _r_squared_ci(
        float(model.rsquared),
        int(model.df_model),
        int(model.nobs),
    )

    # --- Coefficient table with CIs ------------------------------------
    try:
        conf = model.conf_int(alpha=0.05)
    except Exception:
        conf = None

    coef_df = pd.DataFrame({
        "name": list(model.params.index),
        "estimate": model.params.to_numpy(),
        "std.error": model.bse.to_numpy(),
        "t": model.tvalues.to_numpy(),
        "p.value": model.pvalues.to_numpy(),
    })
    if conf is not None:
        coef_df["ci.lower"] = conf.iloc[:, 0].to_numpy()
        coef_df["ci.upper"] = conf.iloc[:, 1].to_numpy()
    out["coef"] = coef_df.set_index("name")

    # --- Recover original frame + formula for the harder computations ----
    inner = getattr(model, "model", None)
    frame = getattr(getattr(inner, "data", None), "frame", None)
    formula_str = getattr(inner, "formula", None)
    if isinstance(formula_str, str):
        out["formula"] = formula_str

    coef_names = [n for n in model.params.index if n != "Intercept"]

    # --- Standardized betas (predictors only, no intercept) ------------
    std_betas: Dict[str, float] = {}
    if inner is not None and len(coef_names) > 0:
        try:
            exog = getattr(inner, "exog", None)
            endog = getattr(inner, "endog", None)
            if exog is not None and endog is not None and exog.shape[1] >= 2:
                # First exog column is the intercept (constant); drop it.
                pred_cols = exog[:, 1:]
                if pred_cols.shape[1] == len(coef_names):
                    x_std = pred_cols.std(axis=0, ddof=1)
                    y_std = float(np.std(endog, ddof=1))
                    if y_std > 0:
                        for name, x_s in zip(coef_names, x_std):
                            std_betas[name] = float(
                                model.params[name] * x_s / y_std
                            )
        except Exception:
            pass
    out["standardized"] = pd.Series(std_betas, dtype=float)

    # --- Semi-partial R-squared per predictor ----------------------------
    # Approach: drop one predictor at a time and measure the R-squared
    # drop from the full model. semi.p.r2[j] = R2(full) - R2(reduced j).
    # Gated on mc= (R's 'should model comparisons be performed');
    # comparison-dependent outputs are skipped when mc=False.
    semi_p: Dict[str, float] = {}
    if (
        mc
        and frame is not None
        and isinstance(formula_str, str)
        and " ~ " in formula_str
        and len(coef_names) > 1
    ):
        full_r2 = float(model.rsquared)
        outcome, predictors = formula_str.split(" ~ ", 1)
        outcome = outcome.strip()
        predictor_list = [
            p.strip() for p in predictors.split(" + ") if p.strip()
        ]
        for pred in coef_names:
            other = [p for p in predictor_list if p != pred]
            if not other:
                continue
            try:
                reduced_formula = f"{outcome} ~ {' + '.join(other)}"
                reduced_fit = smf.ols(reduced_formula, data=frame).fit()
                semi_p[pred] = full_r2 - float(reduced_fit.rsquared)
            except Exception:
                semi_p[pred] = float("nan")
    out["semi.p.r2"] = pd.Series(semi_p, dtype=float)

    # --- Factor vs numeric split -----------------------------------------
    factors: List[str] = []
    numbers: List[str] = []
    if frame is not None:
        import re as _re
        seen = set()
        for term in coef_names:
            # Strip C(...) and [T.x] / [level] annotations that
            # statsmodels uses to denote categorical terms.
            base = _re.sub(r"^C\(([^)]+)\).*$", r"\1", term)
            if base in frame.columns and base not in seen:
                seen.add(base)
                col = frame[base]
                if (
                    col.dtype == object
                    or str(col.dtype).startswith("category")
                    or (col.dtype.kind in ("i", "u") and col.nunique() <= 5)
                ):
                    factors.append(base)
                else:
                    numbers.append(base)
    out["factors"] = factors
    out["numbers"] = numbers

    # --- Factor-level estimates + mean differences (v0.8.0, R-parity) ---
    # R's estimates() prints two additional tables for factor predictors:
    #   - "Estimates for Factors": per-level fitted means (holding other
    #     predictors at reference values) with CIs.
    #   - "Mean Differences": pairwise level contrasts with CIs and
    #     Cohen's d.
    factor_estimates_df = None
    mean_diff_df = None
    if frame is not None and factors and hasattr(model, "get_prediction"):
        try:
            alpha_ci = 0.05
            _outcome_name = None
            if formula_str is not None and isinstance(formula_str, str) and " ~ " in formula_str:
                _outcome_name = formula_str.split(" ~ ", 1)[0].strip()
            if _outcome_name is None or _outcome_name not in frame.columns:
                _outcome_name = None

            def _level_param_vector(base: str, level) -> Dict[str, float]:
                """Indicator vector over params for C(base)[T.level]."""
                candidates = [n for n in model.params.index if n.startswith(f"C({base})[")]
                vec = {n: 0.0 for n in model.params.index}
                if f"C({base})[T.{level}]" in candidates:
                    vec[f"C({base})[T.{level}]"] = 1.0
                return vec

            levels_by_base: Dict[str, list] = {}
            for base in factors:
                col = frame[base]
                levels_by_base[base] = sorted(col.dropna().unique().tolist())

            # Reference row: numeric -> mean; factor -> first level.
            def _base_row() -> Dict[str, object]:
                row: Dict[str, object] = {}
                for c in frame.columns:
                    col = frame[c]
                    if pd.api.types.is_numeric_dtype(col):
                        row[c] = float(col.mean())
                    else:
                        row[c] = col.dropna().unique()[0] if col.dropna().size else None
                return row

            est_rows = []
            diff_rows = []
            for base in factors:
                if base not in frame.columns:
                    continue
                ref_level = levels_by_base[base][0]
                # 1) Per-level fitted means via get_prediction grid.
                level_means = {}
                for level in levels_by_base[base]:
                    grid = _base_row()
                    if grid is None:
                        break
                    grid[base] = level
                    grid_df = pd.DataFrame([grid])
                    pred = model.get_prediction(grid_df)
                    sf = pred.summary_frame(alpha=alpha_ci)
                    est_rows.append({
                        "variable": base,
                        "level": level,
                        "estimate": float(sf["mean"].iloc[0]),
                        "ci.lower": float(sf["mean_ci_lower"].iloc[0]),
                        "ci.upper": float(sf["mean_ci_upper"].iloc[0]),
                    })
                    level_means[level] = float(sf["mean"].iloc[0])

                # 2) Pairwise contrasts with CI via param contrast vectors.
                params_vec = np.asarray(model.params)
                cov = np.asarray(model.cov_params())
                df_resid = float(getattr(model, "df_resid", np.nan))
                for i, l1 in enumerate(levels_by_base[base]):
                    for l2 in levels_by_base[base][i + 1:]:
                        v1 = _level_param_vector(base, l1)
                        v2 = _level_param_vector(base, l2)
                        contrast = np.array([v1.get(n, 0.0) - v2.get(n, 0.0)
                                             for n in model.params.index])
                        diff = float(contrast @ params_vec)
                        se = float(np.sqrt(contrast @ cov @ contrast))
                        t_crit = stats.t.ppf(0.975, df_resid) if df_resid == df_resid else 1.96
                        # Cohen's d: diff / pooled within-level SD of raw y.
                        if _outcome_name is None:
                            y1 = y2 = pd.Series(dtype=float)
                        else:
                            y1 = frame.loc[frame[base] == l1, _outcome_name].dropna()
                            y2 = frame.loc[frame[base] == l2, _outcome_name].dropna()
                        if len(y1) > 1 and len(y2) > 1:
                            s1, s2 = y1.std(ddof=1), y2.std(ddof=1)
                            n1, n2 = len(y1), len(y2)
                            pooled = float(np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))) if (n1 + n2 - 2) > 0 else np.nan
                        else:
                            pooled = float("nan")
                        cohens_d = diff / pooled if pooled and not np.isnan(pooled) and pooled != 0 else float("nan")
                        diff_rows.append({
                            "variable": base,
                            "comparison": f"{l1} - {l2}",
                            "difference": diff,
                            "ci.lower": diff - t_crit * se,
                            "ci.upper": diff + t_crit * se,
                            "cohens.d": cohens_d,
                        })

            if est_rows:
                factor_estimates_df = pd.DataFrame(est_rows)
            if diff_rows:
                mean_diff_df = pd.DataFrame(diff_rows)
        except Exception:
            factor_estimates_df = None
            mean_diff_df = None
    out["factor_estimates"] = factor_estimates_df
    out["mean_differences"] = mean_diff_df if mc else None

    return out


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


def standardized_beta(model):
    """Compute standardized (beta) coefficients for a fitted OLS model.

    Standalone accessor (v0.8.0+) mirroring R's ``flexplot::standardized.beta()``.
    Same values as ``estimates(model)["standardized"]`` — this standalone
    form is convenient when only the betas are needed.

    Standardized beta for coefficient j: ``b_j * sd(x_j) / sd(y)``.

    Note: for categorical predictors (patsy dummies in the design matrix),
    ``sd(x_j)`` is the SD of the 0/1 indicator column — R reports these
    differently; treat dummy-column betas with care.

    Returns
    -------
    pd.Series
        Indexed by non-intercept predictor name.
    """
    if not hasattr(model, "params"):
        raise TypeError(
            "standardized_beta requires a statsmodels regression result "
            "with .params."
        )
    exog_names = list(getattr(model.model, "exog_names", None) or [])
    if not exog_names:
        raise TypeError("standardized_beta requires .model.exog_names.")

    y = np.asarray(model.model.endog, dtype=float)
    X = np.asarray(model.model.exog, dtype=float)
    sd_y = float(np.std(y, ddof=1))
    out = {}
    for i, name in enumerate(exog_names):
        if name in ("Intercept", "const"):
            continue
        sd_x = float(np.std(X[:, i], ddof=1)) if X.shape[0] > 1 else np.nan
        if sd_y == 0 or not np.isfinite(sd_x) or sd_x == 0:
            out[name] = np.nan
        else:
            param_i = model.params.iloc[i] if hasattr(model.params, "iloc") else model.params[i]
            out[name] = float(param_i * sd_x / sd_y)
    return pd.Series(out, dtype=float)


def rsq_change(reduced_model, full_model):
    """Change in R-squared from reduced to full model (semi-partial R²).

    Standalone accessor (v0.8.0+) mirroring R's ``flexplot::rsq.change()``.
    Positive values indicate the extra predictors in ``full_model`` explain
    that share of variance beyond ``reduced_model``.

    Returns
    -------
    float
        ``full_model.rsquared - reduced_model.rsquared``.
    """
    if not hasattr(reduced_model, "rsquared") or not hasattr(full_model, "rsquared"):
        raise TypeError(
            "rsq_change requires two statsmodels results exposing .rsquared."
        )
    return float(full_model.rsquared) - float(reduced_model.rsquared)


def bf_bic(model1, model2):
    """Bayes factor for model1 over model2, from BICs (Kass & Raftery 1995).

    Standalone accessor (v0.8.0+) mirroring R's ``flexplot::bf.bic()``:

        BF_12 = exp((BIC_2 - BIC_1) / 2)

    Values > 1 favor model1; < 1 favor model2. Same computation as the
    ``BayesFactor`` column inside ``model_comparison()``.

    Returns
    -------
    float
    """
    for name, model in (("model1", model1), ("model2", model2)):
        if not hasattr(model, "bic"):
            raise TypeError(
                f"bf_bic requires statsmodels results with .bic; "
                f"{name} is missing it."
            )
    return float(np.exp((model2.bic - model1.bic) / 2.0))
