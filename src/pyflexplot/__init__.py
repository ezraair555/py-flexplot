"""py-flexplot: Intelligent data visualization and statistical tools."""

from . import datasets, flex_nn, ml
from .bluepill import estimate_sd, mixed_model
from .core import added_plot, compare_fits, flexplot, third_eye, visualize
from .descriptives import meansplot, scatter3D
from .ebbr import add_ebb_estimate, fit_beta_prior
from .quality import diagnose, format_summary
from .sem import disturbance_plot, hopper_plot, measurement_plot
from .stats import (
    bf_bic,
    color_table,
    eliminated_columns,
    estimates,
    eta_squared,
    model_comparison,
    p_format,
    rsq_change,
    standardized_beta,
)

__all__ = [
    # flexplot (core visualization)
    "flexplot",
    "visualize",
    "compare_fits",
    "added_plot",
    "third_eye",
    # auto data-quality diagnostics
    "diagnose",
    "format_summary",
    # fifer / fifer2 (biostatistics)
    "model_comparison",
    "estimates",
    "eta_squared",
    "standardized_beta",
    "rsq_change",
    "bf_bic",
    "p_format",
    "eliminated_columns",
    "color_table",
    # ebbr (empirical Bayes shrinkage)
    "fit_beta_prior",
    "add_ebb_estimate",
    # flexplavaan (SEM visualization)
    "hopper_plot",
    "disturbance_plot",
    "measurement_plot",
    # bluepill (simulated datasets)
    "estimate_sd",
    "mixed_model",
    # descriptives (descriptive-statistics visualizations)
    "meansplot",
    "scatter3D",
    # flex_nn (neural-network integration)
    "flex_nn",
    # ml (sklearn / random-forest adapters)
    "ml",
    # textbook example datasets
    "datasets",
]
