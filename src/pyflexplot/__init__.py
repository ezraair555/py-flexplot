"""py-flexplot: Intelligent data visualization and statistical tools."""

from .core import flexplot, visualize, compare_fits, added_plot, third_eye
from .quality import diagnose, format_summary
from .stats import (
    model_comparison,
    estimates,
    eta_squared,
    standardized_beta,
    rsq_change,
    bf_bic,
    p_format,
    eliminated_columns,
    color_table,
)
from .ebbr import fit_beta_prior, add_ebb_estimate
from .sem import hopper_plot, disturbance_plot, measurement_plot
from .bluepill import estimate_sd, mixed_model
from .descriptives import meansplot, scatter3D
from . import flex_nn
from . import ml

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
]
