"""py-flexplot: Intelligent data visualization and statistical tools."""

from .core import flexplot, visualize, compare_fits, added_plot
from .stats import model_comparison, estimates, p_format, eliminated_columns, color_table
from .ebbr import fit_beta_prior, add_ebb_estimate

__all__ = [
    "flexplot",
    "visualize",
    "compare_fits",
    "added_plot",
    "model_comparison",
    "estimates",
    "p_format",
    "eliminated_columns",
    "color_table",
    "fit_beta_prior",
    "add_ebb_estimate"
]
