"""Reusable Plotly chart builders for exploratory data analysis."""

from app.services.plotting.charts import (
    create_boxplot,
    create_categorical_overview_chart,
    create_class_distribution_chart,
    create_correlation_heatmap,
    create_feature_distribution_chart,
    create_histogram,
    create_missing_values_chart,
    create_numeric_summary_chart,
    create_scatter_plot,
    create_target_rate_by_category_chart,
)
from app.services.plotting.base import figure_to_dict

__all__ = [
    "create_boxplot",
    "create_categorical_overview_chart",
    "create_class_distribution_chart",
    "create_correlation_heatmap",
    "create_feature_distribution_chart",
    "create_histogram",
    "create_missing_values_chart",
    "create_numeric_summary_chart",
    "create_scatter_plot",
    "create_target_rate_by_category_chart",
    "figure_to_dict",
]
