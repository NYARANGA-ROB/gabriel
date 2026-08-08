"""Exploratory data analysis service."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.services.plotting import (
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
    figure_to_dict,
)


class EDAServiceError(Exception):
    """Raised when EDA cannot be generated."""

MAX_EDA_ROWS = 10_000


def _is_numeric_column(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def get_column_groups(dataframe: pd.DataFrame) -> dict[str, list[str]]:
    """Group dataset columns by type."""
    numeric = [col for col in dataframe.columns if _is_numeric_column(dataframe[col])]
    categorical = [col for col in dataframe.columns if col not in numeric]
    return {
        "all": [str(col) for col in dataframe.columns],
        "numeric": numeric,
        "categorical": categorical,
    }


def _default_target(columns: list[str]) -> str | None:
    for candidate in ("readmitted", "target", "label", "class", "outcome"):
        matches = [col for col in columns if col.lower() == candidate]
        if matches:
            return matches[0]
    return columns[-1] if columns else None


def _pick_column(columns: list[str], preferred: str | None, fallback_index: int = 0) -> str | None:
    if preferred and preferred in columns:
        return preferred
    return columns[fallback_index] if columns else None


def build_eda_charts(
    filepath: Path,
    target_column: str | None = None,
    feature_column: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
) -> dict:
    """Build all EDA chart payloads for a dataset."""
    try:
        dataframe = pd.read_csv(filepath)
    except Exception as exc:
        raise EDAServiceError("Unable to read dataset for exploratory analysis.") from exc

    if dataframe.empty:
        raise EDAServiceError("Dataset is empty.")

    if len(dataframe) > MAX_EDA_ROWS:
        dataframe = dataframe.sample(n=MAX_EDA_ROWS, random_state=42)

    columns = get_column_groups(dataframe)
    all_columns = columns["all"]
    numeric_columns = columns["numeric"]

    target = _pick_column(all_columns, target_column or _default_target(all_columns))
    feature = _pick_column(numeric_columns or all_columns, feature_column, 0)
    x_col = _pick_column(numeric_columns, x_column, 0)
    y_col = _pick_column(numeric_columns, y_column, 1 if len(numeric_columns) > 1 else 0)

    if x_col == y_col and len(numeric_columns) > 1:
        y_col = numeric_columns[1]

    charts = {
        "missing_values": figure_to_dict(create_missing_values_chart(dataframe)),
        "correlation_heatmap": figure_to_dict(create_correlation_heatmap(dataframe)),
    }

    if target:
        charts["class_distribution"] = figure_to_dict(
            create_class_distribution_chart(dataframe, target)
        )
    else:
        charts["class_distribution"] = None

    if feature:
        charts["feature_distribution"] = figure_to_dict(
            create_feature_distribution_chart(dataframe, feature)
        )
        charts["histogram"] = figure_to_dict(create_histogram(dataframe, feature))
        charts["boxplot"] = figure_to_dict(create_boxplot(dataframe, feature, target))
    else:
        charts["feature_distribution"] = None
        charts["histogram"] = None
        charts["boxplot"] = None

    if x_col and y_col:
        charts["scatter_plot"] = figure_to_dict(
            create_scatter_plot(dataframe, x_col, y_col, target)
        )
    else:
        charts["scatter_plot"] = None

    if target:
        rate_column = _pick_column(columns["categorical"], feature_column, 0)
        if rate_column and rate_column != target:
            charts["target_rate_by_category"] = figure_to_dict(
                create_target_rate_by_category_chart(dataframe, rate_column, target)
            )
        else:
            charts["target_rate_by_category"] = None
    else:
        charts["target_rate_by_category"] = None

    charts["categorical_overview"] = figure_to_dict(create_categorical_overview_chart(dataframe))
    charts["numeric_summary"] = figure_to_dict(create_numeric_summary_chart(dataframe))

    return {
        "charts": charts,
        "columns": columns,
        "selection": {
            "target_column": target,
            "feature_column": feature,
            "x_column": x_col,
            "y_column": y_col,
        },
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
    }
