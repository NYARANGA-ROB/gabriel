"""Reusable Plotly chart functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.services.plotting.base import CHART_COLORS, apply_layout, empty_figure


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def create_missing_values_chart(dataframe: pd.DataFrame) -> go.Figure:
    """Bar chart of missing values per column."""
    missing = dataframe.isnull().sum().sort_values(ascending=True)
    missing = missing[missing > 0]

    if missing.empty:
        return empty_figure("No missing values detected in this dataset.", "Missing Values")

    fig = go.Figure(
        go.Bar(
            x=missing.values,
            y=missing.index.astype(str),
            orientation="h",
            marker={"color": "#2563eb"},
            hovertemplate="Column: %{y}<br>Missing: %{x}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Missing count", yaxis_title="Feature")
    return apply_layout(fig, "Missing Values by Feature")


def create_class_distribution_chart(dataframe: pd.DataFrame, target_column: str) -> go.Figure:
    """Bar chart of target class frequencies."""
    if target_column not in dataframe.columns:
        return empty_figure("Selected target column was not found.", "Class Distribution")

    counts = dataframe[target_column].value_counts().sort_index()
    labels = counts.index.astype(str)

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts.values,
            marker={"color": CHART_COLORS[: len(labels)]},
            hovertemplate="Class: %{x}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title=target_column, yaxis_title="Count")
    return apply_layout(fig, f"Class Distribution — {target_column}")


def create_correlation_heatmap(dataframe: pd.DataFrame) -> go.Figure:
    """Correlation heatmap for numeric features."""
    numeric = dataframe.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return empty_figure("At least two numeric features are required.", "Correlation Heatmap")

    correlation = numeric.corr()
    fig = go.Figure(
        go.Heatmap(
            z=correlation.values,
            x=correlation.columns.astype(str),
            y=correlation.columns.astype(str),
            colorscale="RdBu",
            zmid=0,
            hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Feature", yaxis_title="Feature")
    return apply_layout(fig, "Correlation Heatmap", height=520)


def create_feature_distribution_chart(dataframe: pd.DataFrame, column: str) -> go.Figure:
    """Distribution chart adapted to feature type."""
    if column not in dataframe.columns:
        return empty_figure("Selected feature was not found.", "Feature Distribution")

    series = dataframe[column]
    if _is_numeric(series):
        fig = px.histogram(
            dataframe,
            x=column,
            nbins=min(30, max(10, series.nunique())),
            color_discrete_sequence=[CHART_COLORS[0]],
        )
        fig.update_layout(xaxis_title=column, yaxis_title="Frequency")
        return apply_layout(fig, f"Feature Distribution — {column}")

    counts = series.astype(str).value_counts().head(20)
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker={"color": CHART_COLORS[1]},
            hovertemplate="Category: %{x}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title=column, yaxis_title="Count")
    return apply_layout(fig, f"Feature Distribution — {column}")


def create_histogram(dataframe: pd.DataFrame, column: str) -> go.Figure:
    """Histogram for a selected numeric feature."""
    if column not in dataframe.columns:
        return empty_figure("Selected feature was not found.", "Histogram")

    if not _is_numeric(dataframe[column]):
        return empty_figure("Histogram requires a numeric feature.", "Histogram")

    fig = px.histogram(
        dataframe,
        x=column,
        nbins=min(30, max(10, dataframe[column].nunique())),
        color_discrete_sequence=[CHART_COLORS[2]],
    )
    fig.update_layout(xaxis_title=column, yaxis_title="Frequency")
    return apply_layout(fig, f"Histogram — {column}")


def create_boxplot(
    dataframe: pd.DataFrame,
    column: str,
    target_column: str | None = None,
) -> go.Figure:
    """Boxplot for a numeric feature, optionally grouped by target."""
    if column not in dataframe.columns:
        return empty_figure("Selected feature was not found.", "Boxplot")

    if not _is_numeric(dataframe[column]):
        return empty_figure("Boxplot requires a numeric feature.", "Boxplot")

    if target_column and target_column in dataframe.columns and target_column != column:
        fig = px.box(
            dataframe,
            x=target_column,
            y=column,
            color=target_column,
            color_discrete_sequence=CHART_COLORS,
        )
        fig.update_layout(xaxis_title=target_column, yaxis_title=column)
        return apply_layout(fig, f"Boxplot — {column} by {target_column}")

    fig = px.box(dataframe, y=column, color_discrete_sequence=[CHART_COLORS[3]])
    fig.update_layout(yaxis_title=column)
    return apply_layout(fig, f"Boxplot — {column}")


def create_scatter_plot(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str | None = None,
) -> go.Figure:
    """Interactive scatter plot for two numeric features."""
    for column in (x_column, y_column):
        if column not in dataframe.columns:
            return empty_figure("Selected axis feature was not found.", "Scatter Plot")
        if not _is_numeric(dataframe[column]):
            return empty_figure("Scatter plot requires numeric features.", "Scatter Plot")

    color_arg = None
    if color_column and color_column in dataframe.columns and color_column not in (x_column, y_column):
        color_arg = color_column

    fig = px.scatter(
        dataframe,
        x=x_column,
        y=y_column,
        color=color_arg,
        color_discrete_sequence=CHART_COLORS,
        opacity=0.75,
    )
    fig.update_traces(marker={"size": 8})
    fig.update_layout(xaxis_title=x_column, yaxis_title=y_column)
    title = f"Scatter Plot — {x_column} vs {y_column}"
    if color_arg:
        title += f" (coloured by {color_arg})"
    return apply_layout(fig, title)


def create_target_rate_by_category_chart(
    dataframe: pd.DataFrame,
    category_column: str,
    target_column: str,
) -> go.Figure:
    """Bar chart of positive class rate by categorical feature."""
    if category_column not in dataframe.columns or target_column not in dataframe.columns:
        return empty_figure("Required columns were not found.", "Readmission Rate by Category")

    grouped = (
        dataframe.groupby(category_column, dropna=False)[target_column]
        .agg(rate="mean", count="count")
        .reset_index()
        .sort_values("rate", ascending=False)
        .head(15)
    )
    grouped[category_column] = grouped[category_column].astype(str)

    fig = go.Figure(
        go.Bar(
            x=grouped[category_column],
            y=grouped["rate"] * 100,
            marker={"color": CHART_COLORS[0]},
            text=[f"{value:.1f}%" for value in grouped["rate"] * 100],
            textposition="outside",
            customdata=grouped["count"],
            hovertemplate="Category: %{x}<br>Readmission rate: %{y:.1f}%<br>Count: %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title=category_column, yaxis_title="Readmission rate (%)")
    return apply_layout(fig, f"Readmission Rate by {category_column}", height=460)


def create_categorical_overview_chart(dataframe: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of top categorical values across object columns."""
    summaries: list[tuple[str, str, int]] = []
    for column in dataframe.select_dtypes(include=["object", "category"]).columns:
        top_value = dataframe[column].astype(str).value_counts().head(1)
        if top_value.empty:
            continue
        summaries.append((column, str(top_value.index[0]), int(top_value.iloc[0])))

    if not summaries:
        return empty_figure("No categorical features found.", "Categorical Overview")

    summaries.sort(key=lambda item: item[2], reverse=True)
    summaries = summaries[:12]
    labels = [f"{column}: {value}" for column, value, _ in summaries][::-1]
    counts = [count for _, _, count in summaries][::-1]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker={"color": CHART_COLORS[1]},
            hovertemplate="Category: %{y}<br>Count: %{x}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Count", yaxis_title="Top category per feature")
    return apply_layout(fig, "Categorical Feature Overview", height=480)


def create_numeric_summary_chart(dataframe: pd.DataFrame) -> go.Figure:
    """Bar chart comparing mean values of numeric features."""
    numeric = dataframe.select_dtypes(include=[np.number])
    if numeric.empty:
        return empty_figure("No numeric features found.", "Numeric Feature Means")

    means = numeric.mean().sort_values(ascending=True).tail(15)
    fig = go.Figure(
        go.Bar(
            x=means.values,
            y=means.index.astype(str),
            orientation="h",
            marker={"color": CHART_COLORS[2]},
            hovertemplate="Feature: %{y}<br>Mean: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Mean value", yaxis_title="Feature")
    return apply_layout(fig, "Numeric Feature Means", height=480)
