"""Model comparison charts."""

from __future__ import annotations

import plotly.graph_objects as go

from app.services.plotting.base import CHART_COLORS, apply_layout, empty_figure


def create_metric_comparison_chart(
    labels: list[str],
    values: list[float],
    title: str,
    yaxis_title: str,
    *,
    as_percentage: bool = True,
    best_index: int | None = None,
) -> go.Figure:
    """Grouped bar chart comparing a metric across models."""
    display_values = [value * 100 for value in values] if as_percentage else values
    colors = [
        CHART_COLORS[0] if index == best_index else "#94a3b8"
        for index in range(len(labels))
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=display_values,
            marker={"color": colors},
            text=[f"{value:.2f}{'%' if as_percentage else ''}" for value in display_values],
            textposition="outside",
            hovertemplate="%{x}<br>" + yaxis_title + ": %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Model", yaxis_title=yaxis_title)
    return apply_layout(fig, title, height=420)


def create_timing_comparison_chart(
    labels: list[str],
    values_ms: list[float],
    title: str,
    best_index: int | None = None,
) -> go.Figure:
    """Bar chart for training or prediction time in milliseconds."""
    colors = [
        CHART_COLORS[3] if index == best_index else "#94a3b8"
        for index in range(len(labels))
    ]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values_ms,
            marker={"color": colors},
            text=[f"{value:.2f} ms" for value in values_ms],
            textposition="outside",
            hovertemplate="%{x}<br>Time: %{y:.2f} ms<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Model", yaxis_title="Time (ms)")
    return apply_layout(fig, title, height=420)


def create_roc_comparison_chart(curves: list[dict]) -> go.Figure:
    """Overlay ROC curves from multiple trained models."""
    if not curves:
        return empty_figure("No ROC data available for comparison.", "ROC Comparison")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "#94a3b8"},
            name="Random baseline",
        )
    )

    for index, curve in enumerate(curves):
        fig.add_trace(
            go.Scatter(
                x=curve["fpr"],
                y=curve["tpr"],
                mode="lines",
                line={"color": CHART_COLORS[index % len(CHART_COLORS)], "width": 2.5},
                name=f"{curve['label']} (AUC = {curve['auc']:.3f})",
            )
        )

    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return apply_layout(fig, "ROC Curve Comparison", height=480)


def create_pr_comparison_chart(curves: list[dict]) -> go.Figure:
    """Overlay precision-recall curves from multiple trained models."""
    if not curves:
        return empty_figure("No precision-recall data available for comparison.", "PR Comparison")

    fig = go.Figure()
    for index, curve in enumerate(curves):
        fig.add_trace(
            go.Scatter(
                x=curve["recall"],
                y=curve["precision"],
                mode="lines",
                line={"color": CHART_COLORS[index % len(CHART_COLORS)], "width": 2.5},
                name=f"{curve['label']} (AUC = {curve['auc']:.3f})",
            )
        )

    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return apply_layout(fig, "Precision-Recall Comparison", height=480)
