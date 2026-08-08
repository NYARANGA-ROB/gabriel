"""Plotly charts for the analytics dashboard."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.services.plotting.base import CHART_COLORS, apply_layout, empty_figure


def create_risk_distribution_chart(high: int, moderate: int, low: int) -> go.Figure:
    """Donut chart of risk level distribution."""
    total = high + moderate + low
    if total == 0:
        return empty_figure("No prediction data available.", "Risk Distribution")

    fig = go.Figure(
        go.Pie(
            labels=["High", "Moderate", "Low"],
            values=[high, moderate, low],
            hole=0.45,
            marker={"colors": [CHART_COLORS[2], CHART_COLORS[3], CHART_COLORS[1]]},
            textinfo="label+percent",
            hovertemplate="%{label}<br>Count: %{value}<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=True)
    return apply_layout(fig, "Risk Level Distribution", height=380)


def create_prediction_trend_chart(
    dates: list[str],
    counts: list[int],
    avg_probabilities: list[float],
) -> go.Figure:
    """Dual-axis trend of daily predictions and average probability."""
    if not dates:
        return empty_figure("No prediction trend data available.", "Prediction Trends")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=counts,
            mode="lines+markers",
            name="Predictions",
            line={"color": CHART_COLORS[0], "width": 2.5},
            marker={"size": 7},
            hovertemplate="%{x}<br>Predictions: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[value * 100 for value in avg_probabilities],
            mode="lines+markers",
            name="Avg probability (%)",
            line={"color": CHART_COLORS[2], "width": 2, "dash": "dot"},
            marker={"size": 6},
            hovertemplate="%{x}<br>Avg probability: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Daily predictions", secondary_y=False)
    fig.update_yaxes(title_text="Average probability (%)", secondary_y=True)
    return apply_layout(fig, "Prediction Trends", height=420)


def create_monthly_statistics_chart(
    months: list[str],
    totals: list[int],
    high_risk_counts: list[int],
    avg_probabilities: list[float],
) -> go.Figure:
    """Monthly prediction volume and risk statistics."""
    if not months:
        return empty_figure("No monthly statistics available.", "Monthly Statistics")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=months,
            y=totals,
            name="Total predictions",
            marker_color=CHART_COLORS[0],
            hovertemplate="%{x}<br>Predictions: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=months,
            y=high_risk_counts,
            name="High risk",
            marker_color=CHART_COLORS[2],
            hovertemplate="%{x}<br>High risk: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=[value * 100 for value in avg_probabilities],
            mode="lines+markers",
            name="Avg probability (%)",
            line={"color": CHART_COLORS[3], "width": 2.5},
            marker={"size": 7},
            hovertemplate="%{x}<br>Avg probability: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Patient count", secondary_y=False)
    fig.update_yaxes(title_text="Average probability (%)", secondary_y=True)
    return apply_layout(fig, "Monthly Statistics", height=440)


def create_feature_importance_chart(
    feature_names: list[str],
    importances: list[float],
) -> go.Figure:
    """Horizontal bar chart of top model features."""
    if not feature_names:
        return empty_figure("No feature importance data available.", "Most Important Features")

    pairs = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)[:10]
    labels = [item[0] for item in reversed(pairs)]
    values = [item[1] for item in reversed(pairs)]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": CHART_COLORS[4]},
            text=[f"{value:.4f}" for value in values],
            textposition="outside",
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Importance score", yaxis_title="")
    return apply_layout(fig, "Most Important Features", height=420)
