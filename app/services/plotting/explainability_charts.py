"""LIME and SHAP explanation charts rendered with Plotly."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from app.services.plotting.base import CHART_COLORS, apply_layout, empty_figure


def create_shap_summary_chart(
    shap_matrix: np.ndarray,
    feature_names: list[str],
    feature_matrix: np.ndarray,
) -> go.Figure:
    """Beeswarm-style summary plot of SHAP values."""
    if shap_matrix.size == 0 or not feature_names:
        return empty_figure("No SHAP values available.", "SHAP Summary Plot")

    mean_abs = np.abs(shap_matrix).mean(axis=0)
    order = np.argsort(mean_abs)
    ordered_names = [feature_names[index] for index in order]
    ordered_shap = shap_matrix[:, order]
    ordered_features = feature_matrix[:, order]

    fig = go.Figure()
    rng = np.random.default_rng(42)

    for position, name in enumerate(ordered_names):
        values = ordered_shap[:, position]
        feature_values = ordered_features[:, position]
        jitter = rng.uniform(-0.25, 0.25, size=len(values))
        fig.add_trace(
            go.Scatter(
                x=values,
                y=[position] * len(values) + jitter,
                mode="markers",
                marker={
                    "size": 8,
                    "color": feature_values,
                    "colorscale": "RdBu",
                    "showscale": position == len(ordered_names) - 1,
                    "colorbar": {"title": "Feature value"},
                },
                name=name,
                hovertemplate=f"{name}<br>SHAP: %{{x:.4f}}<extra></extra>",
            )
        )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(ordered_names))),
        ticktext=ordered_names,
        title="Feature",
    )
    fig.update_xaxes(title="SHAP value (impact on readmission risk)")
    return apply_layout(fig, "SHAP Summary Plot", height=520)


def create_shap_importance_chart(
    shap_matrix: np.ndarray,
    feature_names: list[str],
) -> go.Figure:
    """Mean absolute SHAP value bar chart."""
    if shap_matrix.size == 0 or not feature_names:
        return empty_figure("No feature importance data available.", "Feature Importance")

    importance = np.abs(shap_matrix).mean(axis=0)
    order = np.argsort(importance)
    labels = [feature_names[index] for index in order]
    values = importance[order]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": CHART_COLORS[0]},
            text=[f"{value:.4f}" for value in values],
            textposition="outside",
            hovertemplate="%{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Mean |SHAP value|", yaxis_title="")
    return apply_layout(fig, "Feature Importance", height=460)


def create_shap_waterfall_chart(
    shap_row: np.ndarray,
    feature_names: list[str],
    base_value: float,
    feature_values: np.ndarray,
) -> go.Figure:
    """Waterfall chart for a single prediction."""
    if shap_row.size == 0 or not feature_names:
        return empty_figure("No local SHAP values available.", "SHAP Waterfall Plot")

    order = np.argsort(np.abs(shap_row))[::-1]
    labels = [feature_names[index] for index in order]
    values = shap_row[order]
    display_values = feature_values[order]

    hover = [
        f"{label}<br>Value: {value:.3f}<br>SHAP: {shap:.4f}"
        for label, value, shap in zip(labels, display_values, values)
    ]

    fig = go.Figure(
        go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=["relative"] * len(values) + ["total"],
            x=labels + ["Prediction"],
            y=list(values) + [float(np.sum(values))],
            base=base_value,
            text=[f"{value:+.4f}" for value in values] + [""],
            textposition="outside",
            connector={"line": {"color": "#cbd5e1"}},
            increasing={"marker": {"color": CHART_COLORS[2]}},
            decreasing={"marker": {"color": CHART_COLORS[1]}},
            totals={"marker": {"color": CHART_COLORS[0]}},
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Feature", yaxis_title="Model output")
    return apply_layout(fig, "SHAP Waterfall Plot", height=500)


def create_shap_force_chart(
    shap_row: np.ndarray,
    feature_names: list[str],
    base_value: float,
    feature_values: np.ndarray,
) -> go.Figure:
    """Force-style contribution plot for a single prediction."""
    if shap_row.size == 0 or not feature_names:
        return empty_figure("No local SHAP values available.", "SHAP Force Plot")

    order = np.argsort(np.abs(shap_row))[::-1][:12]
    labels = [feature_names[index] for index in order]
    values = shap_row[order]
    colors = [CHART_COLORS[2] if value >= 0 else CHART_COLORS[1] for value in values]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker={"color": colors},
            text=[f"{feature_values[index]:.2f}" for index in order],
            textposition="outside",
            hovertemplate="%{y}<br>SHAP: %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#94a3b8")
    fig.update_layout(
        xaxis_title="SHAP contribution",
        yaxis_title="",
        annotations=[
            {
                "text": f"Base value: {base_value:.4f}",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.08,
                "showarrow": False,
                "font": {"size": 12, "color": "#64748b"},
            }
        ],
    )
    return apply_layout(fig, "SHAP Force Plot", height=480)


def create_shap_dependence_chart(
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    feature_name: str,
    interaction_values: np.ndarray | None = None,
    interaction_name: str | None = None,
) -> go.Figure:
    """Dependence plot for one feature."""
    if feature_values.size == 0:
        return empty_figure("No dependence data available.", "SHAP Dependence Plot")

    marker_kwargs: dict = {"size": 9, "color": CHART_COLORS[0]}
    if interaction_values is not None and interaction_name:
        marker_kwargs = {
            "size": 9,
            "color": interaction_values,
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": interaction_name},
        }

    fig = go.Figure(
        go.Scatter(
            x=feature_values,
            y=shap_values,
            mode="markers",
            marker=marker_kwargs,
            hovertemplate=(
                f"{feature_name}: %{{x}}<br>SHAP: %{{y:.4f}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis_title=feature_name,
        yaxis_title=f"SHAP value for {feature_name}",
    )
    title = f"SHAP Dependence Plot — {feature_name}"
    if interaction_name:
        title += f" (coloured by {interaction_name})"
    return apply_layout(fig, title, height=460)


def create_lime_weights_chart(
    weights_list: list[tuple[str, float]],
    patient_label: str,
) -> go.Figure:
    """Bar chart of LIME feature weights."""
    if not weights_list:
        return empty_figure("No LIME weights available.", "LIME Feature Weights")

    labels = [item[0] for item in weights_list]
    values = [item[1] for item in weights_list]
    colors = [CHART_COLORS[2] if value >= 0 else CHART_COLORS[1] for value in values]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker={"color": colors},
            text=[f"{value:+.4f}" for value in values],
            textposition="outside",
            hovertemplate="%{y}<br>Weight: %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#94a3b8")
    fig.update_layout(xaxis_title="LIME weight", yaxis_title="")
    return apply_layout(fig, f"LIME Feature Weights — {patient_label}", height=460)


def create_lime_interactive_chart(
    weights_list: list[tuple[str, float]],
    feature_values: np.ndarray,
    feature_names: list[str],
    patient_label: str,
) -> go.Figure:
    """Interactive LIME explanation with feature values on hover."""
    if not weights_list:
        return empty_figure("No LIME explanation available.", "Interactive LIME Explanation")

    labels = [item[0] for item in weights_list]
    values = [item[1] for item in weights_list]
    colors = [CHART_COLORS[2] if value >= 0 else CHART_COLORS[1] for value in values]

    value_lookup = {name: feature_values[index] for index, name in enumerate(feature_names)}
    hover = []
    for label, weight in weights_list:
        matched_value = next(
            (value_lookup[name] for name in value_lookup if label == name or label.startswith(f"{name} ")),
            None,
        )
        direction = "increases risk" if weight >= 0 else "decreases risk"
        hover.append(
            f"{label}<br>Weight: {weight:+.4f}<br>Value: {matched_value:.3f}<br>{direction}"
            if matched_value is not None
            else f"{label}<br>Weight: {weight:+.4f}<br>{direction}"
        )

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker={"color": colors},
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#94a3b8")
    fig.update_layout(
        xaxis_title="Contribution to readmission risk",
        yaxis_title="",
        clickmode="event+select",
    )
    return apply_layout(fig, f"Interactive LIME Explanation — {patient_label}", height=500)


def create_shap_lime_comparison_chart(
    feature_names: list[str],
    shap_values: np.ndarray,
    lime_values: np.ndarray,
    feature_values: np.ndarray,
    top_n: int = 8,
) -> go.Figure:
    """Grouped comparison of SHAP and LIME contributions."""
    if shap_values.size == 0 or lime_values.size == 0:
        return empty_figure("No comparison data available.", "SHAP vs LIME Comparison")

    combined = np.abs(shap_values) + np.abs(lime_values)
    order = np.argsort(combined)[::-1][:top_n]
    labels = [feature_names[index] for index in order]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="SHAP",
            x=labels,
            y=shap_values[order],
            marker_color=CHART_COLORS[0],
            text=[f"{value:+.3f}" for value in shap_values[order]],
            textposition="outside",
            hovertemplate="%{x}<br>SHAP: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="LIME",
            x=labels,
            y=lime_values[order],
            marker_color=CHART_COLORS[3],
            text=[f"{value:+.3f}" for value in lime_values[order]],
            textposition="outside",
            hovertemplate="%{x}<br>LIME: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Feature",
        yaxis_title="Contribution",
        legend={"orientation": "h", "y": 1.1},
    )
    return apply_layout(fig, "SHAP vs LIME Comparison", height=500)
