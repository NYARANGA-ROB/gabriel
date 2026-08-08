"""Shared Plotly layout and serialization helpers."""

from __future__ import annotations

import json
from typing import Any

import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

CHART_HEIGHT = 420
CHART_COLORS = [
    "#2563eb",
    "#16a34a",
    "#dc2626",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#be185d",
]


def apply_layout(fig: go.Figure, title: str, *, height: int = CHART_HEIGHT) -> go.Figure:
    """Apply consistent styling to a Plotly figure."""
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        height=height,
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def figure_to_dict(fig: go.Figure) -> dict[str, Any]:
    """Serialize a Plotly figure for client-side rendering."""
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))


def empty_figure(message: str, title: str) -> go.Figure:
    """Create a placeholder figure when a chart cannot be rendered."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "#64748b"},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_layout(fig, title, height=320)
