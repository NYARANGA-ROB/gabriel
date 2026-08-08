"""Evaluation visualisation charts."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from app.services.plotting.base import CHART_COLORS, apply_layout, empty_figure


def create_confusion_matrix_chart(matrix: np.ndarray, labels: list[str]) -> go.Figure:
    """Heatmap for a confusion matrix."""
    labels = [str(label) for label in labels]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    return apply_layout(fig, "Confusion Matrix", height=460)


def create_roc_curve_chart(y_true, y_score, labels: list[str] | None = None) -> go.Figure:
    """ROC curve for binary or one-vs-rest multiclass targets."""
    classes = np.unique(y_true)
    if len(classes) < 2:
        return empty_figure("ROC curve requires at least two classes.", "ROC Curve")

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

    if len(classes) == 2 and y_score.ndim == 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                line={"color": CHART_COLORS[0], "width": 3},
                name=f"ROC (AUC = {roc_auc:.3f})",
            )
        )
    else:
        y_bin = label_binarize(y_true, classes=classes)
        for index, class_label in enumerate(classes):
            if y_bin.shape[1] <= index:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, index], y_score[:, index])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    line={"color": CHART_COLORS[index % len(CHART_COLORS)]},
                    name=f"Class {class_label} (AUC = {roc_auc:.3f})",
                )
            )

    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return apply_layout(fig, "ROC Curve", height=460)


def create_precision_recall_chart(y_true, y_score, labels: list[str] | None = None) -> go.Figure:
    """Precision-recall curve for binary or one-vs-rest multiclass targets."""
    classes = np.unique(y_true)
    if len(classes) < 2:
        return empty_figure("Precision-recall curve requires at least two classes.", "Precision-Recall Curve")

    fig = go.Figure()

    if len(classes) == 2 and y_score.ndim == 1:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                line={"color": CHART_COLORS[1], "width": 3},
                name=f"PR curve (AUC = {pr_auc:.3f})",
            )
        )
    else:
        y_bin = label_binarize(y_true, classes=classes)
        for index, class_label in enumerate(classes):
            if y_bin.shape[1] <= index:
                continue
            precision, recall, _ = precision_recall_curve(y_bin[:, index], y_score[:, index])
            pr_auc = auc(recall, precision)
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    line={"color": CHART_COLORS[index % len(CHART_COLORS)]},
                    name=f"Class {class_label} (AUC = {pr_auc:.3f})",
                )
            )

    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return apply_layout(fig, "Precision-Recall Curve", height=460)


def create_normalized_confusion_matrix_chart(matrix: np.ndarray, labels: list[str]) -> go.Figure:
    """Row-normalized confusion matrix heatmap (%)."""
    labels = [str(label) for label in labels]
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, where=row_sums != 0) * 100

    fig = go.Figure(
        go.Heatmap(
            z=normalized,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=np.round(normalized, 1),
            texttemplate="%{text}%",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Rate: %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    return apply_layout(fig, "Normalized Confusion Matrix (%)", height=460)


def create_calibration_chart(y_true, y_score) -> go.Figure:
    """Reliability diagram for predicted probabilities."""
    if y_score is None or len(np.unique(y_true)) != 2:
        return empty_figure("Calibration requires binary class probabilities.", "Calibration Plot")

    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(y_score, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    mean_predicted = []
    fraction_positive = []
    counts = []
    for index in range(len(bins) - 1):
        mask = bin_indices == index
        if not np.any(mask):
            continue
        mean_predicted.append(y_score[mask].mean())
        fraction_positive.append(y_true[mask].mean())
        counts.append(int(mask.sum()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "#94a3b8"},
            name="Perfect calibration",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mean_predicted,
            y=fraction_positive,
            mode="lines+markers",
            line={"color": CHART_COLORS[0], "width": 3},
            marker={"size": 10},
            name="Model",
            text=[f"n={count}" for count in counts],
            hovertemplate="Mean predicted: %{x:.2f}<br>Observed rate: %{y:.2f}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Fraction of positives")
    return apply_layout(fig, "Calibration Plot", height=460)


def create_feature_importance_chart(
    feature_names: list[str],
    importances: np.ndarray,
    *,
    top_n: int = 15,
) -> go.Figure:
    """Horizontal bar chart of model feature importances."""
    if len(feature_names) == 0 or len(importances) == 0:
        return empty_figure("Feature importance is not available for this model.", "Feature Importance")

    pairs = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)[:top_n]
    names = [name for name, _ in pairs][::-1]
    values = [value for _, value in pairs][::-1]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker={"color": CHART_COLORS[0]},
            hovertemplate="Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
    return apply_layout(fig, "Top Feature Importance", height=480)


def create_prediction_distribution_chart(y_score) -> go.Figure:
    """Histogram of predicted positive probabilities by outcome."""
    if y_score is None:
        return empty_figure("Prediction scores are not available.", "Prediction Distribution")

    fig = go.Figure(
        go.Histogram(
            x=y_score,
            nbinsx=25,
            marker={"color": CHART_COLORS[2]},
            hovertemplate="Score: %{x}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Predicted readmission probability", yaxis_title="Count")
    return apply_layout(fig, "Predicted Probability Distribution", height=420)
