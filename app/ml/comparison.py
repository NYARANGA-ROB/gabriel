"""Build model comparison data and charts."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from app.ml.evaluation import _compute_scores
from app.models.trained_model import TrainedModel
from app.services.plotting.base import figure_to_dict
from app.services.plotting.comparison_charts import (
    create_metric_comparison_chart,
    create_pr_comparison_chart,
    create_roc_comparison_chart,
    create_timing_comparison_chart,
)


class ModelComparisonError(Exception):
    """Raised when comparison cannot be generated."""


METRIC_FIELDS = [
    ("accuracy", "Accuracy", True),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("f1_score", "F1 Score", True),
    ("roc_auc", "ROC AUC", False),
]


def _model_label(trained_model: TrainedModel) -> str:
    return f"{trained_model.model_name} #{trained_model.id}"


def _load_comparison_models(user_id: int) -> list[TrainedModel]:
    models = (
        TrainedModel.query.filter_by(user_id=user_id, status="completed")
        .order_by(TrainedModel.created_at.desc())
        .all()
    )
    return [model for model in models if model.latest_evaluation is not None]


def _compute_roc_curve(model: TrainedModel) -> dict | None:
    if not model.model_file_path:
        return None

    artefact = joblib.load(model.model_file_path)
    estimator = artefact["model"]
    processed = model.processed_dataset
    test_df = pd.read_csv(processed.test_file_path)
    target_column = artefact.get("target_column", processed.target_column)
    feature_columns = artefact.get("feature_columns")

    x_test = test_df[feature_columns]
    y_test = test_df[target_column]
    _, y_score = _compute_scores(estimator, x_test, y_test)

    if y_score is None or len(np.unique(y_test)) != 2 or getattr(y_score, "ndim", 1) != 1:
        return None

    fpr, tpr, _ = roc_curve(y_test, y_score)
    return {
        "label": _model_label(model),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": float(auc(fpr, tpr)),
    }


def _compute_pr_curve(model: TrainedModel) -> dict | None:
    if not model.model_file_path:
        return None

    artefact = joblib.load(model.model_file_path)
    estimator = artefact["model"]
    processed = model.processed_dataset
    test_df = pd.read_csv(processed.test_file_path)
    target_column = artefact.get("target_column", processed.target_column)
    feature_columns = artefact.get("feature_columns")

    x_test = test_df[feature_columns]
    y_test = test_df[target_column]
    _, y_score = _compute_scores(estimator, x_test, y_test)

    if y_score is None or len(np.unique(y_test)) != 2 or getattr(y_score, "ndim", 1) != 1:
        return None

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    return {
        "label": _model_label(model),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "auc": float(auc(recall, precision)),
    }


def _best_index(values: list[float | None], higher_is_better: bool = True) -> int | None:
    valid = [(index, value) for index, value in enumerate(values) if value is not None]
    if not valid:
        return None
    return max(valid, key=lambda item: item[1])[0] if higher_is_better else min(valid, key=lambda item: item[1])[0]


def _select_best_overall(rows: list[dict]) -> int | None:
    if not rows:
        return None

    def sort_key(row: dict) -> tuple:
        return (
            row.get("f1_score") or -1,
            row.get("roc_auc") or -1,
            row.get("accuracy") or -1,
        )

    return max(rows, key=sort_key)["id"]


def _row_index_by_id(rows: list[dict], model_id: int | None) -> int | None:
    if model_id is None:
        return None
    for index, row in enumerate(rows):
        if row["id"] == model_id:
            return index
    return None


def build_model_comparison(user_id: int) -> dict:
    """Assemble comparison table rows, best-model flags, and chart payloads."""
    trained_models = _load_comparison_models(user_id)
    if not trained_models:
        return {
            "models": [],
            "best_model_id": None,
            "best_per_metric": {},
            "charts": {},
            "has_models": False,
        }

    rows: list[dict] = []
    for model in trained_models:
        evaluation = model.latest_evaluation
        rows.append(
            {
                "id": model.id,
                "label": _model_label(model),
                "model_name": model.model_name,
                "model_type": model.model_type,
                "dataset": model.processed_dataset.dataset.original_filename,
                "target": model.processed_dataset.target_column,
                "accuracy": evaluation.accuracy,
                "precision": evaluation.precision,
                "recall": evaluation.recall,
                "f1_score": evaluation.f1_score,
                "roc_auc": evaluation.roc_auc,
                "training_time_seconds": evaluation.training_time_seconds,
                "prediction_time_seconds": evaluation.prediction_time_seconds,
                "training_time_ms": evaluation.training_time_seconds * 1000,
                "prediction_time_ms": evaluation.prediction_time_seconds * 1000,
                "evaluated_at": evaluation.evaluated_at,
            }
        )

    labels = [row["label"] for row in rows]
    best_per_metric: dict[str, int | None] = {}

    for field, _, as_percentage in METRIC_FIELDS:
        values = [row[field] for row in rows]
        best_per_metric[field] = (
            rows[_best_index(values)]["id"] if _best_index(values) is not None else None
        )

    best_per_metric["training_time_seconds"] = (
        rows[_best_index([row["training_time_seconds"] for row in rows], higher_is_better=False)]["id"]
        if rows
        else None
    )
    best_per_metric["prediction_time_seconds"] = (
        rows[_best_index([row["prediction_time_seconds"] for row in rows], higher_is_better=False)]["id"]
        if rows
        else None
    )

    best_model_id = _select_best_overall(rows)
    for row in rows:
        row["is_best_overall"] = row["id"] == best_model_id
        row["best_metrics"] = [
            metric
            for metric, model_id in best_per_metric.items()
            if model_id == row["id"]
        ]

    roc_curves = []
    pr_curves = []
    for model in trained_models:
        try:
            curve = _compute_roc_curve(model)
            if curve:
                roc_curves.append(curve)
        except Exception:
            continue
        try:
            pr_curve = _compute_pr_curve(model)
            if pr_curve:
                pr_curves.append(pr_curve)
        except Exception:
            continue

    charts = {
        "roc_comparison": figure_to_dict(create_roc_comparison_chart(roc_curves)),
        "pr_comparison": figure_to_dict(create_pr_comparison_chart(pr_curves)),
        "roc_auc": figure_to_dict(
            create_metric_comparison_chart(
                labels,
                [row["roc_auc"] or 0 for row in rows],
                "ROC AUC Comparison",
                "ROC AUC",
                as_percentage=False,
                best_index=_row_index_by_id(rows, best_per_metric.get("roc_auc")),
            )
        ),
        "accuracy": figure_to_dict(
            create_metric_comparison_chart(
                labels,
                [row["accuracy"] for row in rows],
                "Accuracy Comparison",
                "Accuracy (%)",
                best_index=_row_index_by_id(rows, best_per_metric.get("accuracy")),
            )
        ),
        "precision": figure_to_dict(
            create_metric_comparison_chart(
                labels,
                [row["precision"] for row in rows],
                "Precision Comparison",
                "Precision (%)",
                best_index=_row_index_by_id(rows, best_per_metric.get("precision")),
            )
        ),
        "recall": figure_to_dict(
            create_metric_comparison_chart(
                labels,
                [row["recall"] for row in rows],
                "Recall Comparison",
                "Recall (%)",
                best_index=_row_index_by_id(rows, best_per_metric.get("recall")),
            )
        ),
        "f1_score": figure_to_dict(
            create_metric_comparison_chart(
                labels,
                [row["f1_score"] for row in rows],
                "F1 Score Comparison",
                "F1 Score (%)",
                best_index=_row_index_by_id(rows, best_per_metric.get("f1_score")),
            )
        ),
        "training_time": figure_to_dict(
            create_timing_comparison_chart(
                labels,
                [row["training_time_ms"] for row in rows],
                "Training Time Comparison",
                best_index=_row_index_by_id(rows, best_per_metric.get("training_time_seconds")),
            )
        ),
        "prediction_time": figure_to_dict(
            create_timing_comparison_chart(
                labels,
                [row["prediction_time_ms"] for row in rows],
                "Prediction Time Comparison",
                best_index=_row_index_by_id(rows, best_per_metric.get("prediction_time_seconds")),
            )
        ),
    }

    return {
        "models": rows,
        "best_model_id": best_model_id,
        "best_per_metric": best_per_metric,
        "charts": charts,
        "has_models": True,
    }
