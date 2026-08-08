"""Comprehensive model evaluation."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from app.extensions import db
from app.models.model_evaluation import ModelEvaluation
from app.models.trained_model import TrainedModel
from app.services.plotting.base import figure_to_dict
from app.services.plotting.evaluation_charts import (
    create_calibration_chart,
    create_confusion_matrix_chart,
    create_feature_importance_chart,
    create_normalized_confusion_matrix_chart,
    create_precision_recall_chart,
    create_prediction_distribution_chart,
    create_roc_curve_chart,
)


def _compute_scores(estimator, x_test, y_test) -> tuple[np.ndarray, np.ndarray | None]:
    y_pred = estimator.predict(x_test)
    y_score = None
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_test)
        classes = np.unique(y_test)
        if len(classes) == 2:
            positive_index = 1 if probabilities.shape[1] > 1 else 0
            y_score = probabilities[:, positive_index]
        else:
            y_score = probabilities
    return y_pred, y_score


def _compute_roc_auc(y_true, y_score) -> float | None:
    classes = np.unique(y_true)
    if y_score is None or len(classes) < 2:
        return None

    try:
        if len(classes) == 2 and y_score.ndim == 1:
            return float(roc_auc_score(y_true, y_score))
        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted"))
    except ValueError:
        return None


def _extract_feature_importance(estimator, feature_names: list[str] | None) -> np.ndarray | None:
    if not feature_names:
        return None

    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_)

    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_)
        if coefficients.ndim > 1:
            return np.mean(np.abs(coefficients), axis=0)
        return np.abs(coefficients)

    return None


def evaluate_model(
    estimator,
    x_test,
    y_test,
    training_time_seconds: float,
    feature_names: list[str] | None = None,
) -> dict:
    """Evaluate a trained estimator and build metrics plus chart payloads."""
    predict_start = time.perf_counter()
    y_pred, y_score = _compute_scores(estimator, x_test, y_test)
    prediction_time_seconds = time.perf_counter() - predict_start

    labels_sorted = sorted(pd.Series(y_test).unique(), key=lambda value: str(value))
    labels = [str(label) for label in labels_sorted]
    matrix = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "roc_auc": _compute_roc_auc(y_test, y_score),
        "training_time_seconds": float(training_time_seconds),
        "prediction_time_seconds": float(prediction_time_seconds),
        "test_samples": int(len(y_test)),
    }

    charts = {
        "confusion_matrix": figure_to_dict(create_confusion_matrix_chart(matrix, labels)),
        "normalized_confusion_matrix": figure_to_dict(
            create_normalized_confusion_matrix_chart(matrix, labels)
        ),
        "prediction_distribution": figure_to_dict(create_prediction_distribution_chart(y_score)),
    }

    importances = _extract_feature_importance(estimator, feature_names)
    if importances is not None and feature_names:
        charts["feature_importance"] = figure_to_dict(
            create_feature_importance_chart(feature_names, importances)
        )
    else:
        charts["feature_importance"] = None

    if y_score is not None and len(labels) >= 2:
        charts["roc_curve"] = figure_to_dict(create_roc_curve_chart(y_test, y_score, labels))
        charts["precision_recall_curve"] = figure_to_dict(
            create_precision_recall_chart(y_test, y_score, labels)
        )
        charts["calibration"] = figure_to_dict(create_calibration_chart(y_test, y_score))
    else:
        charts["roc_curve"] = None
        charts["precision_recall_curve"] = None
        charts["calibration"] = None

    return {
        "metrics": metrics,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix.tolist(),
        },
        "charts": charts,
    }


def save_evaluation_record(
    trained_model: TrainedModel,
    evaluation_result: dict,
) -> ModelEvaluation:
    """Persist an evaluation run to history."""
    metrics = evaluation_result["metrics"]
    record = ModelEvaluation(
        trained_model_id=trained_model.id,
        user_id=trained_model.user_id,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        roc_auc=metrics.get("roc_auc"),
        training_time_seconds=metrics["training_time_seconds"],
        prediction_time_seconds=metrics["prediction_time_seconds"],
    )
    record.metrics = metrics
    record.confusion_matrix = evaluation_result["confusion_matrix"]
    record.charts = evaluation_result["charts"]

    db.session.add(record)
    db.session.commit()
    return record
