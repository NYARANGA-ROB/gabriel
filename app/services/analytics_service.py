"""Analytics dashboard data aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.ml.comparison import build_model_comparison
from app.models.prediction_record import PredictionRecord
from app.models.trained_model import TrainedModel
from app.services.plotting.analytics_charts import (
    create_feature_importance_chart,
    create_monthly_statistics_chart,
    create_prediction_trend_chart,
    create_risk_distribution_chart,
)
from app.services.plotting.base import figure_to_dict


def _best_model_summary(user_id: int) -> dict[str, Any] | None:
    comparison = build_model_comparison(user_id)
    if not comparison.get("has_models"):
        return None

    best_id = comparison.get("best_model_id")
    if best_id is None:
        return None

    best_row = next((row for row in comparison["models"] if row["id"] == best_id), None)
    if best_row is None:
        return None

    return {
        "id": best_row["id"],
        "label": best_row["label"],
        "f1_score": best_row.get("f1_score"),
        "roc_auc": best_row.get("roc_auc"),
        "accuracy": best_row.get("accuracy"),
    }


def _feature_importance_for_model(model_id: int, user_id: int) -> list[tuple[str, float]]:
    trained_model = TrainedModel.query.filter_by(id=model_id, user_id=user_id, status="completed").first()
    if trained_model is None or not trained_model.model_file_path:
        return []

    artefact = joblib.load(trained_model.model_file_path)
    estimator = artefact["model"]
    feature_columns = artefact.get("feature_columns", [])

    if not feature_columns:
        return []

    if hasattr(estimator, "feature_importances_"):
        scores = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        scores = np.abs(coef.reshape(-1))
        if len(scores) != len(feature_columns):
            scores = np.abs(coef).mean(axis=0)
    else:
        return []

    if len(scores) != len(feature_columns):
        return []

    return list(zip(feature_columns, scores.tolist()))


def _daily_trends(records: list[PredictionRecord]) -> tuple[list[str], list[int], list[float]]:
    if not records:
        return [], [], []

    daily_counts: dict[str, int] = defaultdict(int)
    daily_probs: dict[str, list[float]] = defaultdict(list)

    for record in records:
        day = record.created_at.strftime("%Y-%m-%d")
        daily_counts[day] += 1
        daily_probs[day].append(record.probability)

    dates = sorted(daily_counts.keys())
    counts = [daily_counts[day] for day in dates]
    avg_probs = [float(np.mean(daily_probs[day])) for day in dates]
    return dates, counts, avg_probs


def _monthly_statistics(
    records: list[PredictionRecord],
) -> tuple[list[str], list[int], list[int], list[float]]:
    if not records:
        return [], [], [], []

    monthly_totals: dict[str, int] = defaultdict(int)
    monthly_high: dict[str, int] = defaultdict(int)
    monthly_probs: dict[str, list[float]] = defaultdict(list)

    for record in records:
        month = record.created_at.strftime("%b %Y")
        monthly_totals[month] += 1
        monthly_probs[month].append(record.probability)
        if record.risk_level == "High":
            monthly_high[month] += 1

    months = sorted(
        monthly_totals.keys(),
        key=lambda label: pd.to_datetime(label, format="%b %Y"),
    )
    totals = [monthly_totals[month] for month in months]
    high_risk = [monthly_high[month] for month in months]
    avg_probs = [float(np.mean(monthly_probs[month])) for month in months]
    return months, totals, high_risk, avg_probs


def build_analytics_dashboard(user_id: int) -> dict[str, Any]:
    """Assemble KPIs and Plotly chart payloads for the analytics page."""
    records = (
        PredictionRecord.query.filter_by(user_id=user_id)
        .order_by(PredictionRecord.created_at.asc())
        .all()
    )

    total_predictions = len(records)
    high_risk = sum(1 for record in records if record.risk_level == "High")
    low_risk = sum(1 for record in records if record.risk_level == "Low")
    moderate_risk = sum(1 for record in records if record.risk_level == "Moderate")
    avg_probability = float(np.mean([record.probability for record in records])) if records else 0.0

    best_model = _best_model_summary(user_id)
    feature_pairs = (
        _feature_importance_for_model(best_model["id"], user_id) if best_model else []
    )
    feature_names = [name for name, _ in feature_pairs]
    feature_scores = [score for _, score in feature_pairs]

    dates, daily_counts, daily_avg_probs = _daily_trends(records)
    months, monthly_totals, monthly_high, monthly_avg_probs = _monthly_statistics(records)

    charts = {
        "risk_distribution": figure_to_dict(
            create_risk_distribution_chart(high_risk, moderate_risk, low_risk)
        ),
        "prediction_trends": figure_to_dict(
            create_prediction_trend_chart(dates, daily_counts, daily_avg_probs)
        ),
        "monthly_statistics": figure_to_dict(
            create_monthly_statistics_chart(
                months,
                monthly_totals,
                monthly_high,
                monthly_avg_probs,
            )
        ),
        "feature_importance": figure_to_dict(
            create_feature_importance_chart(feature_names, feature_scores)
        ),
    }

    return {
        "has_data": total_predictions > 0,
        "kpis": {
            "total_predictions": total_predictions,
            "high_risk_patients": high_risk,
            "low_risk_patients": low_risk,
            "moderate_risk_patients": moderate_risk,
            "average_probability": avg_probability,
            "average_probability_percent": round(avg_probability * 100, 1),
        },
        "best_model": best_model,
        "top_features": [
            {"feature": name, "importance": round(score, 4)}
            for name, score in sorted(feature_pairs, key=lambda item: item[1], reverse=True)[:5]
        ],
        "charts": charts,
    }
