"""Prediction history storage, search, and export."""

from __future__ import annotations

import csv
import io
from typing import Any

from flask import current_app

from app.models.prediction_record import PredictionRecord
from app.models.trained_model import TrainedModel
from app.repositories.prediction_filters import PredictionFilters
from app.repositories.prediction_repository import PredictionRepository
from app.services.prediction_history_pdf import generate_history_pdf


class PredictionHistoryError(Exception):
    """Raised when prediction history operations fail."""


def save_prediction_record(
    user_id: int,
    patient_id: str,
    trained_model_id: int,
    result: dict[str, Any],
) -> PredictionRecord:
    """Persist a prediction result to history."""
    return PredictionRepository.save(user_id, patient_id, trained_model_id, result)


def query_prediction_history(
    user_id: int,
    *,
    page: int = 1,
    per_page: int = 20,
    search: str | None = None,
    model_id: int | None = None,
    risk_level: str | None = None,
):
    """Return a paginated prediction history query for a user."""
    filters = PredictionFilters(search=search, model_id=model_id, risk_level=risk_level)
    return PredictionRepository.paginate(
        user_id=user_id,
        page=page,
        per_page=per_page,
        filters=filters,
    )


def get_user_prediction_record(record_id: int, user_id: int) -> PredictionRecord | None:
    return PredictionRepository.get_for_user(record_id, user_id)


def delete_prediction_record(record_id: int, user_id: int) -> bool:
    """Delete a single prediction history record."""
    return PredictionRepository.delete_for_user(record_id, user_id)


def _fetch_records(
    user_id: int,
    *,
    search: str | None = None,
    model_id: int | None = None,
    risk_level: str | None = None,
) -> list[PredictionRecord]:
    filters = PredictionFilters(search=search, model_id=model_id, risk_level=risk_level)
    max_rows = current_app.config.get("PREDICTION_EXPORT_MAX_ROWS", 5000)
    records = PredictionRepository.fetch_all(user_id=user_id, filters=filters, limit=max_rows + 1)
    if len(records) > max_rows:
        raise PredictionHistoryError(
            f"Export limited to {max_rows:,} rows. Narrow your filters and try again."
        )
    return records


def export_history_csv(
    user_id: int,
    *,
    search: str | None = None,
    model_id: int | None = None,
    risk_level: str | None = None,
) -> bytes:
    """Export filtered prediction history as CSV bytes."""
    records = _fetch_records(
        user_id,
        search=search,
        model_id=model_id,
        risk_level=risk_level,
    )

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "Patient ID",
            "Prediction",
            "Probability (%)",
            "Risk Level",
            "Model",
            "Date (UTC)",
        ]
    )
    for record in records:
        writer.writerow(
            [
                record.patient_id,
                record.prediction_label,
                round(record.probability * 100, 2),
                record.risk_level,
                record.model_name,
                record.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            ]
        )

    return buffer.getvalue().encode("utf-8")


def export_history_pdf(
    user_id: int,
    app_name: str,
    *,
    search: str | None = None,
    model_id: int | None = None,
    risk_level: str | None = None,
) -> bytes:
    """Export filtered prediction history as PDF bytes."""
    records = _fetch_records(
        user_id,
        search=search,
        model_id=model_id,
        risk_level=risk_level,
    )
    return generate_history_pdf(records, app_name)


def get_history_filter_models(user_id: int) -> list[TrainedModel]:
    """Return models referenced in the user's prediction history."""
    return PredictionRepository.models_for_filter(user_id=user_id)
