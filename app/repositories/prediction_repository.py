"""Prediction record data access."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import joinedload

from app.extensions import db
from app.models.prediction_record import PredictionRecord
from app.models.trained_model import TrainedModel
from app.repositories.prediction_filters import PredictionFilters, apply_prediction_filters


class PredictionRepository:
    """Encapsulates prediction history persistence and queries."""

    @staticmethod
    def base_query(*, user_id: int | None = None):
        query = PredictionRecord.query.order_by(PredictionRecord.created_at.desc())
        if user_id is not None:
            query = query.filter_by(user_id=user_id)
        return query

    @classmethod
    def paginate(
        cls,
        *,
        user_id: int | None = None,
        page: int = 1,
        per_page: int = 20,
        filters: PredictionFilters | None = None,
        eager_load_user: bool = False,
    ):
        query = cls.base_query(user_id=user_id)
        if eager_load_user:
            query = query.options(joinedload(PredictionRecord.user))
        if filters:
            query = apply_prediction_filters(query, filters)
        return query.paginate(page=max(page, 1), per_page=per_page, error_out=False)

    @classmethod
    def fetch_all(
        cls,
        *,
        user_id: int,
        filters: PredictionFilters | None = None,
        limit: int | None = None,
    ) -> list[PredictionRecord]:
        query = cls.base_query(user_id=user_id)
        if filters:
            query = apply_prediction_filters(query, filters)
        if limit is not None:
            query = query.limit(limit)
        return query.all()

    @staticmethod
    def get_for_user(record_id: int, user_id: int) -> PredictionRecord | None:
        return PredictionRecord.query.filter_by(id=record_id, user_id=user_id).first()

    @staticmethod
    def delete_for_user(record_id: int, user_id: int) -> bool:
        record = PredictionRepository.get_for_user(record_id, user_id)
        if record is None:
            return False
        db.session.delete(record)
        db.session.commit()
        return True

    @staticmethod
    def distinct_model_ids(*, user_id: int | None = None) -> list[int]:
        query = db.session.query(PredictionRecord.trained_model_id).distinct()
        if user_id is not None:
            query = query.filter_by(user_id=user_id)
        return [row[0] for row in query.all()]

    @staticmethod
    def models_for_filter(*, user_id: int | None = None) -> list[TrainedModel]:
        ids = PredictionRepository.distinct_model_ids(user_id=user_id)
        if not ids:
            return []
        return (
            TrainedModel.query.filter(TrainedModel.id.in_(ids))
            .order_by(TrainedModel.created_at.desc())
            .all()
        )

    @staticmethod
    def save(
        user_id: int,
        patient_id: str,
        trained_model_id: int,
        result: dict[str, Any],
    ) -> PredictionRecord:
        record = PredictionRecord(
            user_id=user_id,
            trained_model_id=trained_model_id,
            patient_id=patient_id.strip(),
            prediction=str(result.get("prediction", "")),
            prediction_label=result.get("readmission_label", "Unknown"),
            probability=float(result.get("probability", 0.0)),
            risk_level=result.get("risk_level", "Unknown"),
            model_name=result.get("model_label", result.get("model_name", "Unknown")),
        )
        db.session.add(record)
        db.session.commit()
        return record
