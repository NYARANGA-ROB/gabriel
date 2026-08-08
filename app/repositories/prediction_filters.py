"""Shared prediction log query filters."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Query

from app.models.prediction_record import PredictionRecord


@dataclass(frozen=True)
class PredictionFilters:
    """Filter criteria for prediction history queries."""

    search: str | None = None
    model_id: int | None = None
    risk_level: str | None = None

    @classmethod
    def from_request_args(cls, args) -> PredictionFilters:
        """Build filters from a Flask request args mapping."""
        model_id = args.get("model_id", type=int)
        if model_id == 0:
            model_id = None
        risk_level = args.get("risk_level") or None
        search = args.get("search") or None
        if risk_level == "":
            risk_level = None
        return cls(search=search, model_id=model_id, risk_level=risk_level)

    def as_export_kwargs(self) -> dict[str, str | int | None]:
        return {
            "search": self.search,
            "model_id": self.model_id,
            "risk_level": self.risk_level,
        }


def apply_prediction_filters(query: Query, filters: PredictionFilters) -> Query:
    """Apply shared search and filter predicates to a prediction query."""
    if filters.search:
        term = f"%{filters.search.strip()}%"
        query = query.filter(
            or_(
                PredictionRecord.patient_id.ilike(term),
                PredictionRecord.model_name.ilike(term),
                PredictionRecord.prediction_label.ilike(term),
            )
        )

    if filters.model_id:
        query = query.filter(PredictionRecord.trained_model_id == filters.model_id)

    if filters.risk_level:
        query = query.filter(PredictionRecord.risk_level == filters.risk_level)

    return query
