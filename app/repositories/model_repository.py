"""Trained model data access with eager loading."""

from __future__ import annotations

from sqlalchemy import desc
from sqlalchemy.orm import joinedload

from app.models.dataset import Dataset
from app.models.model_evaluation import ModelEvaluation
from app.models.processed_dataset import ProcessedDataset
from app.models.trained_model import TrainedModel


class ModelRepository:
    """Encapsulates trained model queries and relationship loading."""

    @staticmethod
    def _with_dataset_chain():
        return joinedload(TrainedModel.processed_dataset).joinedload(ProcessedDataset.dataset)

    @classmethod
    def list_for_user(cls, user_id: int) -> list[TrainedModel]:
        return (
            TrainedModel.query.filter_by(user_id=user_id)
            .options(cls._with_dataset_chain())
            .order_by(TrainedModel.created_at.desc())
            .all()
        )

    @classmethod
    def list_completed_for_user(cls, user_id: int) -> list[TrainedModel]:
        return (
            TrainedModel.query.filter_by(user_id=user_id, status="completed")
            .options(cls._with_dataset_chain())
            .order_by(TrainedModel.created_at.desc())
            .all()
        )

    @classmethod
    def get_for_user(cls, model_id: int, user_id: int) -> TrainedModel | None:
        return (
            TrainedModel.query.filter_by(id=model_id, user_id=user_id)
            .options(cls._with_dataset_chain())
            .first()
        )

    @classmethod
    def list_processed_runs_for_user(cls, user_id: int) -> list[ProcessedDataset]:
        return (
            ProcessedDataset.query.filter_by(user_id=user_id)
            .options(joinedload(ProcessedDataset.dataset))
            .order_by(ProcessedDataset.created_at.desc())
            .all()
        )

    @staticmethod
    def latest_evaluation(model: TrainedModel) -> ModelEvaluation | None:
        return model.latest_evaluation

    @staticmethod
    def model_choice_label(model: TrainedModel) -> str:
        dataset_name = model.processed_dataset.dataset.original_filename
        return f"{model.model_name} #{model.id} — {dataset_name}"
