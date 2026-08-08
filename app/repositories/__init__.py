"""Data-access layer with optimized queries."""

from app.repositories.model_repository import ModelRepository
from app.repositories.prediction_repository import PredictionRepository

__all__ = ["ModelRepository", "PredictionRepository"]
