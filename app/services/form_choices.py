"""Shared form choice builders for ML workflows."""

from __future__ import annotations

from app.repositories.model_repository import ModelRepository


def completed_model_form_choices(user_id: int) -> list[tuple[int, str]]:
    """Return WTForms choices for completed models owned by a user."""
    models = ModelRepository.list_completed_for_user(user_id)
    return [(0, "Select a trained model")] + [
        (model.id, ModelRepository.model_choice_label(model)) for model in models
    ]


def processed_dataset_form_choices(user_id: int) -> list[tuple[int, str]]:
    """Return WTForms choices for processed datasets owned by a user."""
    runs = ModelRepository.list_processed_runs_for_user(user_id)
    return [
        (run.id, f"#{run.id} — {run.dataset.original_filename} ({run.target_column})")
        for run in runs
    ]


def history_model_form_choices(user_id: int) -> list[tuple[int, str]]:
    """Return WTForms choices for models referenced in prediction history."""
    from app.repositories.prediction_repository import PredictionRepository

    models = PredictionRepository.models_for_filter(user_id=user_id)
    return [(0, "All models")] + [
        (model.id, f"{model.model_name} #{model.id}") for model in models
    ]
