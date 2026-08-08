"""Shared model loading helpers for explainability modules."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from app.explainability.errors import ExplainabilityServiceError
from app.models.trained_model import TrainedModel
from app.services.preprocessing_inference import load_preprocessor, transform_patient_row
from app.utils.patient_fields import map_patient_input_to_row


def load_model_context(trained_model: TrainedModel) -> dict[str, Any]:
    """Load estimator, background data, and feature metadata."""
    if not trained_model.model_file_path:
        raise ExplainabilityServiceError("Model artefact is not available.")

    artefact = joblib.load(trained_model.model_file_path)
    processed = trained_model.processed_dataset
    test_df = pd.read_csv(processed.test_file_path)
    target_column = artefact.get("target_column", processed.target_column)
    feature_columns = artefact.get("feature_columns", [])

    if not feature_columns:
        raise ExplainabilityServiceError("Model feature columns are missing.")

    background = test_df[feature_columns].astype(float)
    if background.empty:
        raise ExplainabilityServiceError("Background data is empty.")

    return {
        "trained_model": trained_model,
        "artefact": artefact,
        "estimator": artefact["model"],
        "model_type": artefact.get("model_type", trained_model.model_type),
        "feature_columns": feature_columns,
        "target_column": target_column,
        "background": background,
        "processed": processed,
        "test_df": test_df,
    }


def resolve_patient_features(
    context: dict[str, Any],
    *,
    patient_data: dict[str, Any] | None = None,
    test_row_index: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Resolve a patient feature row from manual input or a test-set index."""
    feature_columns = context["feature_columns"]
    processed = context["processed"]
    dataset = processed.dataset

    if test_row_index is not None:
        background = context["background"]
        if test_row_index < 0 or test_row_index >= len(background):
            raise ExplainabilityServiceError("Selected test patient is out of range.")

        feature_frame = background.iloc[[test_row_index]].copy()
        row = feature_frame.iloc[0]
        metadata = {
            "source": "test_set",
            "test_row_index": test_row_index,
            "preview": {column: float(row[column]) for column in feature_columns[:5]},
        }
        label = f"Test patient #{test_row_index + 1}"
        return feature_frame, metadata, label

    if not patient_data:
        raise ExplainabilityServiceError("Patient data is required for manual explanations.")

    raw_row = map_patient_input_to_row(
        patient_data,
        dataset.feature_names,
        target_column=processed.target_column,
    )
    if not raw_row:
        raise ExplainabilityServiceError(
            "None of the entered patient fields match the selected model's dataset columns."
        )

    preprocessor = load_preprocessor(processed)
    feature_frame = transform_patient_row(
        preprocessor,
        raw_row,
        expected_columns=feature_columns,
    )
    metadata = {"source": "manual", "mapped_fields": raw_row}
    return feature_frame, metadata, "Manual patient entry"


def list_test_patients(context: dict[str, Any], limit: int = 50) -> list[dict[str, Any]]:
    """Return selectable patients from the model test set."""
    background = context["background"]
    feature_columns = context["feature_columns"]
    patients = []

    for index in range(min(len(background), limit)):
        row = background.iloc[index]
        preview_parts = [f"{column}={row[column]:.2f}" for column in feature_columns[:3]]
        patients.append(
            {
                "index": index,
                "label": f"Test patient #{index + 1}",
                "preview": ", ".join(preview_parts),
            }
        )

    return patients
