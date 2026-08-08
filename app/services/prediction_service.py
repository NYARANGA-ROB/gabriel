"""Patient readmission prediction service."""

from __future__ import annotations

import time
from typing import Any

import joblib
import numpy as np

from app.repositories.model_repository import ModelRepository
from app.services.preprocessing_inference import load_preprocessor, transform_patient_row
from app.utils.patient_fields import map_patient_input_to_row


class PredictionServiceError(Exception):
    """Raised when a patient prediction cannot be generated."""


RISK_THRESHOLDS = {
    "high": 0.70,
    "moderate": 0.40,
}

RECOMMENDATIONS = {
    "high": (
        "Schedule follow-up within 7 days. Consider transitional care management, "
        "medication reconciliation, and early post-discharge contact."
    ),
    "moderate": (
        "Arrange outpatient follow-up within 14 days. Review discharge instructions, "
        "monitor chronic conditions, and reinforce self-management education."
    ),
    "low": (
        "Standard discharge planning is appropriate. Reinforce routine primary care "
        "follow-up and patient education on warning signs."
    ),
}


def _risk_level(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["high"]:
        return "High"
    if probability >= RISK_THRESHOLDS["moderate"]:
        return "Moderate"
    return "Low"


def _risk_badge_class(risk_level: str) -> str:
    return {
        "High": "danger",
        "Moderate": "warning",
        "Low": "success",
    }.get(risk_level, "secondary")


def _positive_class_index(estimator, classes: np.ndarray) -> int:
    if len(classes) == 0:
        return 0
    if len(classes) == 1:
        return 0
    for index, value in enumerate(classes):
        if str(value) in {"1", "1.0", "True", "true", "yes", "Yes"}:
            return index
    return 1 if len(classes) > 1 else 0


def get_completed_models(user_id: int) -> list:
    """Return completed models available for prediction."""
    return ModelRepository.list_completed_for_user(user_id)


def predict_patient(
    user_id: int,
    model_id: int,
    patient_data: dict[str, Any],
) -> dict[str, Any]:
    """Generate a readmission prediction for a single patient."""
    trained_model = ModelRepository.get_for_user(model_id, user_id)
    if trained_model is None:
        raise PredictionServiceError("Selected model was not found.")
    if trained_model.status != "completed" or not trained_model.model_file_path:
        raise PredictionServiceError("Selected model is not ready for prediction.")

    processed = trained_model.processed_dataset
    dataset = processed.dataset
    available_columns = [column for column in dataset.feature_names if column != processed.target_column]

    raw_row = map_patient_input_to_row(
        patient_data,
        dataset.feature_names,
        target_column=processed.target_column,
    )
    if not raw_row:
        raise PredictionServiceError(
            "None of the entered patient fields match the selected model's dataset columns."
        )

    artefact = joblib.load(trained_model.model_file_path)
    feature_columns = artefact.get("feature_columns", [])
    preprocessor = load_preprocessor(processed)
    feature_frame = transform_patient_row(
        preprocessor,
        raw_row,
        expected_columns=feature_columns,
    )

    estimator = artefact["model"]
    predict_start = time.perf_counter()
    prediction = estimator.predict(feature_frame)[0]
    prediction_time_seconds = time.perf_counter() - predict_start

    probability = None
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(feature_frame)[0]
        classes = getattr(estimator, "classes_", np.array([0, 1]))
        positive_index = _positive_class_index(estimator, classes)
        probability = float(probabilities[positive_index])
    elif hasattr(estimator, "decision_function"):
        score = float(estimator.decision_function(feature_frame)[0])
        probability = float(1 / (1 + np.exp(-score)))

    if probability is None:
        probability = 1.0 if str(prediction) in {"1", "1.0", "True"} else 0.0

    risk = _risk_level(probability)
    readmission_label = "Likely readmission" if probability >= 0.5 else "Unlikely readmission"

    return {
        "model_id": trained_model.id,
        "model_name": trained_model.model_name,
        "model_label": f"{trained_model.model_name} #{trained_model.id}",
        "dataset_name": dataset.original_filename,
        "prediction": int(prediction) if str(prediction).isdigit() else str(prediction),
        "probability": probability,
        "probability_percent": round(probability * 100, 1),
        "risk_level": risk,
        "risk_badge_class": _risk_badge_class(risk),
        "recommendation": RECOMMENDATIONS[risk.lower()],
        "readmission_label": readmission_label,
        "prediction_time_ms": round(prediction_time_seconds * 1000, 2),
        "mapped_fields": raw_row,
        "used_feature_count": len(feature_columns),
    }
