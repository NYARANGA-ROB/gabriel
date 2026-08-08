"""SHAP explanation service for global and local model interpretability."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.explainability.explainer_factory import (
    create_shap_explainer,
    expected_base_value,
    positive_shap_values,
)
from app.explainability.errors import ExplainabilityServiceError
from app.explainability.model_context import load_model_context, resolve_patient_features
from app.ml.training import get_user_trained_model
from app.models.trained_model import TrainedModel
from app.services.plotting.base import figure_to_dict
from app.services.plotting.explainability_charts import (
    create_shap_dependence_chart,
    create_shap_force_chart,
    create_shap_importance_chart,
    create_shap_summary_chart,
    create_shap_waterfall_chart,
)
from app.services.prediction_service import get_completed_models


def _compute_shap_matrix(context: dict[str, Any]) -> tuple[np.ndarray, float]:
    estimator = context["estimator"]
    background = context["background"]
    model_type = context["model_type"]

    explainer = create_shap_explainer(model_type, estimator, background)
    try:
        raw_values = explainer.shap_values(background)
    except Exception as exc:
        raise ExplainabilityServiceError(f"SHAP computation failed: {exc}") from exc

    shap_matrix = np.asarray(positive_shap_values(explainer, raw_values), dtype=float)
    if shap_matrix.ndim == 1:
        shap_matrix = shap_matrix.reshape(1, -1)
    if shap_matrix.ndim != 2:
        raise ExplainabilityServiceError("Unexpected SHAP value shape for global explanation.")

    base_value = expected_base_value(explainer, raw_values)
    return shap_matrix, base_value


def _top_feature_index(shap_matrix: np.ndarray, feature_names: list[str]) -> int:
    importance = np.abs(shap_matrix).mean(axis=0)
    return int(np.argmax(importance))


def _build_global_charts(
    shap_matrix: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    dependence_feature: str | None = None,
) -> tuple[dict[str, dict], str]:
    top_index = _top_feature_index(shap_matrix, feature_names)
    dependence_index = top_index
    if dependence_feature and dependence_feature in feature_names:
        dependence_index = feature_names.index(dependence_feature)

    interaction_index = None
    if len(feature_names) > 1:
        importance = np.abs(shap_matrix).mean(axis=0)
        ordered = np.argsort(importance)[::-1]
        for index in ordered:
            if index != dependence_index:
                interaction_index = int(index)
                break

    dependence_chart = create_shap_dependence_chart(
        feature_matrix[:, dependence_index],
        shap_matrix[:, dependence_index],
        feature_names[dependence_index],
        (
            feature_matrix[:, interaction_index]
            if interaction_index is not None
            else None
        ),
        feature_names[interaction_index] if interaction_index is not None else None,
    )

    return {
        "summary": figure_to_dict(create_shap_summary_chart(shap_matrix, feature_names, feature_matrix)),
        "feature_importance": figure_to_dict(
            create_shap_importance_chart(shap_matrix, feature_names)
        ),
        "dependence": figure_to_dict(dependence_chart),
    }, feature_names[dependence_index]


def build_global_explanations(
    user_id: int,
    model_id: int,
    *,
    dependence_feature: str | None = None,
) -> dict[str, Any]:
    """Generate global SHAP explanations for a trained model."""
    trained_model = get_user_trained_model(model_id, user_id)
    if trained_model is None:
        raise ExplainabilityServiceError("Selected model was not found.")
    if trained_model.status != "completed":
        raise ExplainabilityServiceError("Selected model is not ready for explainability analysis.")

    context = load_model_context(trained_model)
    shap_matrix, base_value = _compute_shap_matrix(context)
    feature_names = context["feature_columns"]
    feature_matrix = context["background"].values

    charts, dependence_name = _build_global_charts(
        shap_matrix,
        feature_matrix,
        feature_names,
        dependence_feature=dependence_feature,
    )

    return {
        "model_id": trained_model.id,
        "model_label": f"{trained_model.model_name} #{trained_model.id}",
        "dataset_name": trained_model.processed_dataset.dataset.original_filename,
        "feature_names": feature_names,
        "dependence_feature": dependence_name,
        "sample_count": int(len(context["background"])),
        "base_value": base_value,
        "charts": charts,
        "has_local": False,
    }


def build_local_explanations(
    user_id: int,
    model_id: int,
    patient_data: dict[str, Any],
    *,
    test_row_index: int | None = None,
) -> dict[str, Any]:
    """Generate local SHAP explanations for an individual patient."""
    trained_model = get_user_trained_model(model_id, user_id)
    if trained_model is None:
        raise ExplainabilityServiceError("Selected model was not found.")
    if trained_model.status != "completed":
        raise ExplainabilityServiceError("Selected model is not ready for explainability analysis.")

    context = load_model_context(trained_model)
    processed = context["processed"]
    dataset = processed.dataset

    feature_frame, patient_meta, patient_label = resolve_patient_features(
        context,
        patient_data=patient_data if test_row_index is None else None,
        test_row_index=test_row_index,
    )

    explainer = create_shap_explainer(
        context["model_type"],
        context["estimator"],
        context["background"],
    )

    try:
        raw_values = explainer.shap_values(feature_frame)
    except Exception as exc:
        raise ExplainabilityServiceError(f"Local SHAP computation failed: {exc}") from exc

    shap_row = np.asarray(positive_shap_values(explainer, raw_values), dtype=float)
    if shap_row.ndim > 1:
        shap_row = shap_row.reshape(-1)
    base_value = expected_base_value(explainer, raw_values)
    feature_names = context["feature_columns"]
    feature_values = feature_frame.values.reshape(-1)

    charts = {
        "waterfall": figure_to_dict(
            create_shap_waterfall_chart(shap_row, feature_names, base_value, feature_values)
        ),
        "force": figure_to_dict(
            create_shap_force_chart(shap_row, feature_names, base_value, feature_values)
        ),
    }

    top_contributors = []
    order = np.argsort(np.abs(shap_row))[::-1][:5]
    for index in order:
        top_contributors.append(
            {
                "feature": feature_names[index],
                "value": float(feature_values[index]),
                "shap": float(shap_row[index]),
                "direction": "increases risk" if shap_row[index] >= 0 else "decreases risk",
            }
        )

    return {
        "model_id": trained_model.id,
        "model_label": f"{trained_model.model_name} #{trained_model.id}",
        "dataset_name": dataset.original_filename,
        "patient_label": patient_label,
        "patient_meta": patient_meta,
        "mapped_fields": patient_meta.get("mapped_fields", {}),
        "base_value": base_value,
        "output_value": float(base_value + np.sum(shap_row)),
        "shap_row": shap_row.tolist(),
        "top_contributors": top_contributors,
        "charts": charts,
        "has_local": True,
    }



def list_explainable_models(user_id: int) -> list[TrainedModel]:
    """Return models that can be explained."""
    return get_completed_models(user_id)
