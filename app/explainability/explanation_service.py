"""Orchestrate SHAP, LIME, and combined explainability outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.explainability.errors import ExplainabilityServiceError
from app.explainability.lime_service import build_lime_local_explanations, build_shap_lime_comparison
from app.explainability.model_context import list_test_patients, load_model_context
from app.explainability.shap_service import (
    build_global_explanations,
    build_local_explanations,
    list_explainable_models,
)
from app.ml.training import get_user_trained_model


def build_full_explanations(
    user_id: int,
    model_id: int,
    patient_data: dict[str, Any] | None = None,
    *,
    test_row_index: int | None = None,
    dependence_feature: str | None = None,
) -> dict[str, Any]:
    """Generate SHAP, LIME, and comparison explanations."""
    has_patient = test_row_index is not None or bool(patient_data)

    if has_patient:
        global_result = build_global_explanations(
            user_id,
            model_id,
            dependence_feature=dependence_feature,
        )
        shap_local = build_local_explanations(
            user_id,
            model_id,
            patient_data or {},
            test_row_index=test_row_index,
        )
        lime_local = build_lime_local_explanations(
            user_id,
            model_id,
            patient_data,
            test_row_index=test_row_index,
        )

        trained_model = get_user_trained_model(model_id, user_id)
        context = load_model_context(trained_model)
        feature_names = context["feature_columns"]

        comparison = build_shap_lime_comparison(
            np.array(shap_local["shap_row"]),
            np.array(lime_local["aligned_weights"]),
            feature_names,
            np.array(lime_local["feature_values"]),
        )

        return {
            **global_result,
            "has_local": True,
            "patient_label": lime_local["patient_label"],
            "local": {
                "mapped_fields": shap_local.get("mapped_fields", shap_local.get("patient_meta", {})),
                "patient_meta": lime_local["patient_meta"],
                "base_value": shap_local["base_value"],
                "output_value": shap_local["output_value"],
                "top_contributors": shap_local["top_contributors"],
                "charts": shap_local["charts"],
            },
            "lime": lime_local,
            "comparison": comparison,
            "charts": {
                **global_result["charts"],
                **{f"local_{key}": value for key, value in shap_local["charts"].items()},
                **{f"lime_{key}": value for key, value in lime_local["charts"].items()},
                "shap_lime_comparison": comparison["chart"],
            },
        }

    return build_global_explanations(user_id, model_id, dependence_feature=dependence_feature)


def get_test_patients_for_model(user_id: int, model_id: int) -> list[dict[str, Any]]:
    """List patients from the model test set."""
    trained_model = get_user_trained_model(model_id, user_id)
    if trained_model is None:
        raise ExplainabilityServiceError("Selected model was not found.")
    context = load_model_context(trained_model)
    return list_test_patients(context)
