"""LIME local explanation service."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from app.explainability.model_context import load_model_context, resolve_patient_features
from app.explainability.errors import ExplainabilityServiceError
from app.ml.training import get_user_trained_model
from app.services.plotting.base import figure_to_dict
from app.services.plotting.explainability_charts import (
    create_lime_interactive_chart,
    create_lime_weights_chart,
    create_shap_lime_comparison_chart,
)


def _positive_class_index(estimator) -> int:
    classes = getattr(estimator, "classes_", np.array([0, 1]))
    if len(classes) <= 1:
        return 0
    for index, value in enumerate(classes):
        if str(value) in {"1", "1.0", "True", "true"}:
            return int(index)
    return 1


def _align_lime_weights(
    weights_list: list[tuple[str, float]],
    feature_names: list[str],
) -> np.ndarray:
    aligned = np.zeros(len(feature_names), dtype=float)
    for description, weight in weights_list:
        for index, name in enumerate(feature_names):
            if description == name or description.startswith(f"{name} "):
                aligned[index] = weight
                break
    return aligned


def build_lime_local_explanations(
    user_id: int,
    model_id: int,
    patient_data: dict[str, Any] | None = None,
    *,
    test_row_index: int | None = None,
) -> dict[str, Any]:
    """Generate LIME local explanations for a selected patient."""
    trained_model = get_user_trained_model(model_id, user_id)
    if trained_model is None:
        raise ExplainabilityServiceError("Selected model was not found.")
    if trained_model.status != "completed":
        raise ExplainabilityServiceError("Selected model is not ready for explainability analysis.")

    context = load_model_context(trained_model)
    feature_frame, patient_meta, patient_label = resolve_patient_features(
        context,
        patient_data=patient_data,
        test_row_index=test_row_index,
    )

    estimator = context["estimator"]
    background = context["background"]
    feature_names = context["feature_columns"]
    positive_label = _positive_class_index(estimator)

    explainer = LimeTabularExplainer(
        background.values,
        feature_names=feature_names,
        class_names=["No readmission", "Readmission"],
        mode="classification",
        discretize_continuous=False,
        random_state=42,
    )

    def classifier_fn(features):
        matrix = features
        if not isinstance(matrix, pd.DataFrame):
            matrix = pd.DataFrame(matrix, columns=feature_names)
        return estimator.predict_proba(matrix)

    try:
        lime_explanation = explainer.explain_instance(
            feature_frame.values[0],
            classifier_fn,
            num_features=min(len(feature_names), 10),
            top_labels=1,
        )
    except Exception as exc:
        raise ExplainabilityServiceError(f"LIME explanation failed: {exc}") from exc

    weights_list = lime_explanation.as_list(label=positive_label)
    aligned_weights = _align_lime_weights(weights_list, feature_names)
    feature_values = feature_frame.values.reshape(-1)

    positive_factors = [
        {"feature": feature, "weight": float(weight)}
        for feature, weight in weights_list
        if weight > 0
    ]
    negative_factors = [
        {"feature": feature, "weight": float(weight)}
        for feature, weight in weights_list
        if weight < 0
    ]

    positive_factors.sort(key=lambda item: item["weight"], reverse=True)
    negative_factors.sort(key=lambda item: item["weight"])

    charts = {
        "weights": figure_to_dict(
            create_lime_weights_chart(weights_list, patient_label)
        ),
        "interactive": figure_to_dict(
            create_lime_interactive_chart(
                weights_list,
                feature_values,
                feature_names,
                patient_label,
            )
        ),
    }

    return {
        "patient_label": patient_label,
        "patient_meta": patient_meta,
        "positive_factors": positive_factors,
        "negative_factors": negative_factors,
        "weights_list": [{"feature": f, "weight": float(w)} for f, w in weights_list],
        "aligned_weights": aligned_weights.tolist(),
        "feature_values": feature_values.tolist(),
        "prediction_proba": {
            "no_readmission": float(lime_explanation.predict_proba[0]),
            "readmission": float(lime_explanation.predict_proba[1])
            if len(lime_explanation.predict_proba) > 1
            else float(lime_explanation.predict_proba[0]),
        },
        "charts": charts,
    }


def build_shap_lime_comparison(
    shap_row: np.ndarray,
    lime_weights: np.ndarray,
    feature_names: list[str],
    feature_values: np.ndarray,
) -> dict[str, Any]:
    """Build side-by-side SHAP vs LIME comparison payload."""
    shap_row = np.asarray(shap_row, dtype=float).reshape(-1)
    lime_weights = np.asarray(lime_weights, dtype=float).reshape(-1)

    chart = figure_to_dict(
        create_shap_lime_comparison_chart(
            feature_names,
            shap_row,
            lime_weights,
            feature_values,
        )
    )

    combined = np.abs(shap_row) + np.abs(lime_weights)
    order = np.argsort(combined)[::-1][:8]
    rows = []
    for index in order:
        rows.append(
            {
                "feature": feature_names[index],
                "value": float(feature_values[index]),
                "shap": float(shap_row[index]),
                "lime": float(lime_weights[index]),
                "agreement": "aligned"
                if np.sign(shap_row[index]) == np.sign(lime_weights[index]) or lime_weights[index] == 0
                else "divergent",
            }
        )

    return {"rows": rows, "chart": chart}
