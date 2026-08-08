"""SHAP explainer factory for supported model types."""

from __future__ import annotations

import shap
import numpy as np
import pandas as pd


TREE_MODEL_TYPES = {"decision_tree", "random_forest", "xgboost"}
KERNEL_MODEL_TYPES = {"neural_network"}


def create_shap_explainer(model_type: str, estimator, background: pd.DataFrame):
    """Return a SHAP explainer appropriate for the trained estimator."""
    if model_type in TREE_MODEL_TYPES:
        return shap.TreeExplainer(estimator)

    if model_type == "logistic_regression":
        return shap.LinearExplainer(estimator, background)

    if model_type in KERNEL_MODEL_TYPES:
        sample_size = min(100, len(background))
        background_sample = shap.sample(background, sample_size, random_state=42)
        return shap.KernelExplainer(
            estimator.predict_proba,
            background_sample,
            link="logit",
        )

    return shap.Explainer(estimator.predict_proba, background)


def positive_shap_values(explainer, shap_values):
    """Select SHAP values for the positive readmission class."""
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]

    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    values = np.asarray(values)

    if values.ndim == 3:
        class_index = 1 if values.shape[2] > 1 else 0
        return values[:, :, class_index]

    return values


def expected_base_value(explainer, shap_values=None) -> float:
    """Return the explainer base value for the positive class."""
    if shap_values is not None and hasattr(shap_values, "base_values"):
        base = shap_values.base_values
        if isinstance(base, np.ndarray):
            if base.ndim == 0:
                return float(base)
            if base.ndim == 1:
                return float(base[0]) if len(base) == 1 else float(base[1] if len(base) > 1 else base[0])
            return float(base[0, 1] if base.shape[1] > 1 else base[0, 0])
        return float(base)

    expected = getattr(explainer, "expected_value", 0.0)
    if isinstance(expected, (list, np.ndarray)):
        if len(expected) > 1:
            return float(expected[1])
        return float(expected[0])
    return float(expected)
