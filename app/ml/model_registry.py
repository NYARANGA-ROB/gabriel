"""Machine learning model registry and factory."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "description": "Linear classifier suited to interpretable readmission risk modelling.",
        "estimator_class": LogisticRegression,
        "default_parameters": {
            "C": 0.5,
            "max_iter": 2000,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "random_state": 42,
        },
        "parameter_help": {
            "C": "Inverse regularisation strength.",
            "max_iter": "Maximum solver iterations.",
            "solver": "Optimisation algorithm.",
        },
    },
    "decision_tree": {
        "label": "Decision Tree",
        "description": "Non-linear tree-based model with straightforward decision rules.",
        "estimator_class": DecisionTreeClassifier,
        "default_parameters": {
            "max_depth": 12,
            "min_samples_split": 5,
            "criterion": "gini",
            "class_weight": "balanced",
            "random_state": 42,
        },
        "parameter_help": {
            "max_depth": "Maximum tree depth.",
            "min_samples_split": "Minimum samples required to split a node.",
            "criterion": "Split quality measure.",
        },
    },
    "random_forest": {
        "label": "Random Forest",
        "description": "Ensemble of decision trees for robust classification performance.",
        "estimator_class": RandomForestClassifier,
        "default_parameters": {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 5,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "parameter_help": {
            "n_estimators": "Number of trees in the forest.",
            "max_depth": "Maximum depth of each tree.",
            "min_samples_split": "Minimum samples required to split a node.",
        },
    },
    "xgboost": {
        "label": "XGBoost",
        "description": "Gradient boosted trees optimised for structured healthcare tabular data.",
        "estimator_class": "xgboost",
        "default_parameters": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "eval_metric": "logloss",
            "random_state": 42,
        },
        "parameter_help": {
            "n_estimators": "Number of boosting rounds.",
            "max_depth": "Maximum tree depth.",
            "learning_rate": "Shrinkage rate for each boosting step.",
        },
    },
    "neural_network": {
        "label": "Artificial Neural Network",
        "description": "Multi-layer perceptron for non-linear readmission pattern learning.",
        "estimator_class": MLPClassifier,
        "default_parameters": {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 500,
            "learning_rate_init": 0.001,
            "random_state": 42,
        },
        "parameter_help": {
            "hidden_layer_sizes": "Comma-separated hidden layer sizes, e.g. 64,32.",
            "activation": "Activation function.",
            "max_iter": "Maximum training epochs.",
            "learning_rate_init": "Initial learning rate.",
        },
    },
}


def get_model_choices() -> list[tuple[str, str]]:
    return [(key, meta["label"]) for key, meta in MODEL_REGISTRY.items()]


def get_model_label(model_type: str) -> str:
    return MODEL_REGISTRY.get(model_type, {}).get("label", model_type.replace("_", " ").title())


def _resolve_estimator_class(model_type: str):
    registry = MODEL_REGISTRY[model_type]
    estimator_class = registry["estimator_class"]
    if estimator_class == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ValueError(
                "XGBoost is not installed. Run: pip install xgboost"
            ) from exc
        return XGBClassifier
    return estimator_class


def build_estimator(model_type: str, parameters: dict[str, Any]):
    """Instantiate a scikit-learn compatible estimator."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")

    registry = MODEL_REGISTRY[model_type]
    params = {**registry["default_parameters"], **parameters}

    if model_type == "neural_network" and isinstance(params.get("hidden_layer_sizes"), str):
        layers = tuple(int(size.strip()) for size in params["hidden_layer_sizes"].split(",") if size.strip())
        params["hidden_layer_sizes"] = layers

    estimator_class = _resolve_estimator_class(model_type)
    return estimator_class(**params)


def serializable_parameters(model_type: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Return parameters in a JSON-safe format."""
    merged = {**MODEL_REGISTRY[model_type]["default_parameters"], **parameters}
    if model_type == "neural_network" and isinstance(merged.get("hidden_layer_sizes"), tuple):
        merged["hidden_layer_sizes"] = ",".join(str(size) for size in merged["hidden_layer_sizes"])
    return merged
