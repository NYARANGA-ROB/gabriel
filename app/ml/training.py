"""Model training pipeline."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask

from app.core.datetime_utils import utc_now
from app.core.logging_config import get_logger
from app.extensions import db
from app.ml.evaluation import evaluate_model, save_evaluation_record
from app.ml.model_registry import build_estimator, get_model_label, serializable_parameters
from app.models.processed_dataset import ProcessedDataset
from app.models.trained_model import TrainedModel
from app.repositories.model_repository import ModelRepository

logger = get_logger(__name__)


class MLTrainingError(Exception):
    """Raised when model training fails."""


def _append_progress(trained_model_id: int, step: str, message: str, percent: int) -> None:
    trained_model = db.session.get(TrainedModel, trained_model_id)
    if trained_model is None:
        return

    log = trained_model.progress_log
    log.append(
        {
            "step": step,
            "message": message,
            "percent": percent,
            "timestamp": utc_now().isoformat(),
        }
    )
    trained_model.progress_log = log
    trained_model.progress_percent = percent
    db.session.commit()


def _load_training_frames(
    processed: ProcessedDataset,
    use_smote: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    train_path = (
        processed.train_smote_file_path
        if use_smote and processed.train_smote_file_path
        else processed.train_file_path
    )
    if not train_path:
        raise MLTrainingError("Training data file is not available.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(processed.test_file_path)
    target_column = processed.target_column

    if target_column not in train_df.columns or target_column not in test_df.columns:
        raise MLTrainingError("Target column is missing from the processed dataset files.")

    return train_df, test_df, target_column


def train_model_sync(
    trained_model_id: int,
    models_folder: Path,
    use_smote: bool,
) -> None:
    """Train a model, evaluate it, and persist artefacts."""
    trained_model = db.session.get(TrainedModel, trained_model_id)
    if trained_model is None:
        raise MLTrainingError("Training record not found.")

    processed = trained_model.processed_dataset
    trained_model.status = "training"
    trained_model.progress_log = []
    _append_progress(trained_model.id, "load_data", "Loading processed training and test datasets.", 8)

    train_df, test_df, target_column = _load_training_frames(processed, use_smote)
    feature_columns = [column for column in train_df.columns if column != target_column]

    x_train = train_df[feature_columns]
    y_train = train_df[target_column]
    x_test = test_df[feature_columns]
    y_test = test_df[target_column]

    # Ensure valid column names for tree/boosting libraries.
    x_train = x_train.rename(columns=lambda name: str(name).replace("[", "(").replace("]", ")").replace("<", ""))
    x_test = x_test.rename(columns=lambda name: str(name).replace("[", "(").replace("]", ")").replace("<", ""))
    feature_columns = list(x_train.columns)

    _append_progress(trained_model.id, "prepare_features", "Preparing feature matrix and target vector.", 18)

    parameters = dict(trained_model.parameters)
    if trained_model.model_type == "xgboost":
        positive_count = int((y_train == 1).sum())
        negative_count = int((y_train == 0).sum())
        if positive_count > 0:
            parameters["scale_pos_weight"] = negative_count / positive_count

    estimator = build_estimator(trained_model.model_type, parameters)
    trained_model.parameters = serializable_parameters(trained_model.model_type, parameters)
    db.session.commit()

    _append_progress(
        trained_model.id,
        "initialize_model",
        f"Initialising {get_model_label(trained_model.model_type)}.",
        28,
    )

    _append_progress(trained_model.id, "training", "Training model on the selected dataset.", 45)
    train_start = time.perf_counter()
    estimator.fit(x_train, y_train)
    training_time_seconds = time.perf_counter() - train_start

    _append_progress(
        trained_model.id,
        "evaluation",
        "Evaluating model and generating performance charts.",
        72,
    )
    evaluation_result = evaluate_model(
        estimator,
        x_test,
        y_test,
        training_time_seconds,
        feature_names=feature_columns,
    )
    metrics = evaluation_result["metrics"]

    models_folder.mkdir(parents=True, exist_ok=True)
    filename = f"user{trained_model.user_id}_model{trained_model.id}_{trained_model.model_type}.joblib"
    model_path = models_folder / filename

    artefact = {
        "model": estimator,
        "model_type": trained_model.model_type,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "parameters": trained_model.parameters,
        "metrics": metrics,
        "evaluation": evaluation_result,
    }
    joblib.dump(artefact, model_path)

    trained_model = db.session.get(TrainedModel, trained_model_id)
    trained_model.model_file_path = str(model_path)
    trained_model.metrics = metrics
    save_evaluation_record(trained_model, evaluation_result)
    trained_model.status = "completed"
    trained_model.completed_at = utc_now()
    _append_progress(
        trained_model.id,
        "completed",
        "Training and evaluation completed successfully.",
        100,
    )


def start_training_job(app: Flask, trained_model_id: int, models_folder: Path, use_smote: bool) -> None:
    """Run model training in a background thread."""

    def _runner() -> None:
        with app.app_context():
            trained_model = db.session.get(TrainedModel, trained_model_id)
            if trained_model is None:
                return
            try:
                train_model_sync(trained_model_id, models_folder, use_smote)
            except Exception as exc:
                logger.exception("Training failed for model_id=%s", trained_model_id)
                trained_model = db.session.get(TrainedModel, trained_model_id)
                if trained_model is not None:
                    trained_model.status = "failed"
                    trained_model.error_message = str(exc)
                    _append_progress(
                        trained_model.id,
                        "failed",
                        f"Training failed: {exc}",
                        trained_model.progress_percent,
                    )

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()


def create_training_record(
    user_id: int,
    processed_dataset_id: int,
    model_type: str,
    parameters: dict[str, Any],
    use_smote: bool,
    retrained_from_id: int | None = None,
) -> TrainedModel:
    """Create a queued training record."""
    processed = ProcessedDataset.query.filter_by(id=processed_dataset_id, user_id=user_id).first()
    if processed is None:
        raise MLTrainingError("Processed dataset not found.")

    if use_smote and not processed.train_smote_file_path:
        raise MLTrainingError("SMOTE training data is not available for this processed dataset.")

    trained_model = TrainedModel(
        user_id=user_id,
        processed_dataset_id=processed_dataset_id,
        retrained_from_id=retrained_from_id,
        model_type=model_type,
        model_name=get_model_label(model_type),
        status="queued",
        training_data_source="train_smote" if use_smote else "train",
    )
    trained_model.parameters = serializable_parameters(model_type, parameters)
    trained_model.progress_log = []

    db.session.add(trained_model)
    db.session.commit()
    return trained_model


def get_user_trained_model(model_id: int, user_id: int) -> TrainedModel | None:
    """Return a trained model owned by the given user."""
    return ModelRepository.get_for_user(model_id, user_id)
