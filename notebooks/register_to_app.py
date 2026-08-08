"""Register notebook outputs (dataset, preprocessing, models) in the Flask app database."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from ml_utils import DEFAULT_CSV, PROJECT_ROOT, USER_EMAIL, ensure_project_paths, flask_app_context

if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))


def register_pipeline(
    *,
    csv_path: Path,
    user_email: str,
    processed_id_hint: int | None = None,
) -> None:
    """Import prepared CSV, preprocessing run, and trained joblib files into SQLite."""
    from app.extensions import db
    from app.ml.model_registry import MODEL_REGISTRY, get_model_label
    from app.ml.training import create_training_record, train_model_sync
    from app.models.dataset import Dataset
    from app.models.processed_dataset import ProcessedDataset
    from app.models.trained_model import TrainedModel
    from app.models.user import User
    from app.services.dataset_service import analyze_csv
    from app.services.preprocessing_service import PreprocessingConfig, run_preprocessing_pipeline

    ensure_project_paths()
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    with flask_app_context() as app:
        user = User.query.filter_by(email=user_email).first()
        if user is None:
            raise SystemExit(f"No user with email {user_email}. Register in the web app first.")

        upload_folder = Path(app.config["UPLOAD_FOLDER"])
        dataset_name = csv_path.name

        dataset = Dataset.query.filter_by(
            user_id=user.id,
            original_filename=dataset_name,
        ).first()

        if dataset is None:
            stored_name = f"user{user.id}_{int(time.time())}_{dataset_name}"
            destination = upload_folder / stored_name
            shutil.copy2(csv_path, destination)
            analysis = analyze_csv(destination)
            dataset = Dataset(
                user_id=user.id,
                original_filename=dataset_name,
                stored_filename=stored_name,
                file_path=str(destination),
                file_size=destination.stat().st_size,
                row_count=analysis["row_count"],
                column_count=analysis["column_count"],
                missing_values=analysis["missing_values"],
                duplicate_rows=analysis["duplicate_rows"],
            )
            dataset.feature_names = analysis["feature_names"]
            dataset.missing_by_column = analysis["missing_by_column"]
            db.session.add(dataset)
            db.session.commit()
        else:
            shutil.copy2(csv_path, Path(dataset.file_path))
            analysis = analyze_csv(Path(dataset.file_path))
            dataset.row_count = analysis["row_count"]
            dataset.column_count = analysis["column_count"]
            dataset.feature_names = analysis["feature_names"]
            db.session.commit()

        use_smote = dataset.row_count < 250_000
        processed = None
        if processed_id_hint:
            processed = ProcessedDataset.query.filter_by(
                id=processed_id_hint, user_id=user.id
            ).first()

        if processed is None:
            config = PreprocessingConfig(
                target_column="readmitted",
                missing_strategy="mean",
                categorical_encoding="label",
                scaling_method="standard",
                outlier_method="iqr",
                outlier_action="report_only",
                test_size=0.2,
                apply_smote=use_smote,
                random_state=42,
            )
            processed = run_preprocessing_pipeline(
                dataset,
                config,
                Path(app.config["PROCESSED_FOLDER"]),
            )

        models_folder = Path(app.config["MODELS_FOLDER"])
        for model_type in MODEL_REGISTRY:
            existing = TrainedModel.query.filter_by(
                user_id=user.id,
                processed_dataset_id=processed.id,
                model_type=model_type,
                status="completed",
            ).first()
            if existing and existing.model_file_path and Path(existing.model_file_path).exists():
                print(f"Skip {model_type}: already registered as #{existing.id}")
                continue

            artefact_path = models_folder / f"notebook_{model_type}.joblib"
            if artefact_path.exists():
                trained = create_training_record(
                    user_id=user.id,
                    processed_dataset_id=processed.id,
                    model_type=model_type,
                    parameters={},
                    use_smote=use_smote,
                )
                dest = models_folder / f"user{user.id}_model{trained.id}_{model_type}.joblib"
                shutil.copy2(artefact_path, dest)
                train_model_sync(trained.id, models_folder, use_smote=use_smote)
                print(f"Registered {get_model_label(model_type)} as model #{trained.id}")
            else:
                print(f"No notebook artefact for {model_type} at {artefact_path}")

        print(f"Dataset id={dataset.id}, processed id={processed.id}, user={user.email}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync notebook ML outputs to the web app.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--email", default=USER_EMAIL)
    args = parser.parse_args()
    register_pipeline(csv_path=args.csv, user_email=args.email)


if __name__ == "__main__":
    main()
