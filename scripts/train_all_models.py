"""Import UCI diabetes data, preprocess, and train every registered model."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_diabetes_dataset import (
    DEFAULT_TARGET_ROWS,
    MIN_DATASET_ROWS,
    prepare_diabetes_csv,
)

LARGE_DATASET_SMOTE_THRESHOLD = 250_000


def _resolve_user(db, User, email: str | None):
    if email:
        user = User.query.filter_by(email=email).first()
        if user is None:
            raise SystemExit(f"No user found with email: {email}")
        return user
    user = User.query.order_by(User.id.asc()).first()
    if user is None:
        raise SystemExit("No user found. Register an account first.")
    return user


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UCI diabetes data and train all models.")
    parser.add_argument(
        "--email",
        help="Train for this user email (default: first registered user).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row cap for the prepared CSV.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use balanced 50/50 sampling (legacy mode).",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use the smaller balanced sample instead of the full UCI dataset.",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=DEFAULT_TARGET_ROWS,
        help=f"Target dataset size (default: {DEFAULT_TARGET_ROWS:,}).",
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Keep only real UCI rows (~102k) without expanding to --target-rows.",
    )
    args = parser.parse_args()

    from app import create_app
    from app.extensions import db
    from app.ml.model_registry import MODEL_REGISTRY, serializable_parameters
    from app.ml.training import create_training_record, train_model_sync
    from app.models.dataset import Dataset
    from app.models.trained_model import TrainedModel
    from app.models.user import User
    from app.services.dataset_service import analyze_csv
    from app.services.preprocessing_service import PreprocessingConfig, run_preprocessing_pipeline

    app = create_app()

    with app.app_context():
        user = _resolve_user(db, User, args.email)
        print(f"Training for user id={user.id} ({user.email})")

        use_full = not args.legacy
        target_rows = None if args.no_expand else args.target_rows
        if args.legacy:
            target_rows = None
        if args.sample_size is not None and args.sample_size < MIN_DATASET_ROWS:
            raise SystemExit(f"--sample-size must be at least {MIN_DATASET_ROWS:,}.")
        if not args.no_expand and not args.legacy and args.target_rows < MIN_DATASET_ROWS:
            raise SystemExit(f"--target-rows must be at least {MIN_DATASET_ROWS:,}.")
        prepare_kwargs = {
            "use_full": use_full,
            "balanced": args.balanced,
            "target_rows": target_rows,
        }
        if args.sample_size is not None:
            prepare_kwargs["sample_size"] = args.sample_size
        csv_path = prepare_diabetes_csv(**prepare_kwargs)
        dataset_name = csv_path.name

        upload_folder = Path(app.config["UPLOAD_FOLDER"])
        upload_folder.mkdir(parents=True, exist_ok=True)

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
            dataset.missing_values = analysis["missing_values"]
            dataset.duplicate_rows = analysis["duplicate_rows"]
            dataset.feature_names = analysis["feature_names"]
            dataset.missing_by_column = analysis["missing_by_column"]
            db.session.commit()

        print(f"Using dataset id={dataset.id} ({dataset.row_count:,} rows, {dataset.column_count} columns)")

        use_smote = dataset.row_count < LARGE_DATASET_SMOTE_THRESHOLD
        if not use_smote:
            print("SMOTE disabled for large dataset; using class-balanced model weights instead.")

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
        print(
            f"Preprocessed id={processed.id}: train={processed.train_rows}, "
            f"test={processed.test_rows}, features={processed.feature_count}"
        )

        models_folder = Path(app.config["MODELS_FOLDER"])
        model_types = list(MODEL_REGISTRY.keys())
        results: list[tuple[str, str, float, float]] = []

        for model_type in model_types:
            print(f"\nTraining {model_type} ...")
            trained = create_training_record(
                user_id=user.id,
                processed_dataset_id=processed.id,
                model_type=model_type,
                parameters=serializable_parameters(model_type, {}),
                use_smote=use_smote,
            )
            start = time.perf_counter()
            try:
                train_model_sync(trained.id, models_folder, use_smote=use_smote)
                db.session.refresh(trained)
                elapsed = time.perf_counter() - start
                status = trained.status
                metrics = trained.metrics or {}
                accuracy = metrics.get("accuracy", 0)
                f1 = metrics.get("f1_score", 0)
                results.append((model_type, status, accuracy, f1))
                print(
                    f"  -> {status} in {elapsed:.1f}s "
                    f"(Accuracy={accuracy:.3f}, F1={f1:.3f})"
                )
            except Exception as exc:
                print(f"  -> failed: {exc}")
                results.append((model_type, "failed", 0.0, 0.0))

        print("\n=== SUMMARY ===")
        for model_type, status, accuracy, f1 in results:
            print(f"  {model_type:22} {status:10} Acc={accuracy:.3f}  F1={f1:.3f}")

        completed = TrainedModel.query.filter_by(
            user_id=user.id, status="completed", processed_dataset_id=processed.id
        ).count()
        print(f"\n{completed}/{len(model_types)} models completed on {dataset_name}.")


if __name__ == "__main__":
    main()
