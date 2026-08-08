"""Admin panel data access and management operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import func, or_
from sqlalchemy.orm import joinedload

from app.extensions import db
from app.models.dataset import Dataset
from app.models.model_evaluation import ModelEvaluation
from app.models.prediction_record import PredictionRecord
from app.models.processed_dataset import ProcessedDataset
from app.models.trained_model import TrainedModel
from app.models.user import ROLE_ADMIN, ROLE_USER, User


class AdminServiceError(Exception):
    """Raised when admin operations fail."""


def get_system_statistics() -> dict[str, Any]:
    """Return aggregate counts and activity metrics for the admin dashboard."""
    user_counts = db.session.query(
        func.count(User.id),
        func.count(User.id).filter(User.is_active.is_(True)),
        func.count(User.id).filter(User.role == ROLE_ADMIN),
    ).one()

    prediction_counts = db.session.query(
        func.count(PredictionRecord.id),
        func.count(PredictionRecord.id).filter(PredictionRecord.risk_level == "High"),
        func.count(PredictionRecord.id).filter(PredictionRecord.risk_level == "Low"),
        func.count(PredictionRecord.id).filter(PredictionRecord.risk_level == "Moderate"),
    ).one()

    model_counts = db.session.query(
        func.count(TrainedModel.id),
        func.count(TrainedModel.id).filter(TrainedModel.status == "completed"),
    ).one()

    latest_prediction = (
        PredictionRecord.query.order_by(PredictionRecord.created_at.desc()).first()
    )
    latest_user = User.query.order_by(User.created_at.desc()).first()

    total_users, active_users, admin_users = user_counts
    total_predictions, high_risk, low_risk, moderate_risk = prediction_counts
    total_models, completed_models = model_counts

    return {
        "users": {
            "total": total_users,
            "active": active_users,
            "admins": admin_users,
            "inactive": total_users - active_users,
        },
        "datasets": {
            "uploaded": Dataset.query.count(),
            "processed": ProcessedDataset.query.count(),
        },
        "models": {
            "total": total_models,
            "completed": completed_models,
            "in_progress": total_models - completed_models,
        },
        "predictions": {
            "total": total_predictions,
            "high_risk": high_risk,
            "low_risk": low_risk,
            "moderate_risk": moderate_risk,
        },
        "latest_prediction_at": latest_prediction.created_at if latest_prediction else None,
        "latest_user_email": latest_user.email if latest_user else None,
    }


def query_users(*, search: str | None = None, page: int = 1, per_page: int = 20):
    """Return a paginated list of all users."""
    query = User.query.order_by(User.created_at.desc())
    if search:
        term = f"%{search.strip()}%"
        query = query.filter(
            or_(
                User.email.ilike(term),
                User.first_name.ilike(term),
                User.last_name.ilike(term),
            )
        )
    return query.paginate(page=max(page, 1), per_page=per_page, error_out=False)


def query_all_datasets(*, search: str | None = None, page: int = 1, per_page: int = 20):
    """Return a paginated list of all uploaded datasets."""
    query = (
        Dataset.query.options(joinedload(Dataset.user))
        .order_by(Dataset.uploaded_at.desc())
    )
    if search:
        term = f"%{search.strip()}%"
        query = query.filter(Dataset.original_filename.ilike(term))
    return query.paginate(page=max(page, 1), per_page=per_page, error_out=False)


def query_all_predictions(
    *,
    search: str | None = None,
    model_id: int | None = None,
    risk_level: str | None = None,
    page: int = 1,
    per_page: int = 20,
):
    """Return a paginated list of all prediction records."""
    from app.repositories.prediction_filters import PredictionFilters
    from app.repositories.prediction_repository import PredictionRepository

    filters = PredictionFilters(search=search, model_id=model_id, risk_level=risk_level)
    return PredictionRepository.paginate(
        page=page,
        per_page=per_page,
        filters=filters,
        eager_load_user=True,
    )


def query_all_models(*, search: str | None = None, page: int = 1, per_page: int = 20):
    """Return a paginated list of all trained models."""
    query = (
        TrainedModel.query.options(joinedload(TrainedModel.user))
        .order_by(TrainedModel.created_at.desc())
    )
    if search:
        term = f"%{search.strip()}%"
        query = query.filter(
            or_(
                TrainedModel.model_name.ilike(term),
                TrainedModel.model_type.ilike(term),
                TrainedModel.status.ilike(term),
            )
        )
    return query.paginate(page=max(page, 1), per_page=per_page, error_out=False)


def get_admin_filter_models() -> list[TrainedModel]:
    """Return models referenced in prediction logs."""
    from app.repositories.prediction_repository import PredictionRepository

    return PredictionRepository.models_for_filter()


def _unlink_file(path: str | None) -> None:
    if not path:
        return
    Path(path).unlink(missing_ok=True)


def admin_delete_trained_model(model_id: int, *, commit: bool = True) -> bool:
    """Delete a trained model and related artefacts."""
    model = db.session.get(TrainedModel, model_id)
    if model is None:
        return False

    PredictionRecord.query.filter_by(trained_model_id=model_id).delete()
    ModelEvaluation.query.filter_by(trained_model_id=model_id).delete()
    _unlink_file(model.model_file_path)
    db.session.delete(model)
    if commit:
        db.session.commit()
    return True


def admin_delete_processed_dataset(processed_id: int, *, commit: bool = True) -> bool:
    """Delete a processed dataset and dependent models."""
    processed = db.session.get(ProcessedDataset, processed_id)
    if processed is None:
        return False

    for model in list(processed.trained_models):
        admin_delete_trained_model(model.id, commit=False)

    for path in (
        processed.processed_file_path,
        processed.train_file_path,
        processed.test_file_path,
        processed.train_smote_file_path,
    ):
        _unlink_file(path)

    preprocessor_path = Path(processed.processed_file_path).parent / "_preprocessor.joblib"
    _unlink_file(str(preprocessor_path))

    db.session.delete(processed)
    if commit:
        db.session.commit()
    return True


def admin_delete_dataset(dataset_id: int, *, commit: bool = True) -> bool:
    """Delete a dataset and all dependent processed runs."""
    dataset = db.session.get(Dataset, dataset_id)
    if dataset is None:
        return False

    for processed in list(dataset.processed_runs):
        admin_delete_processed_dataset(processed.id, commit=False)

    _unlink_file(dataset.file_path)
    db.session.delete(dataset)
    if commit:
        db.session.commit()
    return True


def admin_delete_prediction(record_id: int) -> bool:
    """Delete any prediction record."""
    record = db.session.get(PredictionRecord, record_id)
    if record is None:
        return False
    db.session.delete(record)
    db.session.commit()
    return True


def set_user_role(user_id: int, role: str, *, acting_user_id: int) -> None:
    """Update a user's role."""
    if role not in (ROLE_USER, ROLE_ADMIN):
        raise AdminServiceError("Invalid role.")

    user = db.session.get(User, user_id)
    if user is None:
        raise AdminServiceError("User not found.")

    if user_id == acting_user_id and role != ROLE_ADMIN:
        raise AdminServiceError("You cannot remove your own admin access.")

    if user.role == ROLE_ADMIN and role != ROLE_ADMIN:
        admin_count = User.query.filter_by(role=ROLE_ADMIN).count()
        if admin_count <= 1:
            raise AdminServiceError("At least one admin account must remain.")

    user.role = role
    db.session.commit()


def set_user_active(user_id: int, is_active: bool, *, acting_user_id: int) -> None:
    """Activate or deactivate a user account."""
    user = db.session.get(User, user_id)
    if user is None:
        raise AdminServiceError("User not found.")

    if user_id == acting_user_id and not is_active:
        raise AdminServiceError("You cannot deactivate your own account.")

    if user.role == ROLE_ADMIN and not is_active:
        active_admins = User.query.filter_by(role=ROLE_ADMIN, is_active=True).count()
        if active_admins <= 1:
            raise AdminServiceError("At least one active admin account must remain.")

    user.is_active = is_active
    db.session.commit()


def delete_user(user_id: int, *, acting_user_id: int) -> None:
    """Delete a user and their owned resources."""
    user = db.session.get(User, user_id)
    if user is None:
        raise AdminServiceError("User not found.")

    if user_id == acting_user_id:
        raise AdminServiceError("You cannot delete your own account.")

    if user.role == ROLE_ADMIN:
        admin_count = User.query.filter_by(role=ROLE_ADMIN).count()
        if admin_count <= 1:
            raise AdminServiceError("Cannot delete the only admin account.")

    for dataset in list(user.datasets):
        admin_delete_dataset(dataset.id, commit=False)

    for processed in ProcessedDataset.query.filter_by(user_id=user_id).all():
        admin_delete_processed_dataset(processed.id, commit=False)

    for model in TrainedModel.query.filter_by(user_id=user_id).all():
        admin_delete_trained_model(model.id, commit=False)

    PredictionRecord.query.filter_by(user_id=user_id).delete()
    db.session.delete(user)
    db.session.commit()


def get_user_activity_summary() -> list[dict[str, Any]]:
    """Return per-user activity counts for reports."""
    rows = (
        db.session.query(
            User.id,
            User.email,
            User.first_name,
            User.last_name,
            User.role,
            User.is_active,
            func.count(func.distinct(Dataset.id)).label("dataset_count"),
            func.count(func.distinct(TrainedModel.id)).label("model_count"),
            func.count(func.distinct(PredictionRecord.id)).label("prediction_count"),
        )
        .outerjoin(Dataset, Dataset.user_id == User.id)
        .outerjoin(TrainedModel, TrainedModel.user_id == User.id)
        .outerjoin(PredictionRecord, PredictionRecord.user_id == User.id)
        .group_by(User.id)
        .order_by(User.created_at.desc())
        .all()
    )

    return [
        {
            "id": row.id,
            "email": row.email,
            "full_name": f"{row.first_name} {row.last_name}".strip(),
            "role": row.role,
            "is_active": row.is_active,
            "dataset_count": row.dataset_count,
            "model_count": row.model_count,
            "prediction_count": row.prediction_count,
        }
        for row in rows
    ]
