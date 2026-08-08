"""Admin panel routes."""

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user

from app.core.http import pagination_to_dict, pdf_response
from app.core.logging_config import get_logger
from app.forms.admin_forms import AdminPredictionFilterForm, AdminSearchForm
from app.models.user import ROLE_ADMIN, ROLE_USER, User
from app.repositories.prediction_filters import PredictionFilters
from app.services.admin_report_pdf import generate_system_report_pdf
from app.services.admin_service import (
    AdminServiceError,
    admin_delete_dataset,
    admin_delete_prediction,
    admin_delete_trained_model,
    delete_user,
    get_admin_filter_models,
    get_system_statistics,
    get_user_activity_summary,
    query_all_datasets,
    query_all_models,
    query_all_predictions,
    query_users,
    set_user_active,
    set_user_role,
)
from app.utils.dashboard import dashboard_page_context
from app.utils.decorators import admin_required, login_active_required

admin_bp = Blueprint("admin", __name__)
logger = get_logger(__name__)


@admin_bp.route("/")
@login_active_required
@admin_required
def index():
    """Admin dashboard with system statistics."""
    stats = get_system_statistics()
    return render_template(
        "admin/index.html",
        stats=stats,
        **dashboard_page_context("admin.index", url_for, include_admin_nav=True),
    )


@admin_bp.route("/users")
@login_active_required
@admin_required
def users():
    """Manage user accounts and roles."""
    form = AdminSearchForm(request.args, meta={"csrf": False})
    pagination = query_users(
        search=form.search.data,
        page=request.args.get("page", 1, type=int),
    )
    return render_template(
        "admin/users.html",
        form=form,
        pagination=pagination_to_dict(pagination),
        users=pagination.items,
        role_user=ROLE_USER,
        role_admin=ROLE_ADMIN,
        **dashboard_page_context("admin.users", url_for, include_admin_nav=True),
    )


@admin_bp.route("/users/<int:user_id>/role", methods=["POST"])
@login_active_required
@admin_required
def users_set_role(user_id: int):
    """Promote or demote a user."""
    role = request.form.get("role", ROLE_USER)
    try:
        set_user_role(user_id, role, acting_user_id=current_user.id)
        logger.info("Admin %s set role=%s for user_id=%s", current_user.id, role, user_id)
        flash("User role updated.", "success")
    except AdminServiceError as exc:
        flash(str(exc), "danger")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/toggle-active", methods=["POST"])
@login_active_required
@admin_required
def users_toggle_active(user_id: int):
    """Activate or deactivate a user account."""
    target = User.query.get(user_id)
    if target is None:
        flash("User not found.", "warning")
        return redirect(url_for("admin.users"))

    try:
        set_user_active(user_id, not target.is_active, acting_user_id=current_user.id)
        flash("User status updated.", "success")
    except AdminServiceError as exc:
        flash(str(exc), "danger")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/delete", methods=["POST"])
@login_active_required
@admin_required
def users_delete(user_id: int):
    """Delete a user and owned resources."""
    try:
        delete_user(user_id, acting_user_id=current_user.id)
        logger.warning("Admin %s deleted user_id=%s", current_user.id, user_id)
        flash("User deleted.", "success")
    except AdminServiceError as exc:
        flash(str(exc), "danger")
    return redirect(url_for("admin.users"))


@admin_bp.route("/datasets")
@login_active_required
@admin_required
def datasets():
    """Manage all uploaded datasets."""
    form = AdminSearchForm(request.args, meta={"csrf": False})
    pagination = query_all_datasets(
        search=form.search.data,
        page=request.args.get("page", 1, type=int),
    )
    return render_template(
        "admin/datasets.html",
        form=form,
        pagination=pagination_to_dict(pagination),
        datasets=pagination.items,
        **dashboard_page_context("admin.datasets", url_for, include_admin_nav=True),
    )


@admin_bp.route("/datasets/<int:dataset_id>/delete", methods=["POST"])
@login_active_required
@admin_required
def datasets_delete(dataset_id: int):
    """Delete a dataset and dependent artefacts."""
    if admin_delete_dataset(dataset_id):
        flash("Dataset deleted.", "success")
    else:
        flash("Dataset not found.", "warning")
    return redirect(url_for("admin.datasets"))


@admin_bp.route("/predictions")
@login_active_required
@admin_required
def predictions():
    """View and manage all prediction logs."""
    models = get_admin_filter_models()
    form = AdminPredictionFilterForm(request.args, meta={"csrf": False})
    form.model_id.choices = [(0, "All models")] + [(model.id, model.model_name) for model in models]

    filters = PredictionFilters.from_request_args(request.args)
    model_id = filters.model_id or None

    pagination = query_all_predictions(
        search=filters.search,
        model_id=model_id,
        risk_level=filters.risk_level,
        page=request.args.get("page", 1, type=int),
    )
    return render_template(
        "admin/predictions.html",
        form=form,
        pagination=pagination_to_dict(pagination),
        records=pagination.items,
        **dashboard_page_context("admin.predictions", url_for, include_admin_nav=True),
    )


@admin_bp.route("/predictions/<int:record_id>/delete", methods=["POST"])
@login_active_required
@admin_required
def predictions_delete(record_id: int):
    """Delete a prediction log entry."""
    if admin_delete_prediction(record_id):
        flash("Prediction log deleted.", "success")
    else:
        flash("Prediction log not found.", "warning")
    return redirect(url_for("admin.predictions"))


@admin_bp.route("/models")
@login_active_required
@admin_required
def models():
    """Manage all trained models."""
    form = AdminSearchForm(request.args, meta={"csrf": False})
    pagination = query_all_models(
        search=form.search.data,
        page=request.args.get("page", 1, type=int),
    )
    return render_template(
        "admin/models.html",
        form=form,
        pagination=pagination_to_dict(pagination),
        models=pagination.items,
        **dashboard_page_context("admin.models", url_for, include_admin_nav=True),
    )


@admin_bp.route("/models/<int:model_id>/delete", methods=["POST"])
@login_active_required
@admin_required
def models_delete(model_id: int):
    """Delete a trained model."""
    if admin_delete_trained_model(model_id):
        flash("Model deleted.", "success")
    else:
        flash("Model not found.", "warning")
    return redirect(url_for("admin.models"))


@admin_bp.route("/reports")
@login_active_required
@admin_required
def reports():
    """Admin reports hub."""
    stats = get_system_statistics()
    user_activity = get_user_activity_summary()
    return render_template(
        "admin/reports.html",
        stats=stats,
        user_activity=user_activity,
        **dashboard_page_context("admin.reports", url_for, include_admin_nav=True),
    )


@admin_bp.route("/reports/system.pdf")
@login_active_required
@admin_required
def reports_system_pdf():
    """Download a system overview PDF report."""
    stats = get_system_statistics()
    user_activity = get_user_activity_summary()
    pdf_bytes = generate_system_report_pdf(
        stats,
        user_activity,
        current_app.config["APP_NAME"],
    )
    return pdf_response(pdf_bytes, "system-admin-report.pdf")
