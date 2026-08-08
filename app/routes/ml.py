"""Machine learning training routes."""

from pathlib import Path

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user
from sqlalchemy import desc

from app.core.http import pdf_response
from app.core.logging_config import get_logger
from app.forms.ml_forms import TrainModelForm
from app.ml.comparison import build_model_comparison
from app.ml.model_registry import MODEL_REGISTRY, get_model_choices
from app.ml.training import MLTrainingError, create_training_record, start_training_job
from app.models.model_evaluation import ModelEvaluation
from app.repositories.model_repository import ModelRepository
from app.services.comparison_pdf import generate_comparison_pdf
from app.services.form_choices import processed_dataset_form_choices
from app.utils.dashboard import dashboard_page_context
from app.utils.decorators import login_active_required

ml_bp = Blueprint("ml", __name__)
logger = get_logger(__name__)


def _populate_train_form(form: TrainModelForm) -> None:
    form.processed_dataset_id.choices = processed_dataset_form_choices(current_user.id)
    form.model_type.choices = get_model_choices()


def _launch_training(form: TrainModelForm, retrained_from_id: int | None = None):
    trained_model = create_training_record(
        user_id=current_user.id,
        processed_dataset_id=form.processed_dataset_id.data,
        model_type=form.model_type.data,
        parameters=form.extract_parameters(),
        use_smote=bool(form.use_smote_data.data),
        retrained_from_id=retrained_from_id,
    )
    start_training_job(
        current_app._get_current_object(),
        trained_model.id,
        Path(current_app.config["MODELS_FOLDER"]),
        use_smote=bool(form.use_smote_data.data),
    )
    return trained_model


@ml_bp.route("/")
@login_active_required
def index():
    """List trained models."""
    trained_models = ModelRepository.list_for_user(current_user.id)
    return render_template(
        "ml/index.html",
        trained_models=trained_models,
        model_registry=MODEL_REGISTRY,
        **dashboard_page_context("ml.index", url_for),
    )


@ml_bp.route("/train", methods=["GET", "POST"])
@login_active_required
def train():
    """Configure and start model training."""
    form = TrainModelForm()
    _populate_train_form(form)

    if not form.processed_dataset_id.choices:
        flash("You need a processed dataset before training a model.", "warning")
        return redirect(url_for("datasets.index"))

    if form.validate_on_submit():
        try:
            trained_model = _launch_training(form)
        except MLTrainingError as exc:
            flash(str(exc), "danger")
            return render_template(
                "ml/train.html",
                form=form,
                model_registry=MODEL_REGISTRY,
                **dashboard_page_context("ml.index", url_for),
            )

        flash(f"{trained_model.model_name} training started.", "success")
        return redirect(url_for("ml.detail", model_id=trained_model.id))

    return render_template(
        "ml/train.html",
        form=form,
        model_registry=MODEL_REGISTRY,
        **dashboard_page_context("ml.index", url_for),
    )


@ml_bp.route("/<int:model_id>")
@login_active_required
def detail(model_id: int):
    """Display model training status, parameters, and metrics."""
    trained_model = ModelRepository.get_for_user(model_id, current_user.id)
    if trained_model is None:
        flash("Trained model not found.", "warning")
        return redirect(url_for("ml.index"))

    return render_template(
        "ml/detail.html",
        trained_model=trained_model,
        evaluation=ModelRepository.latest_evaluation(trained_model),
        evaluation_history=trained_model.evaluations.order_by(
            desc(ModelEvaluation.evaluated_at)
        ).all(),
        model_registry=MODEL_REGISTRY,
        **dashboard_page_context("ml.index", url_for),
    )


@ml_bp.route("/<int:model_id>/status")
@login_active_required
def status(model_id: int):
    """Return JSON training status for polling."""
    trained_model = ModelRepository.get_for_user(model_id, current_user.id)
    if trained_model is None:
        return jsonify({"error": "Model not found"}), 404

    return jsonify(
        {
            "status": trained_model.status,
            "progress_percent": trained_model.progress_percent,
            "progress_log": trained_model.progress_log,
            "metrics": trained_model.metrics,
            "error_message": trained_model.error_message,
        }
    )


@ml_bp.route("/<int:model_id>/retrain", methods=["GET", "POST"])
@login_active_required
def retrain(model_id: int):
    """Retrain a model using previous configuration."""
    source_model = ModelRepository.get_for_user(model_id, current_user.id)
    if source_model is None:
        flash("Trained model not found.", "warning")
        return redirect(url_for("ml.index"))

    form = TrainModelForm()
    _populate_train_form(form)

    if request.method == "GET":
        form.processed_dataset_id.data = source_model.processed_dataset_id
        form.use_smote_data.data = source_model.training_data_source == "train_smote"
        form.apply_parameters(source_model.model_type, source_model.parameters)

    if form.validate_on_submit():
        try:
            trained_model = _launch_training(form, retrained_from_id=source_model.id)
        except MLTrainingError as exc:
            flash(str(exc), "danger")
            return render_template(
                "ml/train.html",
                form=form,
                model_registry=MODEL_REGISTRY,
                retraining_from=source_model,
                **dashboard_page_context("ml.index", url_for),
            )

        flash(f"Retraining started for {trained_model.model_name}.", "success")
        return redirect(url_for("ml.detail", model_id=trained_model.id))

    return render_template(
        "ml/train.html",
        form=form,
        model_registry=MODEL_REGISTRY,
        retraining_from=source_model,
        **dashboard_page_context("ml.index", url_for),
    )


@ml_bp.route("/compare")
@login_active_required
def compare():
    """Compare all trained models side by side."""
    comparison = build_model_comparison(current_user.id)
    return render_template(
        "ml/compare.html",
        comparison=comparison,
        **dashboard_page_context("ml.compare", url_for),
    )


@ml_bp.route("/compare/pdf")
@login_active_required
def compare_pdf():
    """Export the model comparison report as PDF."""
    comparison = build_model_comparison(current_user.id)
    if not comparison.get("has_models"):
        flash("No completed models are available to export.", "warning")
        return redirect(url_for("ml.compare"))

    try:
        pdf_bytes = generate_comparison_pdf(comparison, current_app.config["APP_NAME"])
    except Exception:
        logger.exception("Model comparison PDF export failed")
        flash("PDF export failed. Ensure kaleido is installed for chart rendering.", "danger")
        return redirect(url_for("ml.compare"))

    return pdf_response(pdf_bytes, "model_comparison_report.pdf")
