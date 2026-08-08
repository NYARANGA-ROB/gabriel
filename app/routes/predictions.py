"""Prediction and analysis routes."""

import io

from flask import Blueprint, current_app, flash, redirect, render_template, request, send_file, session, url_for
from flask_login import current_user

from app.core.http import pagination_to_dict, pdf_response
from app.core.logging_config import get_logger
from app.explainability.errors import ExplainabilityServiceError
from app.explainability.explanation_service import build_local_explanations
from app.forms.history_forms import PredictionHistoryFilterForm
from app.forms.prediction_forms import PatientPredictionForm
from app.repositories.prediction_filters import PredictionFilters
from app.services.form_choices import completed_model_form_choices, history_model_form_choices
from app.services.prediction_history_service import (
    PredictionHistoryError,
    delete_prediction_record,
    export_history_csv,
    export_history_pdf,
    query_prediction_history,
    save_prediction_record,
)
from app.services.prediction_service import PredictionServiceError, get_completed_models, predict_patient
from app.utils.dashboard import dashboard_page_context
from app.utils.decorators import login_active_required

predictions_bp = Blueprint("predictions", __name__)
logger = get_logger(__name__)


@predictions_bp.route("/", methods=["GET", "POST"])
@login_active_required
def index():
    """Patient prediction interface."""
    form = PatientPredictionForm()
    form.trained_model_id.choices = completed_model_form_choices(current_user.id)
    models = get_completed_models(current_user.id)
    result = None
    explanation = None

    if form.validate_on_submit():
        if form.trained_model_id.data == 0:
            flash("Please select a trained model.", "warning")
        else:
            patient_data = form.patient_payload()
            try:
                result = predict_patient(
                    current_user.id,
                    form.trained_model_id.data,
                    patient_data,
                )
                save_prediction_record(
                    current_user.id,
                    form.patient_id.data,
                    form.trained_model_id.data,
                    result,
                )
                session["last_patient_payload"] = patient_data
                session["last_prediction_model_id"] = form.trained_model_id.data
                try:
                    explanation = build_local_explanations(
                        current_user.id,
                        form.trained_model_id.data,
                        patient_data,
                    )
                except ExplainabilityServiceError:
                    explanation = None
            except PredictionServiceError as exc:
                flash(str(exc), "danger")
            except Exception:
                logger.exception("Prediction failed for user_id=%s", current_user.id)
                flash(
                    "Prediction failed. Check that the selected model and patient data are valid.",
                    "danger",
                )

    return render_template(
        "predictions/index.html",
        form=form,
        models=models,
        result=result,
        explanation=explanation,
        **dashboard_page_context("predictions.index", url_for),
    )


@predictions_bp.route("/history")
@login_active_required
def history():
    """Paginated prediction history with search and filters."""
    form = PredictionHistoryFilterForm()
    form.model_id.choices = history_model_form_choices(current_user.id)

    filters = PredictionFilters.from_request_args(request.args)
    form.search.data = filters.search
    form.model_id.data = filters.model_id or 0
    form.risk_level.data = filters.risk_level or ""

    pagination = query_prediction_history(
        current_user.id,
        page=request.args.get("page", 1, type=int),
        per_page=current_app.config["PREDICTION_HISTORY_PAGE_SIZE"],
        **filters.as_export_kwargs(),
    )

    return render_template(
        "predictions/history.html",
        form=form,
        pagination=pagination_to_dict(pagination),
        records=pagination.items,
        filters=filters.as_export_kwargs(),
        export_query=request.args.to_dict(),
        **dashboard_page_context("predictions.history", url_for),
    )


@predictions_bp.route("/history/<int:record_id>/delete", methods=["POST"])
@login_active_required
def history_delete(record_id: int):
    """Delete a prediction history record."""
    if delete_prediction_record(record_id, current_user.id):
        flash("Prediction record deleted.", "success")
    else:
        flash("Prediction record not found.", "warning")

    filters = PredictionFilters.from_request_args(request.args)
    return redirect(
        url_for(
            "predictions.history",
            **filters.as_export_kwargs(),
            page=request.args.get("page", 1),
        )
    )


@predictions_bp.route("/history/export.csv")
@login_active_required
def history_export_csv():
    """Export filtered prediction history as CSV."""
    filters = PredictionFilters.from_request_args(request.args)
    try:
        csv_bytes = export_history_csv(current_user.id, **filters.as_export_kwargs())
    except PredictionHistoryError as exc:
        flash(str(exc), "warning")
        return redirect(url_for("predictions.history", **filters.as_export_kwargs()))

    return send_file(
        io.BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_history.csv",
    )


@predictions_bp.route("/history/export.pdf")
@login_active_required
def history_export_pdf():
    """Export filtered prediction history as PDF."""
    filters = PredictionFilters.from_request_args(request.args)
    try:
        pdf_bytes = export_history_pdf(
            current_user.id,
            current_app.config["APP_NAME"],
            **filters.as_export_kwargs(),
        )
    except PredictionHistoryError as exc:
        flash(str(exc), "warning")
        return redirect(url_for("predictions.history", **filters.as_export_kwargs()))

    return pdf_response(pdf_bytes, "prediction_history.pdf")
