"""Explainable AI routes."""

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from flask_login import current_user

from app.core.http import pdf_response
from app.core.logging_config import get_logger
from app.explainability.errors import ExplainabilityServiceError
from app.explainability.explanation_service import (
    build_full_explanations,
    get_test_patients_for_model,
    list_explainable_models,
)
from app.forms.prediction_forms import PatientPredictionForm
from app.services.form_choices import completed_model_form_choices
from app.services.explainability_pdf import generate_explainability_pdf
from app.utils.dashboard import dashboard_page_context
from app.utils.decorators import login_active_required

explainability_bp = Blueprint("explainability", __name__)
logger = get_logger(__name__)


def _populate_form(form: PatientPredictionForm) -> None:
    form.trained_model_id.choices = completed_model_form_choices(current_user.id)


def _selected_model_id(form: PatientPredictionForm) -> int | None:
    model_id = form.trained_model_id.data or request.args.get("model_id", type=int)
    return model_id if model_id and model_id != 0 else None


@explainability_bp.route("/", methods=["GET", "POST"])
@login_active_required
def index():
    """SHAP and LIME explainability dashboard."""
    form = PatientPredictionForm()
    _populate_form(form)
    models = list_explainable_models(current_user.id)
    explanation = None
    dependence_feature = request.args.get("dependence_feature") or request.form.get("dependence_feature")
    patient_source = request.form.get("patient_source", "manual")
    test_patients: list[dict] = []
    model_id = _selected_model_id(form)

    if model_id:
        try:
            test_patients = get_test_patients_for_model(current_user.id, model_id)
            form.trained_model_id.data = model_id
        except ExplainabilityServiceError:
            test_patients = []

    if request.method == "GET" and request.args.get("model_id"):
        try:
            get_model_id = int(request.args.get("model_id"))
            patient_data = session.get("last_patient_payload")
            model_match = session.get("last_prediction_model_id")
            kwargs = {
                "dependence_feature": dependence_feature,
            }
            if patient_data and model_match == get_model_id:
                kwargs["patient_data"] = patient_data
            explanation = build_full_explanations(current_user.id, get_model_id, **kwargs)
            form.trained_model_id.data = get_model_id
        except (ExplainabilityServiceError, ValueError) as exc:
            flash(str(exc), "danger")

    if request.method == "POST" and request.form.get("submit_explanations"):
        if not model_id:
            flash("Please select a trained model.", "warning")
        elif patient_source == "test":
            test_row_index = request.form.get("test_patient_index", type=int)
            if test_row_index is None:
                flash("Please select a test-set patient.", "warning")
            else:
                try:
                    explanation = build_full_explanations(
                        current_user.id,
                        model_id,
                        test_row_index=test_row_index,
                        dependence_feature=dependence_feature,
                    )
                except ExplainabilityServiceError as exc:
                    flash(str(exc), "danger")
                except Exception:
                    flash("Explanation failed. Ensure SHAP and LIME are installed.", "danger")
        elif form.validate_on_submit():
            try:
                patient_data = form.patient_payload()
                explanation = build_full_explanations(
                    current_user.id,
                    model_id,
                    patient_data,
                    dependence_feature=dependence_feature,
                )
                session["last_patient_payload"] = patient_data
                session["last_prediction_model_id"] = model_id
            except ExplainabilityServiceError as exc:
                flash(str(exc), "danger")
            except Exception:
                logger.exception("Explainability failed for user_id=%s", current_user.id)
                flash("Explanation failed. Ensure SHAP and LIME are installed.", "danger")
        else:
            flash("Please complete the required patient fields.", "warning")

    feature_choices = []
    if explanation and explanation.get("feature_names"):
        feature_choices = explanation["feature_names"]

    return render_template(
        "explainability/index.html",
        form=form,
        models=models,
        explanation=explanation,
        feature_choices=feature_choices,
        dependence_feature=dependence_feature,
        test_patients=test_patients,
        patient_source=patient_source,
        **dashboard_page_context("explainability.index", url_for),
    )


@explainability_bp.route("/pdf")
@login_active_required
def pdf():
    """Export SHAP and LIME explanations as PDF."""
    model_id = request.args.get("model_id", type=int)
    if not model_id:
        flash("A model must be selected before exporting.", "warning")
        return redirect(url_for("explainability.index"))

    dependence_feature = request.args.get("dependence_feature")
    patient_data = session.get("last_patient_payload")
    model_match = session.get("last_prediction_model_id")
    test_row_index = request.args.get("test_row_index", type=int)

    try:
        kwargs = {"dependence_feature": dependence_feature}
        if test_row_index is not None:
            kwargs["test_row_index"] = test_row_index
        elif patient_data and model_match == model_id:
            kwargs["patient_data"] = patient_data
        explanation = build_full_explanations(current_user.id, model_id, **kwargs)
        pdf_bytes = generate_explainability_pdf(explanation, current_app.config["APP_NAME"])
    except ExplainabilityServiceError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("explainability.index"))
    except Exception:
        logger.exception("Explainability PDF export failed")
        flash("PDF export failed. Ensure kaleido is installed for chart rendering.", "danger")
        return redirect(url_for("explainability.index"))

    return pdf_response(pdf_bytes, "explainability_report.pdf")
