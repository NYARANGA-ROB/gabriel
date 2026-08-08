"""Dataset management routes."""

from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user

from app.core.security import resolve_allowed_path
from app.forms.dataset_forms import DatasetUploadForm
from app.forms.eda_forms import EDAForm
from app.forms.preprocessing_forms import PreprocessingForm
from app.models.dataset import Dataset
from app.models.processed_dataset import ProcessedDataset
from app.services.dataset_service import (
    DatasetServiceError,
    get_dataset_preview,
    get_user_dataset,
    save_dataset_upload,
)
from app.services.eda_service import EDAServiceError, build_eda_charts
from app.services.preprocessing_service import (
    PreprocessingConfig,
    PreprocessingServiceError,
    get_user_processed_dataset,
    run_preprocessing_pipeline,
)
from app.utils.dashboard import dashboard_page_context
from app.utils.decorators import login_active_required

datasets_bp = Blueprint("datasets", __name__)

DOWNLOAD_FILES = {
    "processed": ("processed_file_path", "processed_dataset.csv"),
    "train": ("train_file_path", "train_dataset.csv"),
    "test": ("test_file_path", "test_dataset.csv"),
    "train_smote": ("train_smote_file_path", "train_smote_dataset.csv"),
}


@datasets_bp.route("/")
@login_active_required
def index():
    """List datasets and provide upload form."""
    datasets = (
        Dataset.query.filter_by(user_id=current_user.id)
        .order_by(Dataset.uploaded_at.desc())
        .all()
    )
    form = DatasetUploadForm()
    return render_template(
        "datasets/index.html",
        datasets=datasets,
        form=form,
        **dashboard_page_context("datasets.index", url_for),
    )


@datasets_bp.route("/upload", methods=["POST"])
@login_active_required
def upload():
    """Handle CSV dataset upload."""
    form = DatasetUploadForm()
    if not form.validate_on_submit():
        for field_errors in form.errors.values():
            for error in field_errors:
                flash(error, "danger")
        return redirect(url_for("datasets.index"))

    try:
        dataset = save_dataset_upload(
            upload_file=form.dataset_file.data,
            user_id=current_user.id,
            upload_folder=Path(current_app.config["UPLOAD_FOLDER"]),
            allowed_extensions=current_app.config["ALLOWED_DATASET_EXTENSIONS"],
        )
    except DatasetServiceError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("datasets.index"))

    flash(
        f'Dataset "{dataset.original_filename}" uploaded successfully '
        f"({dataset.row_count:,} rows, {dataset.column_count} columns).",
        "success",
    )
    return redirect(url_for("datasets.detail", dataset_id=dataset.id))


@datasets_bp.route("/<int:dataset_id>")
@login_active_required
def detail(dataset_id: int):
    """Display dataset summary and paginated preview."""
    dataset = get_user_dataset(dataset_id, current_user.id)
    if dataset is None:
        flash("Dataset not found.", "warning")
        return redirect(url_for("datasets.index"))

    page = request.args.get("page", 1, type=int)
    per_page = current_app.config["DATASET_PREVIEW_PAGE_SIZE"]

    preview = get_dataset_preview(
        filepath=Path(dataset.file_path),
        page=page,
        per_page=per_page,
        total_rows=dataset.row_count,
    )
    processed_runs = (
        ProcessedDataset.query.filter_by(dataset_id=dataset.id, user_id=current_user.id)
        .order_by(ProcessedDataset.created_at.desc())
        .all()
    )

    return render_template(
        "datasets/detail.html",
        dataset=dataset,
        preview=preview,
        processed_runs=processed_runs,
        **dashboard_page_context("datasets.index", url_for),
    )


@datasets_bp.route("/<int:dataset_id>/eda", methods=["GET"])
@login_active_required
def eda(dataset_id: int):
    """Interactive exploratory data analysis for a dataset."""
    dataset = get_user_dataset(dataset_id, current_user.id)
    if dataset is None:
        flash("Dataset not found.", "warning")
        return redirect(url_for("datasets.index"))

    try:
        eda_result = build_eda_charts(
            filepath=Path(dataset.file_path),
            target_column=request.args.get("target_column"),
            feature_column=request.args.get("feature_column"),
            x_column=request.args.get("x_column"),
            y_column=request.args.get("y_column"),
        )
    except EDAServiceError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("datasets.detail", dataset_id=dataset.id))

    form = EDAForm()
    columns = eda_result["columns"]
    form.target_column.choices = [(column, column) for column in columns["all"]]
    feature_choices = columns["numeric"] or columns["all"]
    form.feature_column.choices = [(column, column) for column in feature_choices]
    scatter_choices = columns["numeric"] or columns["all"]
    form.x_column.choices = [(column, column) for column in scatter_choices]
    form.y_column.choices = [(column, column) for column in scatter_choices]

    selection = eda_result["selection"]
    form.target_column.data = selection["target_column"]
    form.feature_column.data = selection["feature_column"]
    form.x_column.data = selection["x_column"]
    form.y_column.data = selection["y_column"]

    return render_template(
        "datasets/eda.html",
        dataset=dataset,
        form=form,
        charts=eda_result["charts"],
        eda_meta=eda_result,
        **dashboard_page_context("datasets.index", url_for),
    )


@datasets_bp.route("/<int:dataset_id>/preprocess", methods=["GET", "POST"])
@login_active_required
def preprocess(dataset_id: int):
    """Configure and run preprocessing for a dataset."""
    dataset = get_user_dataset(dataset_id, current_user.id)
    if dataset is None:
        flash("Dataset not found.", "warning")
        return redirect(url_for("datasets.index"))

    form = PreprocessingForm()
    form.target_column.choices = [(column, column) for column in dataset.feature_names]

    if form.validate_on_submit():
        config = PreprocessingConfig(
            target_column=form.target_column.data,
            missing_strategy=form.missing_strategy.data,
            categorical_encoding=form.categorical_encoding.data,
            scaling_method=form.scaling_method.data,
            outlier_method=form.outlier_method.data,
            outlier_action=form.outlier_action.data,
            test_size=float(form.test_size.data),
            apply_smote=bool(form.apply_smote.data),
            random_state=int(form.random_state.data or 42),
        )

        try:
            processed = run_preprocessing_pipeline(
                dataset=dataset,
                config=config,
                output_folder=Path(current_app.config["PROCESSED_FOLDER"]),
            )
        except PreprocessingServiceError as exc:
            flash(str(exc), "danger")
            return render_template(
                "datasets/preprocess.html",
                dataset=dataset,
                form=form,
                **dashboard_page_context("datasets.index", url_for),
            )

        flash("Preprocessing pipeline completed successfully.", "success")
        return redirect(url_for("datasets.preprocess_report", dataset_id=dataset.id, processed_id=processed.id))

    return render_template(
        "datasets/preprocess.html",
        dataset=dataset,
        form=form,
        **dashboard_page_context("datasets.index", url_for),
    )


@datasets_bp.route("/<int:dataset_id>/preprocess/<int:processed_id>")
@login_active_required
def preprocess_report(dataset_id: int, processed_id: int):
    """Display preprocessing report."""
    dataset = get_user_dataset(dataset_id, current_user.id)
    processed = get_user_processed_dataset(processed_id, current_user.id)

    if dataset is None or processed is None or processed.dataset_id != dataset.id:
        flash("Preprocessing report not found.", "warning")
        return redirect(url_for("datasets.index"))

    return render_template(
        "datasets/preprocess_report.html",
        dataset=dataset,
        processed=processed,
        report=processed.report,
        **dashboard_page_context("datasets.index", url_for),
    )


@datasets_bp.route("/<int:dataset_id>/preprocess/<int:processed_id>/download")
@login_active_required
def download_processed(dataset_id: int, processed_id: int):
    """Download a processed dataset file."""
    dataset = get_user_dataset(dataset_id, current_user.id)
    processed = get_user_processed_dataset(processed_id, current_user.id)

    if dataset is None or processed is None or processed.dataset_id != dataset.id:
        flash("Processed dataset not found.", "warning")
        return redirect(url_for("datasets.index"))

    file_key = request.args.get("file", "processed")
    if file_key not in DOWNLOAD_FILES:
        abort(400)

    path_attr, download_name = DOWNLOAD_FILES[file_key]
    filepath = getattr(processed, path_attr)
    if not filepath:
        flash("The requested file is not available for this preprocessing run.", "warning")
        return redirect(
            url_for("datasets.preprocess_report", dataset_id=dataset.id, processed_id=processed.id)
        )

    allowed_roots = [
        Path(current_app.config["PROCESSED_FOLDER"]),
        Path(current_app.config["UPLOAD_FOLDER"]),
    ]
    safe_path = resolve_allowed_path(filepath, allowed_roots)
    if safe_path is None:
        abort(403)

    return send_file(
        safe_path,
        as_attachment=True,
        download_name=f"{dataset.original_filename.rsplit('.', 1)[0]}_{download_name}",
    )
