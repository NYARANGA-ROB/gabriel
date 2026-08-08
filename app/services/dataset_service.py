"""Dataset analysis and file handling."""

import math
from pathlib import Path

import pandas as pd
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from app.extensions import db
from app.models.dataset import Dataset


class DatasetServiceError(Exception):
    """Raised when dataset processing fails."""


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    """Check whether the file has an allowed extension."""
    return Path(filename).suffix.lower() in allowed_extensions


def analyze_csv(filepath: Path) -> dict:
    """Analyse a CSV file and return summary statistics."""
    try:
        dataframe = pd.read_csv(filepath)
    except Exception as exc:
        raise DatasetServiceError("Unable to read CSV file. Check the file format.") from exc

    if dataframe.empty:
        raise DatasetServiceError("The uploaded CSV file contains no data rows.")

    missing_by_column = {
        column: int(dataframe[column].isnull().sum()) for column in dataframe.columns
    }

    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "missing_values": int(dataframe.isnull().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "feature_names": [str(column) for column in dataframe.columns],
        "missing_by_column": missing_by_column,
    }


def save_dataset_upload(
    upload_file: FileStorage,
    user_id: int,
    upload_folder: Path,
    allowed_extensions: set[str],
) -> Dataset:
    """Validate, store, and persist metadata for an uploaded dataset."""
    if not upload_file or not upload_file.filename:
        raise DatasetServiceError("No file was provided.")

    original_filename = secure_filename(upload_file.filename)
    if not original_filename:
        raise DatasetServiceError("Invalid filename.")

    if not allowed_file(original_filename, allowed_extensions):
        raise DatasetServiceError("Only CSV files are supported.")

    upload_folder.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
    stored_filename = f"user{user_id}_{timestamp}_{original_filename}"
    filepath = upload_folder / stored_filename

    upload_file.save(filepath)

    try:
        analysis = analyze_csv(filepath)
    except DatasetServiceError:
        filepath.unlink(missing_ok=True)
        raise
    except Exception as exc:
        filepath.unlink(missing_ok=True)
        raise DatasetServiceError("Failed to process the uploaded dataset.") from exc

    dataset = Dataset(
        user_id=user_id,
        original_filename=original_filename,
        stored_filename=stored_filename,
        file_path=str(filepath),
        file_size=filepath.stat().st_size,
        row_count=analysis["row_count"],
        column_count=analysis["column_count"],
        missing_values=analysis["missing_values"],
        duplicate_rows=analysis["duplicate_rows"],
    )
    dataset.feature_names = analysis["feature_names"]
    dataset.missing_by_column = analysis["missing_by_column"]

    db.session.add(dataset)
    db.session.commit()
    return dataset


def get_dataset_preview(filepath: Path, page: int, per_page: int, total_rows: int) -> dict:
    """Return a paginated preview of dataset rows."""
    page = max(page, 1)
    per_page = max(per_page, 1)
    total_pages = max(math.ceil(total_rows / per_page), 1)
    page = min(page, total_pages)

    skip = (page - 1) * per_page
    if skip > 0:
        dataframe = pd.read_csv(filepath, skiprows=range(1, skip + 1), nrows=per_page)
    else:
        dataframe = pd.read_csv(filepath, nrows=per_page)

    rows = dataframe.fillna("").astype(str).values.tolist()

    return {
        "columns": [str(column) for column in dataframe.columns],
        "rows": rows,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "total_rows": total_rows,
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "start_row": skip + 1 if total_rows else 0,
        "end_row": min(skip + len(rows), total_rows),
    }


def get_user_dataset(dataset_id: int, user_id: int) -> Dataset | None:
    """Return a dataset owned by the given user."""
    return Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()
