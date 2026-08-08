"""Fit and apply preprocessing transformers for single-patient inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from app.models.processed_dataset import ProcessedDataset
from app.services.preprocessing_service import (
    PreprocessingConfig,
    PreprocessingServiceError,
    _handle_missing_values,
    _is_categorical,
)


def _encode_categorical_fit(
    features: pd.DataFrame,
    categorical_columns: list[str],
    encoding: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    encoded = features.copy()
    metadata: dict[str, Any] = {
        "method": encoding,
        "categorical_columns": categorical_columns,
    }

    if encoding == "one_hot":
        encoded = pd.get_dummies(encoded, columns=categorical_columns, drop_first=False)
        metadata["dummy_columns"] = list(encoded.columns)
        metadata["category_values"] = {
            column: sorted(features[column].astype(str).unique().tolist())
            for column in categorical_columns
        }
    else:
        label_encoders: dict[str, LabelEncoder] = {}
        label_maps: dict[str, dict[str, int]] = {}
        for column in categorical_columns:
            encoder = LabelEncoder()
            encoded[column] = encoder.fit_transform(encoded[column].astype(str))
            label_encoders[column] = encoder
            label_maps[column] = {
                str(label): int(index) for index, label in enumerate(encoder.classes_)
            }
        metadata["label_encoders"] = label_encoders
        metadata["label_maps"] = label_maps

    return encoded, metadata


def _scale_features_fit(
    features: pd.DataFrame,
    numeric_columns: list[str],
    method: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scaled = features.copy()
    columns_to_scale = [column for column in numeric_columns if column in scaled.columns]

    if method == "none" or not columns_to_scale:
        return scaled, {"method": method, "scaled_columns": [], "scaler": None}

    scaler: StandardScaler | MinMaxScaler
    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    scaled[columns_to_scale] = scaler.fit_transform(scaled[columns_to_scale])
    return scaled, {
        "method": method,
        "scaled_columns": columns_to_scale,
        "scaler": scaler,
    }


def fit_preprocessor(dataframe: pd.DataFrame, config: PreprocessingConfig) -> dict[str, Any]:
    """Fit preprocessing transformers on the raw dataset."""
    if config.target_column not in dataframe.columns:
        raise PreprocessingServiceError("Target column was not found in the dataset.")

    cleaned, missing_report = _handle_missing_values(dataframe.copy(), config.missing_strategy)
    if config.target_column not in cleaned.columns:
        raise PreprocessingServiceError("Target column was removed during missing value handling.")
    if cleaned.empty:
        raise PreprocessingServiceError("No rows remain after missing value handling.")

    feature_columns = [column for column in cleaned.columns if column != config.target_column]
    features = cleaned[feature_columns].copy()
    numeric_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(cleaned[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    missing_fills: dict[str, Any] = {}
    for column, details in missing_report.get("columns_affected", {}).items():
        missing_fills[column] = details.get("fill_value")

    return fit_preprocessor_from_prepared_features(
        features,
        categorical_columns,
        numeric_columns,
        config,
        missing_fills,
    )


def fit_preprocessor_from_prepared_features(
    features: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    config: PreprocessingConfig,
    missing_fills: dict[str, Any],
) -> dict[str, Any]:
    """Fit transformers on cleaned feature columns prior to train/test split."""
    prepared = features.copy()
    for column in categorical_columns:
        prepared[column] = prepared[column].astype(str)

    encoded_features, encoding_metadata = _encode_categorical_fit(
        prepared,
        categorical_columns,
        config.categorical_encoding,
    )
    numeric_after_encoding = [
        column for column in numeric_columns if column in encoded_features.columns
    ]
    scaled_features, scaling_metadata = _scale_features_fit(
        encoded_features,
        numeric_after_encoding,
        config.scaling_method,
    )

    return {
        "config": {
            "target_column": config.target_column,
            "missing_strategy": config.missing_strategy,
            "categorical_encoding": config.categorical_encoding,
            "scaling_method": config.scaling_method,
        },
        "raw_feature_columns": list(features.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_columns": [str(column) for column in scaled_features.columns],
        "missing_fills": missing_fills,
        "encoding": encoding_metadata,
        "scaling": scaling_metadata,
    }


def save_preprocessor(preprocessor: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def _preprocessor_path(processed: ProcessedDataset) -> Path:
    filename = processed.report.get("output_files", {}).get("preprocessor")
    folder = Path(processed.processed_file_path).parent
    if filename:
        return folder / filename
    stem = Path(processed.processed_file_path).stem.replace("_processed", "")
    return folder / f"{stem}_preprocessor.joblib"


def load_preprocessor(processed: ProcessedDataset) -> dict[str, Any]:
    """Load a saved preprocessor or rebuild it from the original dataset."""
    path = _preprocessor_path(processed)
    if path.exists():
        return joblib.load(path)

    dataset = processed.dataset
    dataframe = pd.read_csv(dataset.file_path)
    config = PreprocessingConfig(**processed.config)
    preprocessor = fit_preprocessor(dataframe, config)
    save_preprocessor(preprocessor, path)
    return preprocessor


def _apply_missing_fills(row: dict[str, object], missing_fills: dict[str, Any]) -> dict[str, object]:
    filled = dict(row)
    for column, fill_value in missing_fills.items():
        if column not in filled or filled[column] in (None, ""):
            if fill_value is not None:
                filled[column] = fill_value
    return filled


def _encode_row(
    row_df: pd.DataFrame,
    encoding: dict[str, Any],
) -> pd.DataFrame:
    method = encoding.get("method", "one_hot")
    categorical_columns = encoding.get("categorical_columns", [])

    for column in categorical_columns:
        if column in row_df.columns:
            row_df[column] = row_df[column].astype(str)

    if method == "one_hot":
        encoded = pd.get_dummies(row_df, columns=categorical_columns, drop_first=False)
        for column in encoding.get("dummy_columns", []):
            if column not in encoded.columns:
                encoded[column] = 0
        return encoded[encoding.get("dummy_columns", encoded.columns)]

    encoded = row_df.copy()
    label_maps = encoding.get("label_maps", {})
    for column in categorical_columns:
        if column not in encoded.columns:
            continue
        value = str(encoded.at[encoded.index[0], column])
        mapping = label_maps.get(column, {})
        encoded.at[encoded.index[0], column] = mapping.get(value, next(iter(mapping.values()), 0))
    return encoded


def _scale_row(row_df: pd.DataFrame, scaling: dict[str, Any]) -> pd.DataFrame:
    scaler = scaling.get("scaler")
    columns = scaling.get("scaled_columns", [])
    if scaler is None or not columns:
        return row_df

    scaled = row_df.copy()
    scaled[columns] = scaler.transform(scaled[columns])
    return scaled


def transform_patient_row(
    preprocessor: dict[str, Any],
    raw_row: dict[str, object],
    *,
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Transform a raw patient row into model-ready features."""
    raw_feature_columns = preprocessor["raw_feature_columns"]
    row = _apply_missing_fills(raw_row, preprocessor.get("missing_fills", {}))

    single_row: dict[str, object] = {}
    for column in raw_feature_columns:
        if column in row:
            single_row[column] = row[column]
        elif column in preprocessor.get("numeric_columns", []):
            single_row[column] = 0
        elif column in preprocessor.get("categorical_columns", []):
            category_values = preprocessor.get("encoding", {}).get("category_values", {}).get(column)
            if category_values:
                single_row[column] = category_values[0]
            else:
                label_maps = preprocessor.get("encoding", {}).get("label_maps", {})
                mapping = label_maps.get(column, {})
                single_row[column] = next(iter(mapping.keys()), "Unknown")

    row_df = pd.DataFrame([single_row])
    encoded = _encode_row(row_df, preprocessor.get("encoding", {}))
    scaled = _scale_row(encoded, preprocessor.get("scaling", {}))

    feature_columns = expected_columns or preprocessor.get("feature_columns", list(scaled.columns))
    aligned = scaled.reindex(columns=feature_columns, fill_value=0)
    return aligned.astype(float)
