"""Dataset preprocessing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from app.extensions import db
from app.models.dataset import Dataset
from app.models.processed_dataset import ProcessedDataset


class PreprocessingServiceError(Exception):
    """Raised when preprocessing fails."""


@dataclass
class PreprocessingConfig:
    target_column: str
    missing_strategy: str
    categorical_encoding: str
    scaling_method: str
    outlier_method: str
    outlier_action: str
    test_size: float
    apply_smote: bool
    random_state: int


def _is_categorical(series: pd.Series) -> bool:
    return series.dtype == "object" or str(series.dtype) == "category" or series.dtype == bool


def _handle_missing_values(dataframe: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, dict]:
    report: dict[str, Any] = {
        "strategy": strategy,
        "columns_affected": {},
        "rows_dropped": 0,
        "columns_dropped": [],
    }
    df = dataframe.copy()
    missing_before = int(df.isnull().sum().sum())

    if missing_before == 0:
        return df, report

    if strategy == "drop_rows":
        before = len(df)
        df = df.dropna()
        report["rows_dropped"] = before - len(df)
        return df, report

    if strategy == "drop_columns":
        missing_cols = [col for col in df.columns if df[col].isnull().any()]
        df = df.drop(columns=missing_cols)
        report["columns_dropped"] = missing_cols
        return df, report

    for column in df.columns:
        missing_count = int(df[column].isnull().sum())
        if missing_count == 0:
            continue

        if _is_categorical(df[column]) or strategy == "mode":
            fill_value = df[column].mode(dropna=True)
            value = fill_value.iloc[0] if not fill_value.empty else "Unknown"
        elif strategy == "median":
            value = df[column].median()
        else:
            value = df[column].mean()

        df[column] = df[column].fillna(value)
        report["columns_affected"][column] = {
            "missing_count": missing_count,
            "fill_value": str(value),
        }

    return df, report


def _detect_outliers(dataframe: pd.DataFrame, numeric_columns: list[str], method: str) -> dict[str, set[int]]:
    outliers: dict[str, set[int]] = {column: set() for column in numeric_columns}

    if method == "none":
        return outliers

    for column in numeric_columns:
        series = dataframe[column].dropna()
        if series.empty:
            continue

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (dataframe[column] < lower) | (dataframe[column] > upper)
        else:
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue
            z_scores = (dataframe[column] - mean) / std
            mask = z_scores.abs() > 3

        outliers[column] = set(dataframe.index[mask.fillna(False)])

    return outliers


def _encode_categorical_features(
    features: pd.DataFrame,
    categorical_columns: list[str],
    encoding: str,
) -> tuple[pd.DataFrame, dict]:
    encoded = features.copy()
    encoders: dict[str, list[str]] = {}

    if encoding == "one_hot":
        encoded = pd.get_dummies(encoded, columns=categorical_columns, drop_first=False)
        encoders["one_hot_columns"] = [
            column for column in encoded.columns if any(column.startswith(f"{cat}_") for cat in categorical_columns)
        ]
    else:
        label_maps: dict[str, dict[str, int]] = {}
        for column in categorical_columns:
            encoder = LabelEncoder()
            encoded[column] = encoder.fit_transform(encoded[column].astype(str))
            label_maps[column] = {
                str(label): int(index) for index, label in enumerate(encoder.classes_)
            }
        encoders["label_maps"] = label_maps

    return encoded, {
        "method": encoding,
        "categorical_columns": categorical_columns,
        "features_after_encoding": int(encoded.shape[1]),
        "details": encoders,
    }


def _scale_features(features: pd.DataFrame, numeric_columns: list[str], method: str) -> tuple[pd.DataFrame, dict]:
    scaled = features.copy()
    columns_to_scale = [column for column in numeric_columns if column in scaled.columns]

    if method == "none" or not columns_to_scale:
        return scaled, {"method": method, "scaled_columns": []}

    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    scaled[columns_to_scale] = scaler.fit_transform(scaled[columns_to_scale])
    return scaled, {"method": method, "scaled_columns": columns_to_scale}


def _can_stratify(target: pd.Series) -> bool:
    if target.isnull().any():
        return False
    class_counts = target.value_counts()
    return len(class_counts) > 1 and class_counts.min() >= 2


def _class_distribution(target: pd.Series | np.ndarray) -> dict[str, int]:
    series = target if isinstance(target, pd.Series) else pd.Series(target)
    return {str(index): int(count) for index, count in series.value_counts().items()}


def run_preprocessing_pipeline(
    dataset: Dataset,
    config: PreprocessingConfig,
    output_folder: Path,
) -> ProcessedDataset:
    """Execute the full preprocessing pipeline and persist outputs."""
    if config.target_column not in dataset.feature_names:
        raise PreprocessingServiceError("Selected target column was not found in the dataset.")

    dataframe = pd.read_csv(dataset.file_path)
    if dataframe.empty:
        raise PreprocessingServiceError("Dataset is empty and cannot be preprocessed.")

    report: dict[str, Any] = {
        "initial_rows": int(len(dataframe)),
        "initial_columns": int(len(dataframe.columns)),
        "target_column": config.target_column,
    }

    dataframe, missing_report = _handle_missing_values(dataframe, config.missing_strategy)
    report["missing_value_handling"] = missing_report

    if config.target_column not in dataframe.columns:
        raise PreprocessingServiceError("Target column was removed during missing value handling.")

    if dataframe.empty:
        raise PreprocessingServiceError("No rows remain after missing value handling.")

    feature_columns = [column for column in dataframe.columns if column != config.target_column]
    if not feature_columns:
        raise PreprocessingServiceError("No feature columns remain for preprocessing.")

    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    outlier_map = _detect_outliers(dataframe, numeric_columns, config.outlier_method)
    outlier_rows = set().union(*outlier_map.values()) if outlier_map else set()
    rows_removed = 0

    if config.outlier_action == "remove_rows" and outlier_rows:
        before = len(dataframe)
        dataframe = dataframe.drop(index=list(outlier_rows), errors="ignore").reset_index(drop=True)
        rows_removed = before - len(dataframe)

    report["outlier_detection"] = {
        "method": config.outlier_method,
        "action": config.outlier_action,
        "by_column": {column: len(indexes) for column, indexes in outlier_map.items()},
        "total_outlier_rows": len(outlier_rows),
        "rows_removed": rows_removed,
    }

    if dataframe.empty:
        raise PreprocessingServiceError("No rows remain after outlier removal.")

    features = dataframe[feature_columns].copy()
    for column in categorical_columns:
        features[column] = features[column].astype(str)
    target = dataframe[config.target_column].copy()

    missing_fills: dict[str, Any] = {}
    for column, details in missing_report.get("columns_affected", {}).items():
        missing_fills[column] = details.get("fill_value")

    from app.services.preprocessing_inference import (
        fit_preprocessor_from_prepared_features,
        save_preprocessor,
    )

    preprocessor = fit_preprocessor_from_prepared_features(
        features,
        categorical_columns,
        numeric_columns,
        config,
        missing_fills,
    )

    encoded_features, encoding_report = _encode_categorical_features(
        features,
        categorical_columns,
        config.categorical_encoding,
    )
    report["categorical_encoding"] = encoding_report

    numeric_after_encoding = [
        column for column in numeric_columns if column in encoded_features.columns
    ]
    scaled_features, scaling_report = _scale_features(
        encoded_features,
        numeric_after_encoding,
        config.scaling_method,
    )
    report["feature_scaling"] = scaling_report

    processed_df = scaled_features.copy()
    processed_df[config.target_column] = target.values

    stratify = target if _can_stratify(target) else None
    train_features, test_features, train_target, test_target = train_test_split(
        scaled_features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    report["train_test_split"] = {
        "test_size": config.test_size,
        "train_rows": int(len(train_features)),
        "test_rows": int(len(test_features)),
        "stratified": stratify is not None,
    }

    train_df = train_features.copy()
    train_df[config.target_column] = train_target.values
    test_df = test_features.copy()
    test_df[config.target_column] = test_target.values

    smote_report: dict[str, Any] = {"applied": False}
    train_smote_df = None

    if config.apply_smote:
        train_target_for_smote = train_target.copy()
        if not pd.api.types.is_numeric_dtype(train_target_for_smote):
            target_encoder = LabelEncoder()
            train_target_for_smote = target_encoder.fit_transform(train_target_for_smote.astype(str))
        else:
            train_target_for_smote = train_target_for_smote.astype(float).astype(int).values

        class_counts = pd.Series(train_target_for_smote).value_counts()
        if len(class_counts) < 2:
            raise PreprocessingServiceError("SMOTE requires at least two classes in the target column.")
        if class_counts.min() < 2:
            raise PreprocessingServiceError(
                "SMOTE requires at least two samples in each class of the training target."
            )

        smote = SMOTE(random_state=config.random_state)
        resampled_features, resampled_target = smote.fit_resample(train_features, train_target_for_smote)
        train_smote_df = pd.DataFrame(resampled_features, columns=train_features.columns)
        train_smote_df[config.target_column] = resampled_target

        smote_report = {
            "applied": True,
            "train_rows_before": int(len(train_features)),
            "train_rows_after": int(len(train_smote_df)),
            "class_distribution_before": _class_distribution(train_target),
            "class_distribution_after": _class_distribution(resampled_target),
        }

    report["smote"] = smote_report
    report["final_features"] = [str(column) for column in scaled_features.columns]
    report["final_rows"] = int(len(processed_df))

    output_folder.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
    base_name = f"user{dataset.user_id}_dataset{dataset.id}_{timestamp}"

    processed_path = output_folder / f"{base_name}_processed.csv"
    train_path = output_folder / f"{base_name}_train.csv"
    test_path = output_folder / f"{base_name}_test.csv"
    smote_path = None

    processed_df.to_csv(processed_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    if train_smote_df is not None:
        smote_path = output_folder / f"{base_name}_train_smote.csv"
        train_smote_df.to_csv(smote_path, index=False)

    preprocessor_path = output_folder / f"{base_name}_preprocessor.joblib"
    save_preprocessor(preprocessor, preprocessor_path)

    report["output_files"] = {
        "processed": processed_path.name,
        "train": train_path.name,
        "test": test_path.name,
        "train_smote": smote_path.name if smote_path else None,
        "preprocessor": preprocessor_path.name,
    }

    processed_record = ProcessedDataset(
        dataset_id=dataset.id,
        user_id=dataset.user_id,
        target_column=config.target_column,
        processed_file_path=str(processed_path),
        train_file_path=str(train_path),
        test_file_path=str(test_path),
        train_smote_file_path=str(smote_path) if smote_path else None,
        row_count_processed=int(len(processed_df)),
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        train_smote_rows=int(len(train_smote_df)) if train_smote_df is not None else None,
        feature_count=int(scaled_features.shape[1]),
    )
    processed_record.config = asdict(config)
    processed_record.report = report

    db.session.add(processed_record)
    db.session.commit()
    return processed_record


def get_user_processed_dataset(processed_id: int, user_id: int) -> ProcessedDataset | None:
    """Return a processed dataset owned by the given user."""
    return ProcessedDataset.query.filter_by(id=processed_id, user_id=user_id).first()
