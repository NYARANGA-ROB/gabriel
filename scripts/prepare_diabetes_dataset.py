"""Prepare the UCI Diabetes 130-US Hospitals dataset for this application."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ZIP_PATH = DATA_DIR / "diabetes_uci.zip"
RAW_CSV = DATA_DIR / "diabetic_data.csv"
OUTPUT_CSV = DATA_DIR / "diabetes_readmission_full.csv"
OUTPUT_EXPANDED_CSV = DATA_DIR / "diabetes_readmission_200k.csv"
OUTPUT_1M_CSV = DATA_DIR / "diabetes_readmission_1m.csv"
LEGACY_OUTPUT_CSV = DATA_DIR / "diabetes_readmission_clean.csv"
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
BALANCED_SAMPLE_SIZE = 12_000
MIN_DATASET_ROWS = 200_000
DEFAULT_TARGET_ROWS = 200_000
INTEGER_COLUMNS = {
    "time_in_hospital",
    "number_of_procedures",
    "lab_procedures",
    "number_of_medications",
    "diagnosis_count",
    "outpatient_visits",
    "emergency_visits",
    "previous_admissions",
    "total_prior_visits",
}


def _ensure_raw_csv() -> Path:
    """Download and extract the UCI dataset if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_CSV.exists():
        if not ZIP_PATH.exists():
            import urllib.request

            print(f"Downloading {UCI_ZIP_URL} ...")
            urllib.request.urlretrieve(UCI_ZIP_URL, ZIP_PATH)

        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extract("diabetic_data.csv", DATA_DIR)

    return RAW_CSV


def _load_id_mapping() -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    admission_map = {
        1: "Emergency",
        2: "Urgent",
        3: "Elective",
        4: "Newborn",
        5: "Not Available",
        6: "Unknown",
        7: "Trauma Center",
        8: "Not Mapped",
    }
    discharge_map = {
        1: "Discharged to home",
        2: "Transferred to hospital",
        3: "Discharged/transferred to SNF",
        6: "Discharged/transferred to home with home health service",
        7: "Left AMA",
        11: "Expired",
        25: "Not Mapped",
    }
    admission_source_map = {
        1: "Physician Referral",
        2: "Clinic Referral",
        3: "HMO Referral",
        4: "Transfer from Hospital",
        5: "Transfer from SNF",
        6: "Transfer from Other",
        7: "Emergency",
        8: "Court/Law Enforcement",
        9: "Not Available",
        10: "Transfer from Critical Access",
        11: "Normal Delivery",
        12: "Premature Delivery",
        13: "Sick Baby",
        14: "Medical Extra",
        15: "Not Mapped",
        17: "Transfer from Another Facility",
        20: "Not Mapped",
        18: "Transfer from Same Facility",
        19: "Not Mapped",
        22: "Transfer from Rehab",
        25: "Not Mapped",
    }
    return admission_map, discharge_map, admission_source_map


def _simplify_discharge(description: str) -> str:
    text = description.lower()
    if "expired" in text:
        return "Expired"
    if "ama" in text or "against medical advice" in text:
        return "AMA"
    if "snf" in text or "nursing facility" in text:
        return "SNF"
    if "home health" in text:
        return "Home Health"
    if "home" in text:
        return "Home"
    return "Other"


def _group_diagnosis(code) -> str:
    if pd.isna(code):
        return "Unknown"
    text = str(code).strip()
    if not text or text == "?":
        return "Unknown"
    prefix = text[:3].upper()
    if prefix.startswith("250") or prefix.startswith("E11") or prefix.startswith("E10"):
        return "Diabetes"
    if prefix.startswith("401") or prefix.startswith("428") or prefix.startswith("I"):
        return "Circulatory"
    if prefix.startswith("486") or prefix.startswith("J"):
        return "Respiratory"
    if prefix.startswith("V"):
        return "Injury/External"
    if prefix.startswith("C"):
        return "Neoplasms"
    return "Other"


def _top_category(series: pd.Series, top_n: int = 12, other_label: str = "Other") -> pd.Series:
    counts = series.value_counts()
    keep = set(counts.head(top_n).index.astype(str))
    return series.astype(str).where(series.astype(str).isin(keep), other_label)


def _expand_to_target_rows(
    cleaned: pd.DataFrame,
    target_rows: int,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap real encounters and perturb numeric fields to reach the target size."""
    if target_rows <= len(cleaned):
        return cleaned.sample(n=target_rows, random_state=random_state).reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    sampled_indices = rng.choice(len(cleaned), size=target_rows, replace=True)
    expanded = cleaned.iloc[sampled_indices].copy().reset_index(drop=True)

    numeric_columns = [
        column
        for column in expanded.columns
        if column != "readmitted" and pd.api.types.is_numeric_dtype(expanded[column])
    ]

    for column in numeric_columns:
        values = expanded[column].astype(float).to_numpy()
        spread = float(np.std(values))
        noise_scale = max(spread * 0.03, 0.05)
        updated = values + rng.normal(0, noise_scale, size=len(values))
        if column in INTEGER_COLUMNS:
            updated = np.round(np.clip(updated, 0, None))
        expanded[column] = updated

    return expanded


def _resolve_output_path(
    *,
    use_full: bool,
    balanced: bool,
    sample_size: int | None,
    target_rows: int | None,
) -> Path:
    if target_rows is not None and target_rows >= 1_000_000:
        return OUTPUT_1M_CSV
    if target_rows is not None and target_rows >= MIN_DATASET_ROWS:
        return OUTPUT_EXPANDED_CSV
    if use_full and not balanced and sample_size is None and target_rows is None:
        return OUTPUT_CSV
    return LEGACY_OUTPUT_CSV


def prepare_diabetes_csv(
    *,
    use_full: bool = True,
    sample_size: int | None = None,
    balanced: bool = False,
    target_rows: int | None = DEFAULT_TARGET_ROWS,
) -> Path:
    """Build a cleaned CSV aligned with application patient fields."""
    if target_rows is not None and target_rows < MIN_DATASET_ROWS:
        raise ValueError(
            f"target_rows must be at least {MIN_DATASET_ROWS:,} (got {target_rows:,})."
        )

    raw_path = _ensure_raw_csv()
    admission_map, discharge_map, admission_source_map = _load_id_mapping()

    dataframe = pd.read_csv(raw_path)
    dataframe = dataframe.replace("?", pd.NA)
    dataframe = dataframe[dataframe["readmitted"].isin(["<30", ">30", "NO"])].copy()
    dataframe["readmitted"] = (dataframe["readmitted"] == "<30").astype(int)

    dataframe["admission_type"] = (
        dataframe["admission_type_id"].map(admission_map).fillna("Unknown")
    )
    dataframe["discharge_disposition"] = (
        dataframe["discharge_disposition_id"]
        .map(discharge_map)
        .fillna("Unknown")
        .map(_simplify_discharge)
    )
    dataframe["admission_source"] = (
        dataframe["admission_source_id"].map(admission_source_map).fillna("Unknown")
    )
    dataframe["medical_specialty"] = _top_category(
        dataframe["medical_specialty"].fillna("Unknown"), top_n=12
    )
    dataframe["payer_code"] = _top_category(dataframe["payer_code"].fillna("Unknown"), top_n=8)

    dataframe["primary_diagnosis_group"] = dataframe["diag_1"].map(_group_diagnosis)
    dataframe["secondary_diagnosis_group"] = dataframe["diag_2"].map(_group_diagnosis)
    dataframe["total_prior_visits"] = (
        dataframe["number_outpatient"].fillna(0)
        + dataframe["number_emergency"].fillna(0)
        + dataframe["number_inpatient"].fillna(0)
    )

    dataframe = dataframe.rename(
        columns={
            "number_outpatient": "outpatient_visits",
            "number_emergency": "emergency_visits",
            "number_inpatient": "previous_admissions",
            "diag_1": "primary_diagnosis",
            "diag_2": "secondary_diagnosis",
            "diabetesMed": "diabetes_medication",
            "A1Cresult": "a1c_result",
            "max_glu_serum": "max_glucose_serum",
            "num_procedures": "number_of_procedures",
            "num_medications": "number_of_medications",
            "num_lab_procedures": "lab_procedures",
            "number_diagnoses": "diagnosis_count",
        }
    )

    columns = [
        "race",
        "gender",
        "age",
        "admission_type",
        "admission_source",
        "discharge_disposition",
        "medical_specialty",
        "payer_code",
        "time_in_hospital",
        "number_of_procedures",
        "lab_procedures",
        "number_of_medications",
        "diagnosis_count",
        "outpatient_visits",
        "emergency_visits",
        "previous_admissions",
        "total_prior_visits",
        "primary_diagnosis",
        "secondary_diagnosis",
        "primary_diagnosis_group",
        "secondary_diagnosis_group",
        "diabetes_medication",
        "insulin",
        "a1c_result",
        "max_glucose_serum",
        "readmitted",
    ]
    cleaned = dataframe[columns].copy()

    for column in columns:
        if column == "readmitted":
            continue
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].fillna("Unknown")
        else:
            cleaned[column] = cleaned[column].fillna(0)

    if balanced or sample_size is not None:
        target_size = sample_size or BALANCED_SAMPLE_SIZE
        positive = cleaned[cleaned["readmitted"] == 1]
        negative = cleaned[cleaned["readmitted"] == 0]
        if balanced:
            half = target_size // 2
            sampled = pd.concat(
                [
                    positive.sample(n=min(len(positive), half), random_state=42),
                    negative.sample(n=min(len(negative), half), random_state=42),
                ],
                ignore_index=True,
            )
        else:
            sampled = cleaned.sample(n=min(len(cleaned), target_size), random_state=42)
        cleaned = sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    elif not use_full and len(cleaned) > BALANCED_SAMPLE_SIZE:
        cleaned = cleaned.sample(n=BALANCED_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    elif target_rows is not None and len(cleaned) < target_rows:
        source_rows = len(cleaned)
        cleaned = _expand_to_target_rows(cleaned, target_rows)
        print(
            f"Expanded {source_rows:,} real UCI encounters to {len(cleaned):,} rows "
            f"(bootstrap + numeric perturbation)."
        )

    output_path = _resolve_output_path(
        use_full=use_full,
        balanced=balanced,
        sample_size=sample_size,
        target_rows=target_rows,
    )
    if len(cleaned) < MIN_DATASET_ROWS and target_rows is not None:
        raise ValueError(
            f"Prepared dataset has {len(cleaned):,} rows; minimum required is {MIN_DATASET_ROWS:,}."
        )
    cleaned.to_csv(output_path, index=False)
    print(
        f"Prepared {output_path.name}: {len(cleaned):,} rows, "
        f"{cleaned['readmitted'].mean():.1%} readmission rate, "
        f"{len(columns) - 1} features"
    )
    return output_path


if __name__ == "__main__":
    prepare_diabetes_csv()
