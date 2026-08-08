"""Patient input field definitions for readmission prediction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatientField:
    """A single patient attribute shown on the prediction form."""

    key: str
    label: str
    field_type: str
    aliases: tuple[str, ...]
    choices: tuple[tuple[str, str], ...] = ()
    help_text: str = ""


def _normalize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


PATIENT_FIELDS: tuple[PatientField, ...] = (
    PatientField("age", "Age", "integer", ("age",), help_text="Patient age in years"),
    PatientField(
        "gender",
        "Gender",
        "select",
        ("gender", "sex"),
        (("M", "Male"), ("F", "Female"), ("Unknown", "Unknown")),
    ),
    PatientField(
        "race",
        "Race",
        "select",
        ("race", "ethnicity"),
        (
            ("Caucasian", "Caucasian"),
            ("AfricanAmerican", "African American"),
            ("Hispanic", "Hispanic"),
            ("Asian", "Asian"),
            ("Other", "Other"),
        ),
    ),
    PatientField(
        "admission_type",
        "Admission Type",
        "select",
        ("admission_type", "admissiontype"),
        (
            ("Emergency", "Emergency"),
            ("Urgent", "Urgent"),
            ("Elective", "Elective"),
        ),
    ),
    PatientField(
        "discharge_disposition",
        "Discharge Disposition",
        "select",
        ("discharge_disposition", "discharged_to"),
        (
            ("Home", "Home"),
            ("SNF", "Skilled nursing facility"),
            ("Home Health", "Home with home health"),
            ("AMA", "Against medical advice"),
            ("Expired", "Expired"),
        ),
    ),
    PatientField(
        "length_of_stay",
        "Length of Stay",
        "integer",
        ("length_of_stay", "los"),
        help_text="Total length of stay in days",
    ),
    PatientField(
        "time_in_hospital",
        "Time in Hospital",
        "integer",
        ("time_in_hospital",),
        help_text="Days spent in hospital during this encounter",
    ),
    PatientField(
        "number_of_procedures",
        "Number of Procedures",
        "integer",
        ("number_of_procedures", "num_procedures"),
    ),
    PatientField(
        "number_of_medications",
        "Number of Medications",
        "integer",
        ("number_of_medications", "num_medications"),
    ),
    PatientField(
        "emergency_visits",
        "Emergency Visits",
        "integer",
        ("emergency_visits", "number_emergency"),
        help_text="Emergency visits in the prior year",
    ),
    PatientField(
        "outpatient_visits",
        "Outpatient Visits",
        "integer",
        ("outpatient_visits", "number_outpatient"),
        help_text="Outpatient visits in the prior year",
    ),
    PatientField(
        "previous_admissions",
        "Previous Admissions",
        "integer",
        ("previous_admissions", "number_inpatient", "prior_inpatient"),
        help_text="Inpatient admissions in the prior year",
    ),
    PatientField(
        "primary_diagnosis",
        "Primary Diagnosis",
        "text",
        ("primary_diagnosis", "diagnosis", "diag_1"),
    ),
    PatientField(
        "secondary_diagnosis",
        "Secondary Diagnosis",
        "text",
        ("secondary_diagnosis", "diag_2"),
    ),
    PatientField(
        "diabetes_medication",
        "Diabetes Medication",
        "select",
        ("diabetes_medication", "diabetesmed", "diabetes_med"),
        (
            ("Yes", "Yes"),
            ("No", "No"),
            ("No change", "No change"),
        ),
    ),
    PatientField(
        "insulin",
        "Insulin",
        "select",
        ("insulin",),
        (("Yes", "Yes"), ("No", "No")),
    ),
    PatientField(
        "a1c_result",
        "A1C Result",
        "select",
        ("a1c_result", "a1cresult"),
        (
            (">7", "> 7"),
            ("<=7", "<= 7"),
            ("None", "Not measured"),
        ),
    ),
    PatientField(
        "max_glucose_serum",
        "Max Glucose Serum",
        "select",
        ("max_glucose_serum", "max_glu_serum"),
        (
            (">200", "> 200"),
            (">300", "> 300"),
            ("Norm", "Normal"),
            ("None", "Not measured"),
        ),
    ),
)


def field_choices(key: str) -> list[tuple[str, str]]:
    """Return select choices for a patient field key."""
    for field in PATIENT_FIELDS:
        if field.key == key:
            return list(field.choices)
    return []


def resolve_dataset_column(field: PatientField, available_columns: list[str]) -> str | None:
    """Map a patient field to the first matching dataset column name."""
    normalized = {_normalize_column_name(column): column for column in available_columns}
    for alias in field.aliases:
        match = normalized.get(_normalize_column_name(alias))
        if match:
            return match
    return None


def map_patient_input_to_row(
    patient_data: dict[str, object],
    available_columns: list[str],
    *,
    target_column: str | None = None,
) -> dict[str, object]:
    """Convert form values into a raw dataset row using column aliases."""
    row: dict[str, object] = {}
    excluded = {target_column} if target_column else set()

    for field in PATIENT_FIELDS:
        column = resolve_dataset_column(field, available_columns)
        if column is None or column in excluded:
            continue
        value = patient_data.get(field.key)
        if value is None or value == "":
            continue
        row[column] = value

    return row
