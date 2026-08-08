"""Patient prediction forms."""

from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField, StringField, SubmitField
from wtforms.validators import DataRequired, NumberRange, Optional

from app.utils.patient_fields import PATIENT_FIELDS, field_choices


def _coerce_optional_int(value):
    if value in (None, ""):
        return None
    return int(value)


class PatientPredictionForm(FlaskForm):
    """Collect patient information and model selection for inference."""

    trained_model_id = SelectField(
        "Model",
        validators=[DataRequired()],
        coerce=int,
    )
    patient_id = StringField(
        "Patient ID",
        validators=[DataRequired()],
        render_kw={"placeholder": "e.g. P-10042"},
    )

    age = IntegerField("Age", validators=[DataRequired(), NumberRange(min=0, max=120)])
    gender = SelectField(
        "Gender",
        validators=[DataRequired()],
        choices=[("", "Select gender")] + field_choices("gender"),
    )
    race = SelectField(
        "Race",
        validators=[DataRequired()],
        choices=[("", "Select race")] + field_choices("race"),
    )
    admission_type = SelectField(
        "Admission Type",
        validators=[DataRequired()],
        choices=[("", "Select admission type")] + field_choices("admission_type"),
    )
    discharge_disposition = SelectField(
        "Discharge Disposition",
        validators=[DataRequired()],
        choices=[("", "Select disposition")] + field_choices("discharge_disposition"),
    )
    length_of_stay = IntegerField(
        "Length of Stay",
        validators=[DataRequired(), NumberRange(min=0, max=365)],
    )
    time_in_hospital = IntegerField(
        "Time in Hospital",
        validators=[DataRequired(), NumberRange(min=0, max=365)],
    )
    number_of_procedures = IntegerField(
        "Number of Procedures",
        validators=[DataRequired(), NumberRange(min=0, max=100)],
    )
    number_of_medications = IntegerField(
        "Number of Medications",
        validators=[DataRequired(), NumberRange(min=0, max=100)],
    )
    emergency_visits = IntegerField(
        "Emergency Visits",
        validators=[Optional(), NumberRange(min=0, max=100)],
    )
    outpatient_visits = IntegerField(
        "Outpatient Visits",
        validators=[Optional(), NumberRange(min=0, max=100)],
    )
    previous_admissions = IntegerField(
        "Previous Admissions",
        validators=[Optional(), NumberRange(min=0, max=100)],
    )
    primary_diagnosis = StringField(
        "Primary Diagnosis",
        validators=[DataRequired()],
    )
    secondary_diagnosis = StringField(
        "Secondary Diagnosis",
        validators=[Optional()],
    )
    diabetes_medication = SelectField(
        "Diabetes Medication",
        validators=[DataRequired()],
        choices=[("", "Select option")] + field_choices("diabetes_medication"),
    )
    insulin = SelectField(
        "Insulin",
        validators=[DataRequired()],
        choices=[("", "Select option")] + field_choices("insulin"),
    )
    a1c_result = SelectField(
        "A1C Result",
        validators=[DataRequired()],
        choices=[("", "Select result")] + field_choices("a1c_result"),
    )
    max_glucose_serum = SelectField(
        "Max Glucose Serum",
        validators=[DataRequired()],
        choices=[("", "Select result")] + field_choices("max_glucose_serum"),
    )

    submit = SubmitField("Generate prediction")

    def patient_payload(self) -> dict:
        """Return form values as a patient data dictionary."""
        payload = {}
        for field in PATIENT_FIELDS:
            value = getattr(self, field.key).data
            if value not in (None, ""):
                payload[field.key] = value
        return payload
