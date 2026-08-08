"""Prediction history filter forms."""

from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, SubmitField
from wtforms.validators import Optional


class PredictionHistoryFilterForm(FlaskForm):
    """Search and filter prediction history records."""

    search = StringField("Search", validators=[Optional()])
    model_id = SelectField("Model", coerce=int, validators=[Optional()])
    risk_level = SelectField(
        "Risk level",
        choices=[
            ("", "All risk levels"),
            ("High", "High"),
            ("Moderate", "Moderate"),
            ("Low", "Low"),
        ],
        validators=[Optional()],
    )
    submit = SubmitField("Apply filters")
