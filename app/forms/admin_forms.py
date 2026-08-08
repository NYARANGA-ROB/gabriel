"""Admin panel forms."""

from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, SubmitField
from wtforms.validators import Optional


class AdminSearchForm(FlaskForm):
    """Generic search form for admin list views."""

    search = StringField("Search", validators=[Optional()])
    submit = SubmitField("Search")


class AdminPredictionFilterForm(FlaskForm):
    """Filter all prediction logs in the admin panel."""

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
