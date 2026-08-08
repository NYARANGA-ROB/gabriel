"""Exploratory data analysis configuration form."""

from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField
from wtforms.validators import Optional


class EDAForm(FlaskForm):
    """Configure columns used in exploratory charts."""

    class Meta:
        csrf = False

    target_column = SelectField("Target column", validators=[Optional()], coerce=str)
    feature_column = SelectField("Feature for distribution / histogram / boxplot", validators=[Optional()], coerce=str)
    x_column = SelectField("Scatter plot X-axis", validators=[Optional()], coerce=str)
    y_column = SelectField("Scatter plot Y-axis", validators=[Optional()], coerce=str)
    submit = SubmitField("Update charts")
