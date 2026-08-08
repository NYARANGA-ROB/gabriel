"""Preprocessing configuration form."""

from flask_wtf import FlaskForm
from wtforms import BooleanField, FloatField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, NumberRange, Optional


class PreprocessingForm(FlaskForm):
    """Configure and run the preprocessing pipeline."""

    target_column = SelectField(
        "Target column",
        validators=[DataRequired()],
        coerce=str,
    )
    missing_strategy = SelectField(
        "Missing value handling",
        choices=[
            ("mean", "Mean imputation (numeric) / mode (categorical)"),
            ("median", "Median imputation (numeric) / mode (categorical)"),
            ("mode", "Mode imputation (all columns)"),
            ("drop_rows", "Drop rows with missing values"),
            ("drop_columns", "Drop columns with missing values"),
        ],
        default="mean",
        validators=[DataRequired()],
    )
    categorical_encoding = SelectField(
        "Categorical encoding",
        choices=[
            ("one_hot", "One-hot encoding"),
            ("label", "Label encoding"),
        ],
        default="one_hot",
        validators=[DataRequired()],
    )
    scaling_method = SelectField(
        "Feature scaling",
        choices=[
            ("standard", "StandardScaler (z-score)"),
            ("minmax", "MinMaxScaler (0–1)"),
            ("none", "No scaling"),
        ],
        default="standard",
        validators=[DataRequired()],
    )
    outlier_method = SelectField(
        "Outlier detection",
        choices=[
            ("iqr", "Interquartile range (IQR)"),
            ("zscore", "Z-score (|z| > 3)"),
            ("none", "None"),
        ],
        default="iqr",
        validators=[DataRequired()],
    )
    outlier_action = SelectField(
        "Outlier action",
        choices=[
            ("report_only", "Report only"),
            ("remove_rows", "Remove outlier rows"),
        ],
        default="report_only",
        validators=[DataRequired()],
    )
    test_size = FloatField(
        "Test set proportion",
        default=0.2,
        validators=[DataRequired(), NumberRange(min=0.1, max=0.5)],
    )
    apply_smote = BooleanField("Apply SMOTE to training set", default=False)
    random_state = IntegerField(
        "Random seed",
        default=42,
        validators=[Optional(), NumberRange(min=0, max=99999)],
    )
    submit = SubmitField("Run preprocessing pipeline")
