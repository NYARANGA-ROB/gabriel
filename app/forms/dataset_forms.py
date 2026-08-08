"""Dataset upload form."""

from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileRequired
from wtforms import FileField, SubmitField


class DatasetUploadForm(FlaskForm):
    """CSV dataset upload form."""

    dataset_file = FileField(
        "CSV file",
        validators=[
            FileRequired(message="Please select a CSV file to upload."),
            FileAllowed(["csv"], message="Only CSV files are supported."),
        ],
    )
    submit = SubmitField("Upload dataset")
