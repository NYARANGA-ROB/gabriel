"""Dataset model for uploaded CSV files."""

import json
from datetime import datetime

from app.extensions import db


class Dataset(db.Model):
    """Uploaded dataset metadata and analysis summary."""

    __tablename__ = "datasets"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), nullable=False, unique=True)
    file_path = db.Column(db.String(512), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    row_count = db.Column(db.Integer, nullable=False)
    column_count = db.Column(db.Integer, nullable=False)
    missing_values = db.Column(db.Integer, nullable=False, default=0)
    duplicate_rows = db.Column(db.Integer, nullable=False, default=0)
    feature_names_json = db.Column(db.Text, nullable=False)
    missing_by_column_json = db.Column(db.Text, nullable=False, default="{}")
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    user = db.relationship("User", backref=db.backref("datasets", lazy="dynamic"))

    @property
    def feature_names(self) -> list[str]:
        return json.loads(self.feature_names_json)

    @feature_names.setter
    def feature_names(self, names: list[str]) -> None:
        self.feature_names_json = json.dumps(names)

    @property
    def missing_by_column(self) -> dict[str, int]:
        return json.loads(self.missing_by_column_json)

    @missing_by_column.setter
    def missing_by_column(self, values: dict[str, int]) -> None:
        self.missing_by_column_json = json.dumps(values)

    def __repr__(self) -> str:
        return f"<Dataset {self.original_filename}>"
