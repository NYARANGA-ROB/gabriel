"""Processed dataset model."""

import json
from datetime import datetime

from app.extensions import db


class ProcessedDataset(db.Model):
    """Stored output and report from a preprocessing run."""

    __tablename__ = "processed_datasets"

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("datasets.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    target_column = db.Column(db.String(255), nullable=False)
    config_json = db.Column(db.Text, nullable=False)
    report_json = db.Column(db.Text, nullable=False)
    processed_file_path = db.Column(db.String(512), nullable=False)
    train_file_path = db.Column(db.String(512), nullable=False)
    test_file_path = db.Column(db.String(512), nullable=False)
    train_smote_file_path = db.Column(db.String(512), nullable=True)
    row_count_processed = db.Column(db.Integer, nullable=False)
    train_rows = db.Column(db.Integer, nullable=False)
    test_rows = db.Column(db.Integer, nullable=False)
    train_smote_rows = db.Column(db.Integer, nullable=True)
    feature_count = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    dataset = db.relationship("Dataset", backref=db.backref("processed_runs", lazy="dynamic"))
    user = db.relationship("User", backref=db.backref("processed_datasets", lazy="dynamic"))

    @property
    def config(self) -> dict:
        return json.loads(self.config_json)

    @config.setter
    def config(self, value: dict) -> None:
        self.config_json = json.dumps(value)

    @property
    def report(self) -> dict:
        return json.loads(self.report_json)

    @report.setter
    def report(self, value: dict) -> None:
        self.report_json = json.dumps(value)

    def __repr__(self) -> str:
        return f"<ProcessedDataset dataset={self.dataset_id} id={self.id}>"
