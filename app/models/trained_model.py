"""Trained machine learning model metadata."""

import json
from datetime import datetime

from sqlalchemy import desc

from app.extensions import db


class TrainedModel(db.Model):
    """Persisted ML model training run."""

    __tablename__ = "trained_models"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    processed_dataset_id = db.Column(
        db.Integer, db.ForeignKey("processed_datasets.id"), nullable=False, index=True
    )
    retrained_from_id = db.Column(db.Integer, db.ForeignKey("trained_models.id"), nullable=True)
    model_type = db.Column(db.String(64), nullable=False)
    model_name = db.Column(db.String(128), nullable=False)
    status = db.Column(db.String(32), nullable=False, default="queued")
    progress_percent = db.Column(db.Integer, nullable=False, default=0)
    model_file_path = db.Column(db.String(512), nullable=True)
    parameters_json = db.Column(db.Text, nullable=False)
    metrics_json = db.Column(db.Text, nullable=True)
    progress_log_json = db.Column(db.Text, nullable=False, default="[]")
    training_data_source = db.Column(db.String(32), nullable=False, default="train")
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    completed_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship("User", backref=db.backref("trained_models", lazy="dynamic"))
    processed_dataset = db.relationship(
        "ProcessedDataset", backref=db.backref("trained_models", lazy="dynamic")
    )
    retrained_from = db.relationship("TrainedModel", remote_side=[id], backref="retrains")

    @property
    def parameters(self) -> dict:
        return json.loads(self.parameters_json)

    @parameters.setter
    def parameters(self, value: dict) -> None:
        self.parameters_json = json.dumps(value)

    @property
    def metrics(self) -> dict | None:
        return json.loads(self.metrics_json) if self.metrics_json else None

    @metrics.setter
    def metrics(self, value: dict | None) -> None:
        self.metrics_json = json.dumps(value) if value else None

    @property
    def progress_log(self) -> list[dict]:
        return json.loads(self.progress_log_json)

    @progress_log.setter
    def progress_log(self, value: list[dict]) -> None:
        self.progress_log_json = json.dumps(value)

    def __repr__(self) -> str:
        return f"<TrainedModel {self.model_name} id={self.id}>"

    @property
    def latest_evaluation(self):
        from app.models.model_evaluation import ModelEvaluation

        return self.evaluations.order_by(desc(ModelEvaluation.evaluated_at)).first()
