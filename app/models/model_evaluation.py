"""Model evaluation history."""

import json
from datetime import datetime

from app.extensions import db


class ModelEvaluation(db.Model):
    """Stored evaluation results for a trained model."""

    __tablename__ = "model_evaluations"

    id = db.Column(db.Integer, primary_key=True)
    trained_model_id = db.Column(
        db.Integer, db.ForeignKey("trained_models.id"), nullable=False, index=True
    )
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    roc_auc = db.Column(db.Float, nullable=True)
    training_time_seconds = db.Column(db.Float, nullable=False)
    prediction_time_seconds = db.Column(db.Float, nullable=False)
    metrics_json = db.Column(db.Text, nullable=False)
    confusion_matrix_json = db.Column(db.Text, nullable=False)
    charts_json = db.Column(db.Text, nullable=False)
    evaluated_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    trained_model = db.relationship(
        "TrainedModel", backref=db.backref("evaluations", lazy="dynamic", cascade="all, delete-orphan")
    )
    user = db.relationship("User", backref=db.backref("model_evaluations", lazy="dynamic"))

    @property
    def metrics(self) -> dict:
        return json.loads(self.metrics_json)

    @metrics.setter
    def metrics(self, value: dict) -> None:
        self.metrics_json = json.dumps(value)

    @property
    def confusion_matrix(self) -> dict:
        return json.loads(self.confusion_matrix_json)

    @confusion_matrix.setter
    def confusion_matrix(self, value: dict) -> None:
        self.confusion_matrix_json = json.dumps(value)

    @property
    def charts(self) -> dict:
        return json.loads(self.charts_json)

    @charts.setter
    def charts(self, value: dict) -> None:
        self.charts_json = json.dumps(value)

    def __repr__(self) -> str:
        return f"<ModelEvaluation model={self.trained_model_id} id={self.id}>"
