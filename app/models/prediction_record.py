"""Stored patient prediction history."""

from datetime import datetime

from app.extensions import db


class PredictionRecord(db.Model):
    """A single patient readmission prediction audit entry."""

    __tablename__ = "prediction_records"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    trained_model_id = db.Column(
        db.Integer, db.ForeignKey("trained_models.id"), nullable=False, index=True
    )
    patient_id = db.Column(db.String(64), nullable=False, index=True)
    prediction = db.Column(db.String(32), nullable=False)
    prediction_label = db.Column(db.String(64), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(32), nullable=False)
    model_name = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    user = db.relationship("User", backref=db.backref("prediction_records", lazy="dynamic"))
    trained_model = db.relationship(
        "TrainedModel", backref=db.backref("prediction_records", lazy="dynamic")
    )

    def __repr__(self) -> str:
        return f"<PredictionRecord patient={self.patient_id} id={self.id}>"
