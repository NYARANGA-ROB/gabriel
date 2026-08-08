"""Shared helpers for the ML Jupyter notebook workflow."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
PROCESSED_DIR = UPLOADS_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "app" / "ml" / "models"
PIPELINE_STATE_PATH = DATA_DIR / "pipeline_state.json"

MIN_DATASET_ROWS = 200_000
DEFAULT_TARGET_ROWS = 200_000
DEFAULT_CSV = DATA_DIR / "diabetes_readmission_200k.csv"
TARGET_COLUMN = "readmitted"
USER_EMAIL = "gabriel.odiah@gmail.com"  # change to your login email when syncing to the web app

ML_LIFECYCLE_STAGES = [
    "Problem Definition",
    "Data Acquisition",
    "Data Understanding",
    "Data Cleaning & Preprocessing",
    "Exploratory Data Analysis (EDA)",
    "Feature Engineering",
    "Feature Selection",
    "Data Splitting",
    "Model Selection",
    "Model Training",
    "Hyperparameter Tuning",
    "Model Evaluation",
    "Model Interpretation",
    "Model Deployment",
    "Monitoring & Maintenance",
    "Model Retraining",
]


@contextmanager
def flask_app_context():
    """Open a Flask application context for DB and config access."""
    from app import create_app

    app = create_app()
    with app.app_context():
        yield app


def ensure_project_paths() -> None:
    """Create folders used by notebooks and the web app."""
    for folder in (DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def validate_dataset_rows(
    data: pd.DataFrame | Path | int,
    *,
    minimum: int = MIN_DATASET_ROWS,
) -> int:
    """Ensure the dataset meets the minimum row requirement."""
    if isinstance(data, int):
        row_count = data
    elif isinstance(data, Path):
        row_count = sum(1 for _ in open(data)) - 1
    else:
        row_count = len(data)

    if row_count < minimum:
        raise ValueError(
            f"Dataset has {row_count:,} rows but at least {minimum:,} are required."
        )
    return row_count


def save_pipeline_state(state: dict) -> None:
    """Persist notebook pipeline artefacts for monitoring/retraining."""
    import json

    ensure_project_paths()
    PIPELINE_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_pipeline_state() -> dict:
    """Load persisted pipeline state if available."""
    import json

    if not PIPELINE_STATE_PATH.exists():
        return {}
    return json.loads(PIPELINE_STATE_PATH.read_text(encoding="utf-8"))
