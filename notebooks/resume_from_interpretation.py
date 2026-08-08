#!/usr/bin/env python3
"""Resume notebook stages from evaluation onward using existing trained models."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import nbformat
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from nbclient import NotebookClient

from ml_utils import (
    DEFAULT_CSV,
    MODELS_DIR,
    PIPELINE_STATE_PATH,
    TARGET_COLUMN,
    USER_EMAIL,
    flask_app_context,
    load_pipeline_state,
    save_pipeline_state,
)

NOTEBOOK = Path(__file__).with_name("hospital_readmission_ml_pipeline.ipynb")
EXECUTED = Path(__file__).with_name("hospital_readmission_ml_pipeline.executed.ipynb")
FIGURES_DIR = Path(__file__).with_name("figures")


def _as_figure(figure):
    if figure is None:
        return None
    if isinstance(figure, go.Figure):
        return figure
    if isinstance(figure, dict):
        return go.Figure(figure)
    if isinstance(figure, list):
        return go.Figure(data=figure)
    return None


def save_charts(prefix: str, charts: dict) -> list[str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, raw in (charts or {}).items():
        fig = _as_figure(raw)
        if fig is None:
            continue
        out = FIGURES_DIR / f"{prefix}_{name}.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
        saved.append(str(out))
        print(f"saved {out}")
    return saved


def bootstrap_globals() -> dict:
    """Restore notebook variables from the latest DB artefacts."""
    from app.extensions import db
    from app.ml.comparison import build_model_comparison
    from app.ml.evaluation import evaluate_model
    from app.models.dataset import Dataset
    from app.models.processed_dataset import ProcessedDataset
    from app.models.trained_model import TrainedModel
    from app.models.user import User

    with flask_app_context() as app:
        user = User.query.filter_by(email=USER_EMAIL).first()
        if user is None:
            raise RuntimeError(f"User {USER_EMAIL} not found")

        dataset = (
            Dataset.query.filter_by(user_id=user.id, original_filename=DEFAULT_CSV.name)
            .order_by(Dataset.id.desc())
            .first()
        )
        if dataset is None:
            raise RuntimeError("200k dataset not found for user; run full notebook first")

        processed = (
            ProcessedDataset.query.filter_by(user_id=user.id, dataset_id=dataset.id)
            .order_by(ProcessedDataset.id.desc())
            .first()
        )
        if processed is None:
            raise RuntimeError("Processed dataset missing")

        train_path = processed.train_smote_file_path or processed.train_file_path
        test_path = processed.test_file_path
        train_df = pd.read_csv(train_path, nrows=5)
        test_df = pd.read_csv(test_path)
        selected = [c for c in train_df.columns if c != TARGET_COLUMN]

        comparison = build_model_comparison(user.id)
        best_id = comparison["best_model_id"]
        if best_id is None and comparison["models"]:
            best_id = max(comparison["models"], key=lambda row: row.get("f1_score") or 0)["id"]
        # Prefer a recent 200k model if "best" points at an old tiny dataset.
        recent = (
            TrainedModel.query.filter_by(user_id=user.id, status="completed")
            .filter(TrainedModel.processed_dataset_id == processed.id)
            .order_by(TrainedModel.id.desc())
            .all()
        )
        if recent:
            recent_ids = {m.id for m in recent}
            if best_id not in recent_ids:
                scored = [r for r in comparison["models"] if r["id"] in recent_ids]
                if scored:
                    best_id = max(scored, key=lambda row: row.get("f1_score") or 0)["id"]

        tuned_path = MODELS_DIR / "notebook_xgboost_tuned.joblib"
        if tuned_path.exists():
            tuned_payload = joblib.load(tuned_path)
            if isinstance(tuned_payload, dict) and "model" in tuned_payload:
                tuned_model = tuned_payload["model"]
                selected = tuned_payload.get("feature_columns") or selected
            else:
                tuned_model = tuned_payload
            x_test = test_df[selected]
            y_test = test_df[TARGET_COLUMN]
            tuned_eval = evaluate_model(
                tuned_model,
                x_test,
                y_test,
                0.0,
                feature_names=selected,
            )
        else:
            # Fall back to best registry model metrics.
            trained = db.session.get(TrainedModel, best_id)
            tuned_eval = {
                "metrics": trained.metrics or {},
                "charts": {},
            }

        pipeline = {stage: "complete" for stage in [
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
        ]}

        return {
            "USER_ID": user.id,
            "DATASET_ID": dataset.id,
            "PROCESSED_ID": processed.id,
            "TRAIN_PATH": processed.train_file_path,
            "TEST_PATH": processed.test_file_path,
            "SMOTE_PATH": processed.train_smote_file_path,
            "SELECTED_FEATURES": selected,
            "BEST_MODEL_ID": best_id,
            "TUNED_EVAL": tuned_eval,
            "test_df": test_df,
            "row_count": dataset.row_count,
            "PIPELINE": pipeline,
            "comparison_charts": comparison.get("charts", {}),
            "MODELS_DIR": MODELS_DIR,
            "TARGET_COLUMN": TARGET_COLUMN,
            "app_config_models": Path(app.config["MODELS_FOLDER"]),
        }


def run_remaining(state: dict) -> None:
    from app.explainability.explanation_service import build_full_explanations, list_explainable_models
    from app.models.prediction_record import PredictionRecord
    from app.models.trained_model import TrainedModel
    from app.services.analytics_service import build_analytics_dashboard
    from app.services.prediction_service import predict_patient
    from app.services.eda_service import build_eda_charts

    saved = []

    # EDA visuals from a sample of the source CSV
    eda_sample_path = Path("/tmp/eda_sample_resume.csv")
    raw = pd.read_csv(DEFAULT_CSV, nrows=10_000)
    raw.to_csv(eda_sample_path, index=False)
    eda_result = build_eda_charts(eda_sample_path)
    saved += save_charts("eda", eda_result.get("charts", {}))

    # Comparison charts already computed
    saved += save_charts("comparison", state["comparison_charts"])
    saved += save_charts("tuned", state["TUNED_EVAL"].get("charts", {}))

    with flask_app_context():
        explainable = list_explainable_models(state["USER_ID"])
        interpret_model_id = state["BEST_MODEL_ID"] or (explainable[0].id if explainable else None)
        print("Using model", interpret_model_id)
        try:
            explanation = build_full_explanations(
                state["USER_ID"], interpret_model_id, test_row_index=0
            )
        except Exception as exc:
            print(f"Full SHAP/LIME failed ({exc}); falling back to global SHAP only.")
            from app.explainability.shap_service import build_global_explanations

            explanation = build_global_explanations(state["USER_ID"], interpret_model_id)
        saved += save_charts("explain", explanation.get("charts", {}))
        state["PIPELINE"]["Model Interpretation"] = "complete"

        trained = TrainedModel.query.filter_by(
            id=state["BEST_MODEL_ID"], user_id=state["USER_ID"]
        ).first()
        patient_row = state["test_df"].drop(columns=[TARGET_COLUMN]).iloc[0].to_dict()
        prediction = predict_patient(state["USER_ID"], state["BEST_MODEL_ID"], patient_row)
        print("Smoke prediction:", prediction)

        pipeline_state = {
            "dataset_id": state["DATASET_ID"],
            "processed_id": state["PROCESSED_ID"],
            "best_model_id": state["BEST_MODEL_ID"],
            "tuned_model_path": str(MODELS_DIR / "notebook_xgboost_tuned.joblib"),
            "selected_features": state["SELECTED_FEATURES"],
            "row_count": int(state["row_count"]),
            "metrics": state["TUNED_EVAL"].get("metrics", {}),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "figures": saved,
        }
        save_pipeline_state(pipeline_state)
        print("Pipeline state saved:", PIPELINE_STATE_PATH)
        state["PIPELINE"]["Model Deployment"] = "complete"

        pred_count = PredictionRecord.query.filter_by(user_id=state["USER_ID"]).count()
        dashboard = build_analytics_dashboard(state["USER_ID"])
        print("Logged predictions:", pred_count)
        saved += save_charts("analytics", dashboard.get("charts", {}))
        state["PIPELINE"]["Monitoring & Maintenance"] = "complete"
        state["PIPELINE"]["Model Retraining"] = "documented"

    # Embed chart outputs into an executed notebook copy for inspection.
    nb = nbformat.read(NOTEBOOK, as_version=4)
    summary_cell = nbformat.v4.new_code_cell(
        source=(
            "import json\n"
            f"print(json.dumps({json.dumps({k: v for k, v in load_pipeline_state().items() if k != 'selected_features'})}, indent=2))\n"
            f"print('figures_dir', r'{FIGURES_DIR}')\n"
        )
    )
    summary_cell.outputs = []
    nb.cells.append(summary_cell)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    # Only execute the appended summary cell by clearing earlier code cells.
    for cell in nb.cells[:-1]:
        if cell.cell_type == "code":
            cell.source = "pass  # skipped; artefacts restored by resume_from_interpretation.py\n"
    client.execute()
    nbformat.write(nb, EXECUTED)
    print("Wrote", EXECUTED)
    print(f"Saved {len(saved)} chart HTML files under {FIGURES_DIR}")


def main() -> None:
    pio.renderers.default = "json"
    state = bootstrap_globals()
    print(
        "Restored:",
        {
            "user_id": state["USER_ID"],
            "dataset_id": state["DATASET_ID"],
            "processed_id": state["PROCESSED_ID"],
            "best_model_id": state["BEST_MODEL_ID"],
            "rows": state["row_count"],
        },
    )
    run_remaining(state)


if __name__ == "__main__":
    main()
