# Machine learning notebook

All ML work runs in **one Jupyter notebook** covering the complete lifecycle from problem definition through retraining.

## Notebook

**`hospital_readmission_ml_pipeline.ipynb`**

Run all cells top-to-bottom.

## Lifecycle stages covered

1. Problem Definition  
2. Data Acquisition (UCI Diabetes 130-US, ≥200,000 rows)  
3. Data Understanding  
4. Data Cleaning & Preprocessing  
5. Exploratory Data Analysis (EDA)  
6. Feature Engineering  
7. Feature Selection  
8. Data Splitting  
9. Model Selection  
10. Model Training (5 models)  
11. Hyperparameter Tuning (RandomizedSearchCV on XGBoost)  
12. Model Evaluation  
13. Model Interpretation (SHAP & LIME)  
14. Model Deployment (Flask web app)  
15. Monitoring & Maintenance  
16. Model Retraining  

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=gabriel-ml --display-name "Python 3 (Gabriel ML)"
cd notebooks
jupyter lab
```

Open **`hospital_readmission_ml_pipeline.ipynb`**.

## Configuration

Edit `notebooks/ml_utils.py`:

```python
USER_EMAIL = "your@email.com"
MIN_DATASET_ROWS = 200_000
DEFAULT_TARGET_ROWS = 200_000
```

## Regenerate the notebook

```bash
python notebooks/generate_master_notebook.py
```

## Web app

After running the notebook, start the app:

```bash
python run.py
```

Log in at http://127.0.0.1:5000 — models, predictions, explainability, and analytics use notebook outputs.
