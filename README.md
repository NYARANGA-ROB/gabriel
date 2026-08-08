# 30-Day Hospital Readmission Risk Framework

A Flask-based web application for predicting and analysing 30-day hospital readmission risk using secondary healthcare data. Built as an MSc dissertation project, the framework supports dataset management, machine learning model training, patient-level prediction, explainable AI (SHAP/LIME), analytics, and role-based administration.

## Features

- **Dataset management** — CSV upload, validation, EDA, and preprocessing pipelines
- **Machine learning** — Train and evaluate classifiers (Logistic Regression, Random Forest, XGBoost, etc.)
- **Model comparison** — Side-by-side metrics, charts, and PDF reports
- **Patient prediction** — Individual readmission risk scores with clinical recommendations
- **Explainability** — Global and local SHAP/LIME explanations
- **Prediction history** — Searchable audit trail with CSV/PDF export
- **Analytics dashboard** — Population-level KPIs and trend charts
- **Admin panel** — User, dataset, model, and log management with RBAC

## Screenshots

| Dashboard | Patient prediction |
|-----------|-------------------|
| ![Dashboard placeholder](docs/screenshots/dashboard.png) | ![Prediction placeholder](docs/screenshots/prediction.png) |

| Model comparison | Explainability |
|------------------|----------------|
| ![Comparison placeholder](docs/screenshots/comparison.png) | ![Explainability placeholder](docs/screenshots/explainability.png) |

> Placeholder paths — add screenshots to `docs/screenshots/` after running the application.

## Requirements

- Python 3.12+
- Linux/macOS/Windows
- 4 GB RAM minimum (8 GB recommended for SHAP/LIME on larger datasets)

## Installation

### 1. Clone and enter the project

```bash
git clone <repository-url>
cd Gabriel
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set a strong `SECRET_KEY`. For production, also set `FLASK_ENV=production` and `AUTO_PROMOTE_ADMIN=false`.

### 5. Initialise the database

```bash
flask init-db
```

### 6. Run the application

```bash
python run.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Machine learning (Jupyter notebook)

All ML work runs in **one notebook**: `notebooks/hospital_readmission_ml_pipeline.ipynb` — covering problem definition through model retraining (≥200,000 rows).

```bash
pip install -r requirements.txt
python -m ipykernel install --user --name=gabriel-ml --display-name "Python 3 (Gabriel ML)"
cd notebooks
jupyter lab
```

Open **`hospital_readmission_ml_pipeline.ipynb`** and run all cells. See [`notebooks/README.md`](notebooks/README.md) for the full lifecycle.

Set your login email in `notebooks/ml_utils.py` (`USER_EMAIL`) before deployment cells.

### 7. Create an admin user (production)

```bash
flask promote-admin
```

In development, the first registered user is promoted automatically when `AUTO_PROMOTE_ADMIN=true`.

## Project Architecture

The application follows a layered **Flask factory** pattern with clear separation of concerns:

```
Gabriel/
├── run.py                 # Application entry point
├── config.py              # Environment-based configuration
├── app/
│   ├── __init__.py        # Application factory
│   ├── core/              # Logging, HTTP helpers, security utilities
│   ├── extensions.py      # Flask-SQLAlchemy, Flask-Login
│   ├── models/            # SQLAlchemy ORM entities
│   ├── repositories/      # Optimised data-access queries
│   ├── routes/            # HTTP blueprints (thin controllers)
│   ├── services/          # Business logic and PDF exports
│   ├── ml/                # Training, evaluation, model registry
│   ├── explainability/    # SHAP/LIME orchestration
│   ├── forms/             # WTForms validation
│   ├── utils/             # Decorators, dashboard navigation
│   ├── templates/         # Jinja2 views
│   └── static/            # CSS, JavaScript
├── uploads/               # User-uploaded and processed datasets
└── instance/              # SQLite database (default)
```

### Layer responsibilities

| Layer | Responsibility |
|-------|----------------|
| **Routes** | HTTP handling, authentication decorators, template rendering |
| **Services** | Domain logic, orchestration, exports |
| **Repositories** | Database queries with eager loading and shared filters |
| **Models** | Persistence schema and relationships |
| **ML / Explainability** | Training pipelines and XAI computations |

### Request flow

```
Browser → Blueprint (route) → Service → Repository / ML module → Database / Filesystem
```

### Key design decisions

- **Thin routes** — Shared dashboard context, pagination, and PDF responses live in `app/core/`
- **DRY prediction filters** — `PredictionFilters` dataclass used by user history and admin logs
- **Eager loading** — `ModelRepository` prevents N+1 queries on list views
- **Structured logging** — Configured at startup; exceptions logged with stack traces
- **Security** — Inactive users cannot access protected routes; file downloads validated against allowed roots; production requires explicit `SECRET_KEY`

## API Documentation

The application is primarily server-rendered. JSON endpoints support asynchronous UI polling.

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/auth/register` | Create account |
| GET/POST | `/auth/login` | Sign in |
| GET | `/auth/logout` | Sign out |
| GET/POST | `/auth/profile` | Update profile |

### Datasets

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/datasets/` | List and upload datasets |
| GET | `/datasets/<id>` | Dataset detail and preview |
| GET/POST | `/datasets/<id>/eda` | Exploratory data analysis |
| GET/POST | `/datasets/<id>/preprocess` | Run preprocessing pipeline |
| GET | `/datasets/<id>/preprocess/<pid>/download` | Download processed files |

### Machine learning

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/models/` | List trained models |
| GET/POST | `/models/train` | Start training job |
| GET | `/models/<id>` | Model detail and metrics |
| GET | `/models/<id>/status` | **JSON** training status poll |
| GET/POST | `/models/<id>/retrain` | Retrain from existing model |
| GET | `/models/compare` | Model comparison dashboard |
| GET | `/models/compare/pdf` | Comparison PDF export |

#### Training status response (`GET /models/<id>/status`)

```json
{
  "status": "training",
  "progress_percent": 45,
  "progress_log": [{"step": "fit", "message": "...", "percent": 45}],
  "metrics": null,
  "error_message": null
}
```

### Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/predictions/` | Patient prediction form |
| GET | `/predictions/history` | Prediction audit log |
| POST | `/predictions/history/<id>/delete` | Delete record |
| GET | `/predictions/history/export.csv` | CSV export |
| GET | `/predictions/history/export.pdf` | PDF export |

### Explainability & analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/explainability/` | SHAP/LIME dashboard |
| GET | `/explainability/pdf` | Explanation PDF |
| GET | `/analytics/` | Analytics dashboard |

### Admin (requires `admin` role)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/` | System statistics |
| GET | `/admin/users` | User management |
| GET | `/admin/datasets` | All datasets |
| GET | `/admin/predictions` | All prediction logs |
| GET | `/admin/models` | All trained models |
| GET | `/admin/reports` | Reports hub |
| GET | `/admin/reports/system.pdf` | System report PDF |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | dev key | Session signing key (**required in production**) |
| `DATABASE_URL` | SQLite | SQLAlchemy database URI |
| `ADMIN_EMAIL` | — | Email to auto-promote when no admin exists |
| `AUTO_PROMOTE_ADMIN` | `true` (dev) | Auto-promote first/admin user |
| `PREDICTION_EXPORT_MAX_ROWS` | `5000` | Maximum rows per history export |

## Deployment Guide

### Production checklist

1. Set `FLASK_ENV=production` and a strong `SECRET_KEY`
2. Set `AUTO_PROMOTE_ADMIN=false` and run `flask promote-admin`
3. Use **PostgreSQL** instead of SQLite (`DATABASE_URL=postgresql://...`)
4. Serve behind **Gunicorn** + Nginx with HTTPS
5. Move file storage to a persistent volume or object store
6. Replace background training threads with a task queue (Celery/RQ) for scalability

### Gunicorn example

```bash
pip install gunicorn
export FLASK_ENV=production
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
gunicorn -w 4 -b 0.0.0.0:8000 "run:app"
```

### Nginx reverse proxy (snippet)

```nginx
server {
    listen 80;
    server_name your-domain.example;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 32M;
    }
}
```

### Docker (outline)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
ENV FLASK_ENV=production
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "run:app"]
```

Mount volumes for `instance/`, `uploads/`, and `app/ml/models/`.

## Development

```bash
# Activate environment
source .venv/bin/activate

# Run with debug
FLASK_DEBUG=1 python run.py

# Initialise database
flask init-db

# Promote admin
flask promote-admin
```

## Licence

Academic dissertation project — see author for usage terms.
