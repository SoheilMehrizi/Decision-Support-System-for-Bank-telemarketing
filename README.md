## Decision Support System for Bank Telemarketing Campaigns

Production-ready FastAPI service and ML pipelines that help campaign managers predict whether clients will subscribe to term deposits using the UCI Bank Marketing dataset. The system covers data ingestion, preprocessing, model training/evaluation, prediction, and knowledge extraction (global/local rules) with a PostgreSQL-backed data store.

### Key Features
- Data ingestion from CSV and database
- Cleaning and feature preprocessing
- Model training and selection (scikit-learn, MLflow-ready)
- Predictions via REST API
- Knowledge extraction: surrogate and local decision rules
- AuthN/AuthZ with JWT; role for superuser-only operations

### Tech Stack
- Python 3.10
- FastAPI + Uvicorn
- PostgreSQL + SQLAlchemy
- scikit-learn, imbalanced-learn, TensorFlow (optional), scipy, pandas
- Docker and Docker Compose

---

## Quickstart

### 1) Run with Docker Compose (recommended)
1. Copy env file and adjust if needed:
```bash
cp .env.example .env
```
2. Build and start services:
```bash
docker compose up --build
```
3. Create a superuser in the running app container:
```bash
docker exec -it fastapi python create_superuser.py
```
4. Open API docs: `http://localhost:8000/docs` (Swagger UI)

### 2) Run locally without Docker
Prerequisites: Python 3.10+, a running PostgreSQL, and a database URL.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip -r requirements.txt

# Set env (adjust to your DB)
export DATABASE_URL="postgresql+psycopg2://postgres:password@localhost:5432/mydb"

# Ensure Python can import the top-level src/ when running the app from deployment/app
cd deployment/app
export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Configuration

Environment variables used by the app:
- `DATABASE_URL` (required): SQLAlchemy URL to PostgreSQL (e.g. `postgresql+psycopg2://postgres:password@db:5432/mydb`).

Notes:
- JWT `SECRET_KEY` is currently defined in `deployment/app/utils/jwt.py`. For production, replace it with a strong secret (or refactor to load from environment).
- `.env` files are supported via `python-dotenv` and loaded by `deployment/app/database.py`.

See `.env.example` for a starting point.

---

## Project Structure

```
├── deployment/
│   ├── Dockerfile
│   └── app/
│       ├── main.py
│       ├── database.py
│       ├── models/
│       │   ├── users.py
│       │   └── bank_data.py
│       ├── repositories/
│       │   ├── users.py
│       │   └── bank_data_repository.py
│       ├── routers/
│       │   ├── auth.py
│       │   ├── bank_data.py
│       │   ├── prediction.py
│       │   └── knowledge_extraction.py
│       ├── schemas/
│       │   ├── users.py
│       │   ├── bank_data.py
│       │   ├── prediction.py
│       │   └── knowledge_extraction.py
│       └── utils/
│           └── jwt.py
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── knowledge_extraction.py
│   ├── models/
│   │   ├── model_repository.py
│   │   ├── model_selection.py
│   │   └── ...
│   └── pipelines/
│       ├── model_training_pipeline.py
│       └── prediction_pipeline.py
├── tests/
├── configs/
│   ├── config_repository.py
│   └── models_config.json
├── docs/
│   └── project_documentation.md
├── docker-compose.yml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## API Overview

Base URL: `http://localhost:8000`

### Auth
- `POST /auth/token` — obtain JWT via OAuth2 password flow (form fields: `username`, `password`).

Example:
```bash
curl -X POST \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d 'username=admin&password=secret' \
  http://localhost:8000/auth/token
```

Use the returned token as `Authorization: Bearer <token>` for protected routes.

### Bank data
- `POST /bank_data/` — create one or many bank data rows (superuser only).
- `POST /bank_data/slide-window/` — move 10% of test data to training (superuser only).

`BankDataCreate` fields: `age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome, y, training_data`.

Batch example:
```bash
curl -X POST http://localhost:8000/bank_data/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '[{"age":30,"job":"admin.","marital":"single","education":"university.degree","default":"no","balance":1000,"housing":"yes","loan":"no","contact":"cellular","day":10,"month":"may","duration":120,"campaign":2,"pdays":-1,"previous":0,"poutcome":"unknown","y":"no","training_data":true}]'
```

### ML
- `POST /ML/train-model/` — trigger model training pipeline (requires user auth).
- `POST /ML/predict/` — run predictions using the registered model.
- `POST /ML/general-knowledge/` — extract general or local rules (providing filters yields local rules).

Prediction example with explicit data:
```bash
curl -X POST http://localhost:8000/ML/predict/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "estimator": "Random_Forest",
    "data": {
      "age": 30, "job": "admin.", "marital": "single", "education": "university.degree",
      "default": "no", "balance": 1000, "housing": "yes", "loan": "no", "contact": "cellular",
      "day": 10, "month": "may", "duration": 120, "campaign": 2, "pdays": -1, "previous": 0,
      "poutcome": "unknown"
    }
  }'
```

If `data` is omitted, the service predicts on the DB test split.

---

## ML Pipelines

- Training: `src/pipelines/model_training_pipeline.py` loads data from DB, cleans, splits into X/y, infers categorical/numeric columns, and trains a model via `src/models/model_selection.py`.
- Prediction and knowledge extraction: `src/pipelines/prediction_pipeline.py` loads cleaned data and the registered model from `ModelRepository`, performs predictions, and extracts surrogate/local rules.

Tracking with MLflow is supported by dependencies, and can be enabled in model training utilities.

---

## Development

### Tests
Tests live under `tests/`. Run with:
```bash
pytest -q
```

### Contributing
See `CONTRIBUTING.md` for guidelines. Please follow Conventional Commits and open PRs with a clear description and checklist.

### Code of Conduct
See `CODE_OF_CONDUCT.md`.

---

## Dataset
This project uses the “Bank Marketing” dataset from the UCI Machine Learning Repository. The goal is to predict if a client will subscribe to a term deposit (`y ∈ {"yes","no"}`). Typical features include demographic, financial, and campaign-related attributes.

Refer to `docs/project_documentation.md` for a deeper dive and to the provided system diagram.

---

## License
Licensed under the terms of the `LICENSE` file.

