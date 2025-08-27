## Project Documentation

### Overview
This system provides an end-to-end ML decision support workflow for bank telemarketing campaigns, exposing a FastAPI service for data operations, predictions, and knowledge extraction. Data is stored in PostgreSQL and accessed through SQLAlchemy repositories.

### Architecture
- FastAPI app (`deployment/app`) encapsulates routers, schemas, repositories, and models.
- ML pipelines live under `src/pipelines` and use utilities from `src/models` and preprocessing steps in `src/data_preprocessing.py`.
- Data access layer is implemented in `deployment/app/repositories` and `deployment/app/models`.
- `docker-compose.yml` runs both the API and PostgreSQL for local development.

### Data Model
- `users` table (see `deployment/app/models/users.py`)
- `bank_data` table (see `deployment/app/models/bank_data.py`) with domain constraints (e.g., `valid_age`, `valid_day`).

### Security
- OAuth2 password flow to obtain JWT via `POST /auth/token`.
- `Bearer` token is required for protected endpoints; superuser is required for administrative data operations.
- Replace the static `SECRET_KEY` in `deployment/app/utils/jwt.py` in production or externalize it into the environment.

### Pipelines
- Training (`src/pipelines/model_training_pipeline.py`):
  1) Load train/test data from DB via `src/data_ingestion.load_data_from_db`
  2) Clean datasets with `src.data_preprocessing.cleaning_pipeline_step`
  3) Split features/target and infer column types
  4) Train and evaluate model via `src/models/model_selection.train_log_compare_models`

- Prediction & Knowledge Extraction (`src/pipelines/prediction_pipeline.py`):
  - `predict(X, estimator)` runs inference with the registered model
  - `extract_general_rules_pipeline` and `extract_local_rules_pipeline` leverage surrogate/local rules to explain predictions

### Endpoints
- Auth: `POST /auth/token`
- Bank data: `POST /bank_data/`, `POST /bank_data/slide-window/`
- ML: `POST /ML/train-model/`, `POST /ML/predict/`, `POST /ML/general-knowledge/`

### Environment
- Required: `DATABASE_URL` (SQLAlchemy database URL)
- Optional: refactor to support `SECRET_KEY`, `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`

### Local Development Tips
- Ensure `PYTHONPATH` includes the repository root when running the app from `deployment/app`.
- Use `uvicorn main:app --reload` for local dev.
- Run tests with `pytest`.

### Operations
- Creating a superuser: `docker exec -it fastapi python create_superuser.py`
- Sliding window: `POST /bank_data/slide-window/` moves 10% of test rows to training and flips the same number from training to test.

### Future Improvements
- Externalize JWT secrets and settings to environment variables.
- Add migrations (Alembic) and seed scripts.
- Add CI for tests and linting.
- Expand MLflow tracking integration and model registry.

