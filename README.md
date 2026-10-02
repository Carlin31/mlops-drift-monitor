# MLOps Drift Monitor

FastAPI service that serves a dummy scikit-learn classifier (Breast Cancer dataset)
and asynchronously monitors data drift on incoming prediction payloads using PSI
and the Kolmogorov-Smirnov test.

## Local development

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements-dev.txt
python scripts/generate_reference.py   # optional: lifespan generates these on first run too
PYTHONPATH=src uvicorn app.main:app --reload
```

## Tests

```bash
pytest -v
```

## Docker

```bash
docker compose up --build
```

## Endpoints

- `POST /predict` — body: 30 named float features (see `app.schemas.prediction.FEATURE_NAMES`). Returns `{"prediction": 0|1, "probability": [p0, p1]}`.
- `GET /observability/metrics` — latest PSI/KS drift report over the last `WINDOW_SIZE` requests (default 100), or `{"status": "insufficient_data"}` until `MIN_SAMPLES_FOR_DRIFT` (default 30) requests have arrived.
- `GET /health` — liveness + model-loaded check.

## Configuration

All settings in `app.core.config.Settings` are overridable via environment variables
(e.g. `WINDOW_SIZE=200`, `PSI_THRESHOLD=0.25`).
