# Diseño: MLOps Drift Monitor API

**Fecha:** 2026-10-02
**Estado:** Aprobado

## 1. Objetivo

Proyecto Python de referencia, estructurado para producción, que expone una API FastAPI con un modelo de clasificación (dummy, scikit-learn) y un módulo de observabilidad asíncrono que detecta data drift calculando el Population Stability Index (PSI) y la prueba de Kolmogorov-Smirnov (KS) sobre los payloads entrantes, comparados contra un dataset de referencia.

## 2. Alcance

Incluye: API de inferencia, módulo de cálculo de drift (PSI + KS), buffer de ventana deslizante, endpoint de métricas de observabilidad, tests unitarios/integración con pytest, Dockerfile y docker-compose.

Fuera de alcance: entrenamiento real/versionado de modelos (MLflow, etc.), persistencia en base de datos, autenticación, alerting externo, CI/CD.

## 3. Dataset y modelo

- Dataset de referencia: `sklearn.datasets.load_breast_cancer()` (30 features numéricas, clasificación binaria).
- El dataset de referencia (`X`, nombres de features) se persiste en `data/reference.csv` mediante `scripts/generate_reference.py`. Este script también entrena el modelo dummy (`LogisticRegression` o `RandomForestClassifier`, `random_state` fijo) y lo persiste en `models/model.pkl` vía `joblib`.
- En el arranque de la app (`lifespan`), si `models/model.pkl` o `data/reference.csv` no existen, se generan on-the-fly llamando a la misma lógica del script (fallback para entornos limpios, p.ej. contenedor recién construido sin artifacts pre-generados).
- Los nombres de las 30 features de `load_breast_cancer()` se normalizan a identificadores válidos de Python/Pydantic (p.ej. `mean radius` -> `mean_radius`).

## 4. Estructura de carpetas

```
mlops-drift-monitor/
├── src/app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app, lifespan, incluye routers
│   ├── api/
│   │   ├── __init__.py
│   │   ├── predict.py              # POST /predict
│   │   └── observability.py        # GET /observability/metrics, GET /health
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # Settings (pydantic-settings)
│   │   └── model.py                 # get_model(), load/train dummy classifier
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── prediction.py            # PredictionRequest / PredictionResponse
│   └── observability/
│       ├── __init__.py
│       ├── metrics.py               # compute_psi(), compute_ks(), build_drift_report()
│       ├── buffer.py                # DriftBuffer (deque + asyncio.Lock)
│       └── store.py                 # MetricsStore in-memory (singleton)
├── data/
│   └── reference.csv                # generado por scripts/generate_reference.py
├── models/
│   └── model.pkl                    # generado por scripts/generate_reference.py
├── scripts/
│   └── generate_reference.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # fixtures: TestClient, reset buffer/store
│   ├── test_predict.py
│   ├── test_metrics.py              # unitarios PSI/KS
│   └── test_observability_api.py    # integración end-to-end
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

## 5. API

### `POST /predict`
- **Request body**: objeto JSON con las 30 features nombradas (float), validado por `PredictionRequest` (Pydantic).
- **Response**: `{"prediction": 0|1, "probability": [p0, p1]}`.
- **Comportamiento**: valida input -> predice con el modelo cargado en memoria -> encola `BackgroundTasks.add_task(ingest_and_recompute, payload)` -> responde.
- **Errores**: 422 (validación Pydantic automática), 503 si el modelo no está cargado.

### `GET /observability/metrics`
- **Response** cuando hay datos suficientes:
```json
{
  "status": "ok",
  "window_size": 100,
  "samples_in_window": 87,
  "last_updated": "2026-10-02T12:00:00Z",
  "overall_drift_detected": false,
  "features": {
    "mean_radius": {"psi": 0.03, "ks_statistic": 0.08, "ks_pvalue": 0.42, "drift": false},
    "...": "..."
  }
}
```
- **Response** cuando `samples_in_window < MIN_SAMPLES_FOR_DRIFT`: `{"status": "insufficient_data", "samples_in_window": N, "min_required": 30}`.

### `GET /health`
- `{"status": "ok", "model_loaded": true}`.

## 6. Módulo de observabilidad (core)

- **`DriftBuffer`** (`observability/buffer.py`): envuelve `collections.deque(maxlen=settings.WINDOW_SIZE)`. Método async `append(sample: dict)` protegido con `asyncio.Lock`; método `snapshot() -> list[dict]` para copiar el estado actual sin bloquear la ingesta siguiente.
- **`metrics.py`** (funciones puras, testeables sin FastAPI):
  - `compute_psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float`: PSI por feature con bucketing por cuantiles del set de referencia, suavizado con epsilon para evitar `log(0)`/división por cero.
  - `compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]`: wrapper de `scipy.stats.ks_2samp`.
  - `build_drift_report(reference_df, current_samples, psi_threshold=0.2, ks_pvalue_threshold=0.05) -> dict`: itera las 30 features, agrega PSI/KS, marca `drift=True` por feature si `psi > psi_threshold` o `ks_pvalue < ks_pvalue_threshold`, y `overall_drift_detected` si alguna feature tiene drift.
- **`store.py`**: `MetricsStore` singleton in-memory con el último reporte + timestamp; `get()`/`set()` sin lock (un solo writer: la background task).
- **Tarea async** `ingest_and_recompute(payload)` en `api/predict.py` (o `observability/service.py`): agrega el payload al buffer; si `len(buffer) >= MIN_SAMPLES_FOR_DRIFT`, ejecuta `build_drift_report` dentro de `asyncio.to_thread(...)` (CPU-bound, no bloquea el event loop) y guarda el resultado en el store.

## 7. Configuración (`core/config.py`)

`Settings(BaseSettings)` con: `WINDOW_SIZE=100`, `MIN_SAMPLES_FOR_DRIFT=30`, `PSI_BUCKETS=10`, `PSI_THRESHOLD=0.2`, `KS_PVALUE_THRESHOLD=0.05`, `MODEL_PATH="models/model.pkl"`, `REFERENCE_DATA_PATH="data/reference.csv"`. Todas sobreescribibles por variables de entorno.

## 8. Manejo de errores

- Payload inválido -> 422 automático de Pydantic (FastAPI).
- Modelo no cargado al momento del request -> 503 con mensaje explícito.
- Buffer insuficiente -> no es un error, es un estado (`insufficient_data`) en `/observability/metrics`.
- PSI con bucket vacío -> epsilon (`1e-6`) sumado a las proporciones antes del logaritmo.

## 9. Testing (pytest)

- `test_metrics.py`: `compute_psi`/`compute_ks` con (a) misma distribución -> PSI≈0, KS p-value alto; (b) distribución desplazada (`loc` distinto) -> PSI > 0.2, KS p-value < 0.05.
- `test_predict.py`: request válido -> 200 y schema correcto; request con feature faltante o tipo incorrecto -> 422.
- `test_observability_api.py`: dispara N `POST /predict` (N >= `MIN_SAMPLES_FOR_DRIFT`) contra `TestClient` (las `BackgroundTasks` de Starlette se ejecutan de forma síncrona antes de devolver la respuesta al test, por lo que no se necesitan sleeps/polling) y verifica que `GET /observability/metrics` devuelve `status: "ok"` con reporte poblado; con menos requests, verifica `status: "insufficient_data"`.
- `conftest.py`: fixture que crea un `TestClient` nuevo y resetea `DriftBuffer`/`MetricsStore` antes de cada test para aislar estado entre tests.

## 10. Docker

- **Dockerfile**: base `python:3.11-slim`, usuario no-root, copia `requirements.txt` primero (cache de capas), instala dependencias, copia el resto del código, corre `scripts/generate_reference.py` en build (o se genera on-the-fly en el lifespan si falta), expone `8000`, `CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]`.
- **docker-compose.yml**: un servicio `api`, build desde el `Dockerfile` local, puerto `8000:8000`, `restart: unless-stopped`.
- **.dockerignore**: excluye `__pycache__`, `.git`, `tests/`, `.venv`, `docs/`.

## 11. Dependencias

- `requirements.txt`: `fastapi`, `uvicorn[standard]`, `scikit-learn`, `numpy`, `scipy`, `pydantic`, `pydantic-settings`, `joblib`, `pandas`.
- `requirements-dev.txt`: `-r requirements.txt`, `pytest`, `httpx` (requerido por `TestClient` en FastAPI/Starlette moderno).
