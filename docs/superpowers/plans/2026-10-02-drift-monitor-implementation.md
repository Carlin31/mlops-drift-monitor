# MLOps Drift Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-structured Python project exposing a FastAPI API that serves a dummy scikit-learn classifier and an asynchronous observability module that detects data drift (PSI + Kolmogorov-Smirnov) on incoming prediction payloads, with pytest coverage, a Dockerfile and docker-compose.yml.

**Architecture:** `src/app` holds the FastAPI app split into `api` (routers), `core` (settings + model/dataset artifacts), `schemas` (Pydantic I/O models), and `observability` (drift buffer, PSI/KS metrics, in-memory store). A `BackgroundTask` fired from `POST /predict` feeds a sliding-window buffer and recomputes drift metrics off the event loop via `asyncio.to_thread`; `GET /observability/metrics` reads the latest computed report.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, scikit-learn, NumPy, SciPy, pandas, Pydantic v2 / pydantic-settings, joblib, pytest, httpx, Docker.

**Spec:** `docs/superpowers/specs/2026-10-02-drift-monitor-design.md`

---

### Task 1: Project scaffolding and tooling config

**Files:**
- Create: `src/app/__init__.py`
- Create: `src/app/api/__init__.py`
- Create: `src/app/core/__init__.py`
- Create: `src/app/schemas/__init__.py`
- Create: `src/app/observability/__init__.py`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.gitignore`

- [ ] **Step 1: Create package directories and empty `__init__.py` files**

```bash
mkdir -p src/app/api src/app/core src/app/schemas src/app/observability tests scripts data models
touch src/app/__init__.py src/app/api/__init__.py src/app/core/__init__.py src/app/schemas/__init__.py src/app/observability/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml` with pytest config**

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `requirements.txt`**

```
fastapi
uvicorn[standard]
scikit-learn
numpy
scipy
pandas
pydantic
pydantic-settings
joblib
```

- [ ] **Step 4: Write `requirements-dev.txt`**

```
-r requirements.txt
pytest
httpx
```

- [ ] **Step 5: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/
.pytest_cache/
data/
models/
.env
```

- [ ] **Step 6: Create a virtualenv and install dev dependencies**

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements-dev.txt
```

Expected: install completes with no errors.

- [ ] **Step 7: Commit**

```bash
git add src tests scripts data models pyproject.toml requirements.txt requirements-dev.txt .gitignore
git commit -m "$(cat <<'EOF'
Scaffold project structure and dependency files

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

Note: `data/` and `models/` are gitignored (generated artifacts), but `git add` on an empty directory is a no-op — only tracked files under `src`/`tests`/`scripts` and the config files will actually be staged. That's expected.

---

### Task 2: Settings module

**Files:**
- Create: `src/app/core/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from app.core.config import Settings


def test_default_settings_values():
    settings = Settings()

    assert settings.window_size == 100
    assert settings.min_samples_for_drift == 30
    assert settings.psi_buckets == 10
    assert settings.psi_threshold == 0.2
    assert settings.ks_pvalue_threshold == 0.05
    assert settings.model_path == "models/model.pkl"
    assert settings.reference_data_path == "data/reference.csv"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.config'` (or similar import error).

- [ ] **Step 3: Write minimal implementation**

```python
# src/app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    window_size: int = 100
    min_samples_for_drift: int = 30
    psi_buckets: int = 10
    psi_threshold: float = 0.2
    ks_pvalue_threshold: float = 0.05
    model_path: str = "models/model.pkl"
    reference_data_path: str = "data/reference.csv"


settings = Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/app/core/config.py tests/test_config.py
git commit -m "$(cat <<'EOF'
Add Settings module for app configuration

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Model and dataset artifact module

**Files:**
- Create: `src/app/core/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model.py
import numpy as np

from app.core.model import (
    ensure_artifacts_exist,
    generate_artifacts,
    load_model,
    normalized_feature_names,
)


def test_normalized_feature_names_have_no_spaces():
    names = normalized_feature_names()

    assert len(names) == 30
    assert all(" " not in name for name in names)


def test_generate_artifacts_creates_model_and_reference_files(tmp_path):
    model_path = tmp_path / "model.pkl"
    reference_path = tmp_path / "reference.csv"

    generate_artifacts(model_path, reference_path)

    assert model_path.exists()
    assert reference_path.exists()

    model = load_model(model_path)
    prediction = model.predict(np.zeros((1, 30)))
    assert prediction.shape == (1,)


def test_ensure_artifacts_exist_skips_generation_when_present(tmp_path):
    model_path = tmp_path / "model.pkl"
    reference_path = tmp_path / "reference.csv"
    generate_artifacts(model_path, reference_path)
    original_mtime = model_path.stat().st_mtime

    ensure_artifacts_exist(model_path, reference_path)

    assert model_path.stat().st_mtime == original_mtime
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.model'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/app/core/model.py
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42


def normalized_feature_names() -> list[str]:
    data = load_breast_cancer()
    return [name.replace(" ", "_") for name in data.feature_names]


def generate_artifacts(model_path: Path, reference_path: Path) -> None:
    data = load_breast_cancer()
    feature_names = normalized_feature_names()
    reference_df = pd.DataFrame(data.data, columns=feature_names)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(data.data, data.target)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    reference_df.to_csv(reference_path, index=False)


def ensure_artifacts_exist(model_path: Path, reference_path: Path) -> None:
    if not model_path.exists() or not reference_path.exists():
        generate_artifacts(model_path, reference_path)


def load_model(model_path: Path):
    return joblib.load(model_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/app/core/model.py tests/test_model.py
git commit -m "$(cat <<'EOF'
Add model/dataset artifact generation and loading

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Reference-generation script

**Files:**
- Create: `scripts/generate_reference.py`

- [ ] **Step 1: Write the script**

```python
# scripts/generate_reference.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.core.config import settings
from app.core.model import generate_artifacts


def main() -> None:
    generate_artifacts(Path(settings.model_path), Path(settings.reference_data_path))
    print(f"Generated {settings.model_path} and {settings.reference_data_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and verify artifacts are created**

Run: `python scripts/generate_reference.py`
Expected: prints `Generated models/model.pkl and data/reference.csv`, and both files exist on disk (`ls models data`).

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_reference.py
git commit -m "$(cat <<'EOF'
Add CLI script to generate model and reference dataset artifacts

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Dynamic prediction schema

**Files:**
- Create: `src/app/schemas/prediction.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schemas.py
import pytest
from pydantic import ValidationError

from app.schemas.prediction import FEATURE_NAMES, PredictionRequest


def test_feature_names_match_count():
    assert len(FEATURE_NAMES) == 30


def test_valid_payload_parses_successfully():
    payload = {name: 1.0 for name in FEATURE_NAMES}

    request = PredictionRequest(**payload)

    assert getattr(request, FEATURE_NAMES[0]) == 1.0


def test_missing_feature_raises_validation_error():
    payload = {name: 1.0 for name in FEATURE_NAMES[1:]}

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.schemas.prediction'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/app/schemas/prediction.py
from pydantic import BaseModel, create_model

from app.core.model import normalized_feature_names

FEATURE_NAMES: list[str] = normalized_feature_names()

PredictionRequest = create_model(
    "PredictionRequest",
    **{name: (float, ...) for name in FEATURE_NAMES},
)


class PredictionResponse(BaseModel):
    prediction: int
    probability: list[float]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/app/schemas/prediction.py tests/test_schemas.py
git commit -m "$(cat <<'EOF'
Add dynamic Pydantic schema for prediction requests/responses

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Async drift buffer

**Files:**
- Create: `src/app/observability/buffer.py`
- Test: `tests/test_buffer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_buffer.py
import asyncio

from app.observability.buffer import DriftBuffer


def test_append_and_snapshot_returns_added_sample():
    buffer = DriftBuffer(maxlen=10)

    asyncio.run(buffer.append({"a": 1.0}))
    snapshot = asyncio.run(buffer.snapshot())

    assert snapshot == [{"a": 1.0}]
    assert len(buffer) == 1


def test_buffer_evicts_oldest_when_maxlen_exceeded():
    buffer = DriftBuffer(maxlen=2)

    async def fill():
        await buffer.append({"v": 1})
        await buffer.append({"v": 2})
        await buffer.append({"v": 3})

    asyncio.run(fill())
    snapshot = asyncio.run(buffer.snapshot())

    assert snapshot == [{"v": 2}, {"v": 3}]
    assert len(buffer) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_buffer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.observability.buffer'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/app/observability/buffer.py
import asyncio
from collections import deque


class DriftBuffer:
    def __init__(self, maxlen: int) -> None:
        self._buffer: deque[dict[str, float]] = deque(maxlen=maxlen)
        self._lock = asyncio.Lock()

    async def append(self, sample: dict[str, float]) -> None:
        async with self._lock:
            self._buffer.append(sample)

    async def snapshot(self) -> list[dict[str, float]]:
        async with self._lock:
            return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_buffer.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/app/observability/buffer.py tests/test_buffer.py
git commit -m "$(cat <<'EOF'
Add async sliding-window buffer for drift samples

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: PSI / Kolmogorov-Smirnov metrics

**Files:**
- Create: `src/app/observability/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import numpy as np

from app.observability.metrics import build_drift_report, compute_ks, compute_psi


def test_compute_psi_identical_distributions_is_near_zero():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=0, scale=1, size=1000)
    current = rng.normal(loc=0, scale=1, size=200)

    psi = compute_psi(reference, current, buckets=10)

    assert psi < 0.1


def test_compute_psi_shifted_distribution_exceeds_threshold():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=0, scale=1, size=1000)
    current = rng.normal(loc=5, scale=1, size=200)

    psi = compute_psi(reference, current, buckets=10)

    assert psi > 0.2


def test_compute_ks_identical_distributions_has_high_pvalue():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=0, scale=1, size=1000)
    current = rng.normal(loc=0, scale=1, size=200)

    _, pvalue = compute_ks(reference, current)

    assert pvalue > 0.05


def test_compute_ks_shifted_distribution_has_low_pvalue():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=0, scale=1, size=1000)
    current = rng.normal(loc=5, scale=1, size=200)

    _, pvalue = compute_ks(reference, current)

    assert pvalue < 0.05


def test_build_drift_report_contains_all_reference_columns():
    import pandas as pd

    reference_df = pd.DataFrame({"f1": np.linspace(0, 1, 100), "f2": np.linspace(10, 20, 100)})
    current_samples = [{"f1": 0.5, "f2": 15.0} for _ in range(40)]

    report = build_drift_report(
        reference_df,
        current_samples,
        psi_threshold=0.2,
        ks_pvalue_threshold=0.05,
        psi_buckets=10,
    )

    assert set(report["features"].keys()) == {"f1", "f2"}
    assert isinstance(report["overall_drift_detected"], bool)
    for feature_report in report["features"].values():
        assert {"psi", "ks_statistic", "ks_pvalue", "drift"} <= feature_report.keys()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.observability.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/app/observability/metrics.py
import numpy as np
from scipy.stats import ks_2samp

EPSILON = 1e-6


def compute_psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(np.quantile(reference, quantiles))
    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_props = ref_counts / len(reference) + EPSILON
    cur_props = cur_counts / len(current) + EPSILON

    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    result = ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def build_drift_report(
    reference_df,
    current_samples: list[dict[str, float]],
    psi_threshold: float,
    ks_pvalue_threshold: float,
    psi_buckets: int,
) -> dict:
    features_report = {}
    overall_drift = False

    for feature in reference_df.columns:
        reference_values = reference_df[feature].to_numpy()
        current_values = np.array([sample[feature] for sample in current_samples])

        psi = compute_psi(reference_values, current_values, buckets=psi_buckets)
        ks_statistic, ks_pvalue = compute_ks(reference_values, current_values)
        drift = psi > psi_threshold or ks_pvalue < ks_pvalue_threshold

        features_report[feature] = {
            "psi": round(psi, 4),
            "ks_statistic": round(ks_statistic, 4),
            "ks_pvalue": round(ks_pvalue, 4),
            "drift": drift,
        }
        overall_drift = overall_drift or drift

    return {"overall_drift_detected": overall_drift, "features": features_report}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/app/observability/metrics.py tests/test_metrics.py
git commit -m "$(cat <<'EOF'
Add PSI and Kolmogorov-Smirnov drift metric functions

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: In-memory metrics store

**Files:**
- Create: `src/app/observability/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_store.py
from app.observability.store import MetricsStore


def test_store_starts_empty():
    store = MetricsStore()

    report, last_updated = store.get()

    assert report is None
    assert last_updated is None


def test_set_and_get_returns_report_with_timestamp():
    store = MetricsStore()

    store.set({"overall_drift_detected": False, "features": {}})
    report, last_updated = store.get()

    assert report == {"overall_drift_detected": False, "features": {}}
    assert last_updated is not None


def test_reset_clears_report():
    store = MetricsStore()
    store.set({"overall_drift_detected": True, "features": {}})

    store.reset()
    report, last_updated = store.get()

    assert report is None
    assert last_updated is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.observability.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/app/observability/store.py
from datetime import datetime, timezone
from typing import Optional


class MetricsStore:
    def __init__(self) -> None:
        self._report: Optional[dict] = None
        self._last_updated: Optional[datetime] = None

    def set(self, report: dict) -> None:
        self._report = report
        self._last_updated = datetime.now(timezone.utc)

    def get(self) -> tuple[Optional[dict], Optional[datetime]]:
        return self._report, self._last_updated

    def reset(self) -> None:
        self._report = None
        self._last_updated = None


metrics_store = MetricsStore()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_store.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/app/observability/store.py tests/test_store.py
git commit -m "$(cat <<'EOF'
Add in-memory metrics store for latest drift report

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: FastAPI app, lifespan wiring, and health endpoint

**Files:**
- Create: `src/app/api/observability.py` (health route only in this task)
- Create: `src/app/main.py`
- Create: `tests/conftest.py`
- Test: `tests/test_health.py`

- [ ] **Step 1: Write `tests/conftest.py` (shared fixture, needed before the test below)**

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.observability.buffer import DriftBuffer
from app.observability.store import metrics_store


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        app.state.drift_buffer = DriftBuffer(maxlen=settings.window_size)
        metrics_store.reset()
        yield test_client
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_health.py
def test_health_reports_model_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_health.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 4: Write `src/app/api/observability.py` with the health route**

```python
# src/app/api/observability.py
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {"status": "ok", "model_loaded": model_loaded}
```

- [ ] **Step 5: Write `src/app/main.py`**

```python
# src/app/main.py
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from app.api import observability
from app.core.config import settings
from app.core.model import ensure_artifacts_exist, load_model
from app.observability.buffer import DriftBuffer


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(settings.model_path)
    reference_path = Path(settings.reference_data_path)

    ensure_artifacts_exist(model_path, reference_path)

    app.state.model = load_model(model_path)
    app.state.reference_df = pd.read_csv(reference_path)
    app.state.drift_buffer = DriftBuffer(maxlen=settings.window_size)

    yield


app = FastAPI(title="MLOps Drift Monitor", lifespan=lifespan)
app.include_router(observability.router)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_health.py -v`
Expected: PASS (1 passed)

- [ ] **Step 7: Commit**

```bash
git add src/app/main.py src/app/api/observability.py tests/conftest.py tests/test_health.py
git commit -m "$(cat <<'EOF'
Wire up FastAPI app with lifespan model loading and health endpoint

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Predict endpoint with background drift ingestion

**Files:**
- Create: `src/app/api/predict.py`
- Modify: `src/app/main.py` (register the new router)
- Test: `tests/test_predict.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_predict.py
from app.schemas.prediction import FEATURE_NAMES


def _valid_payload() -> dict:
    return {name: 1.0 for name in FEATURE_NAMES}


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert len(body["probability"]) == 2


def test_predict_rejects_missing_feature(client):
    payload = _valid_payload()
    del payload[FEATURE_NAMES[0]]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_non_numeric_feature(client):
    payload = _valid_payload()
    payload[FEATURE_NAMES[0]] = "not-a-number"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_predict.py -v`
Expected: FAIL with 404 (route `/predict` does not exist yet) on the first test.

- [ ] **Step 3: Write minimal implementation**

```python
# src/app/api/predict.py
import asyncio

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from app.core.config import settings
from app.observability.metrics import build_drift_report
from app.observability.store import metrics_store
from app.schemas.prediction import FEATURE_NAMES, PredictionRequest, PredictionResponse

router = APIRouter()


async def _ingest_and_recompute(request: Request, sample: dict[str, float]) -> None:
    buffer = request.app.state.drift_buffer
    await buffer.append(sample)

    if len(buffer) < settings.min_samples_for_drift:
        return

    current_samples = await buffer.snapshot()
    reference_df = request.app.state.reference_df

    report = await asyncio.to_thread(
        build_drift_report,
        reference_df,
        current_samples,
        settings.psi_threshold,
        settings.ks_pvalue_threshold,
        settings.psi_buckets,
    )
    report["samples_in_window"] = len(current_samples)
    metrics_store.set(report)


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: PredictionRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> PredictionResponse:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    features = [getattr(payload, name) for name in FEATURE_NAMES]
    feature_array = np.array([features])

    prediction = int(model.predict(feature_array)[0])
    probability = model.predict_proba(feature_array)[0].tolist()

    sample = dict(zip(FEATURE_NAMES, features))
    background_tasks.add_task(_ingest_and_recompute, request, sample)

    return PredictionResponse(prediction=prediction, probability=probability)
```

- [ ] **Step 4: Register the router in `src/app/main.py`**

Modify `src/app/main.py`:

```python
# src/app/main.py
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from app.api import observability, predict
from app.core.config import settings
from app.core.model import ensure_artifacts_exist, load_model
from app.observability.buffer import DriftBuffer


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(settings.model_path)
    reference_path = Path(settings.reference_data_path)

    ensure_artifacts_exist(model_path, reference_path)

    app.state.model = load_model(model_path)
    app.state.reference_df = pd.read_csv(reference_path)
    app.state.drift_buffer = DriftBuffer(maxlen=settings.window_size)

    yield


app = FastAPI(title="MLOps Drift Monitor", lifespan=lifespan)
app.include_router(observability.router)
app.include_router(predict.router)
```

(Only the import line and the final `app.include_router(predict.router)` are new; everything else is unchanged from Task 9.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_predict.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/app/api/predict.py src/app/main.py tests/test_predict.py
git commit -m "$(cat <<'EOF'
Add predict endpoint with background drift ingestion

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Observability metrics endpoint

**Files:**
- Modify: `src/app/api/observability.py`
- Test: `tests/test_observability_api.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_observability_api.py
from app.core.config import settings
from app.schemas.prediction import FEATURE_NAMES


def _valid_payload() -> dict:
    return {name: 1.0 for name in FEATURE_NAMES}


def test_metrics_reports_insufficient_data_before_min_samples(client):
    for _ in range(settings.min_samples_for_drift - 1):
        client.post("/predict", json=_valid_payload())

    response = client.get("/observability/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_data"
    assert body["samples_in_window"] == settings.min_samples_for_drift - 1


def test_metrics_reports_ok_after_min_samples(client):
    for _ in range(settings.min_samples_for_drift):
        client.post("/predict", json=_valid_payload())

    response = client.get("/observability/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["samples_in_window"] == settings.min_samples_for_drift
    assert "features" in body
    assert "overall_drift_detected" in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_observability_api.py -v`
Expected: FAIL with 404 (route `/observability/metrics` does not exist yet)

- [ ] **Step 3: Add the metrics route to `src/app/api/observability.py`**

```python
# src/app/api/observability.py
from fastapi import APIRouter, Request

from app.core.config import settings
from app.observability.store import metrics_store

router = APIRouter()


@router.get("/observability/metrics")
async def get_metrics(request: Request) -> dict:
    buffer = request.app.state.drift_buffer
    samples_in_window = len(buffer)

    if samples_in_window < settings.min_samples_for_drift:
        return {
            "status": "insufficient_data",
            "samples_in_window": samples_in_window,
            "min_required": settings.min_samples_for_drift,
        }

    report, last_updated = metrics_store.get()
    if report is None:
        return {
            "status": "insufficient_data",
            "samples_in_window": samples_in_window,
            "min_required": settings.min_samples_for_drift,
        }

    return {
        "status": "ok",
        "window_size": settings.window_size,
        "samples_in_window": report["samples_in_window"],
        "last_updated": last_updated.isoformat(),
        "overall_drift_detected": report["overall_drift_detected"],
        "features": report["features"],
    }


@router.get("/health")
async def health(request: Request) -> dict:
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {"status": "ok", "model_loaded": model_loaded}
```

(The `/health` route is unchanged from Task 9; `/observability/metrics` is the new addition, plus the new imports of `settings` and `metrics_store`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_observability_api.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/app/api/observability.py tests/test_observability_api.py
git commit -m "$(cat <<'EOF'
Add observability metrics endpoint exposing latest drift report

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Dockerfile, docker-compose.yml, and .dockerignore

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`

- [ ] **Step 1: Write `.dockerignore`**

```
__pycache__/
*.pyc
.git
.gitignore
.venv/
venv/
tests/
docs/
*.md
.pytest_cache/
data/
models/
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/

ENV PYTHONPATH=/app/src

RUN python scripts/generate_reference.py

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Write `docker-compose.yml`**

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    restart: unless-stopped
```

- [ ] **Step 4: Build and run the container, verify endpoints respond**

```bash
docker compose build
docker compose up -d
curl -s http://localhost:8000/health
```

Expected: `{"status":"ok","model_loaded":true}`

```bash
docker compose down
```

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "$(cat <<'EOF'
Add Dockerfile and docker-compose for isolated API deployment

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# MLOps Drift Monitor

FastAPI service that serves a dummy scikit-learn classifier (Breast Cancer dataset)
and asynchronously monitors data drift on incoming prediction payloads using PSI
and the Kolmogorov-Smirnov test.

## Local development

\`\`\`bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements-dev.txt
python scripts/generate_reference.py   # optional: lifespan generates these on first run too
PYTHONPATH=src uvicorn app.main:app --reload
\`\`\`

## Tests

\`\`\`bash
pytest -v
\`\`\`

## Docker

\`\`\`bash
docker compose up --build
\`\`\`

## Endpoints

- `POST /predict` — body: 30 named float features (see `app.schemas.prediction.FEATURE_NAMES`). Returns `{"prediction": 0|1, "probability": [p0, p1]}`.
- `GET /observability/metrics` — latest PSI/KS drift report over the last `WINDOW_SIZE` requests (default 100), or `{"status": "insufficient_data"}` until `MIN_SAMPLES_FOR_DRIFT` (default 30) requests have arrived.
- `GET /health` — liveness + model-loaded check.

## Configuration

All settings in `app.core.config.Settings` are overridable via environment variables
(e.g. `WINDOW_SIZE=200`, `PSI_THRESHOLD=0.25`).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
Add project README with setup and usage instructions

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass (config, model, schemas, buffer, metrics, store, health, predict, observability_api — no failures).

- [ ] **Step 2: Rebuild the Docker image end-to-end as a final sanity check**

```bash
docker compose build --no-cache
docker compose up -d
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{}'
```

Expected: HTTP 422 (empty body fails validation — confirms the container's app wiring and schema validation work end-to-end). Then:

```bash
docker compose down
```

- [ ] **Step 3: Confirm git log shows a clean, incremental history**

Run: `git log --oneline`
Expected: one commit per task, in order, with no uncommitted changes (`git status` clean).
