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
