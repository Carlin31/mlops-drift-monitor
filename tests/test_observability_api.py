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
