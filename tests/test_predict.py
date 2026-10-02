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
