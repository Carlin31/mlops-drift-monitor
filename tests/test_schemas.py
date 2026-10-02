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
