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
