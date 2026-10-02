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
