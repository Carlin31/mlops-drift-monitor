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
