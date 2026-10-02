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
