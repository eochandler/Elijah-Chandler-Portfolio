from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from src.turbofan.config import load_config
from src.turbofan.features import feature_columns_for_config, transform_cycle_history

from .model_loader import load_serving_bundle
from .schemas import PredictRequest, PredictResponse

logger = logging.getLogger('turbofan-api')
logging.basicConfig(level=logging.INFO)
MODEL_BUNDLE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_BUNDLE
    MODEL_BUNDLE = load_serving_bundle()
    yield


app = FastAPI(title='Turbofan RUL API', lifespan=lifespan)


@app.get('/', include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url='/docs')


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/model-info')
def model_info() -> dict[str, object]:
    config = load_config()
    return {
        'model_name': config['mlflow']['model_name'],
        'version': '1',
        'training_date': 'unknown',
        'validation_rmse': MODEL_BUNDLE.get('config', {}).get('validation_rmse'),
        'evaluation_split': MODEL_BUNDLE.get('config', {}).get('evaluation_split'),
        'expected_input_columns': ['cycle', *[f'sensor_{sensor_id}' for sensor_id in config['keep_sensors']]],
    }


@app.post('/predict', response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    config = load_config()
    min_cycles = int(config.get('min_cycles_required', 30))
    if len(payload.cycles) < min_cycles:
        raise HTTPException(status_code=422, detail=f'At least {min_cycles} cycles of history are required for a prediction.')

    required_fields = {f'sensor_{sensor_id}' for sensor_id in config.get('keep_sensors', [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21])}
    required_fields.add('cycle')
    for cycle in payload.cycles:
        cycle_data = cycle.model_dump()
        missing = sorted(required_fields - set(cycle_data.keys()))
        if missing:
            raise HTTPException(status_code=422, detail=f'Missing required sensor values: {missing}')

    start = time.perf_counter()
    kept_sensors = MODEL_BUNDLE.get('kept_sensors') or list(required_fields - {'cycle'})
    feature_frame = transform_cycle_history([cycle.model_dump() for cycle in payload.cycles], engine_id=payload.engine_id, kept_sensors=kept_sensors)
    feature_columns = MODEL_BUNDLE.get('feature_columns') or feature_columns_for_config(config, kept_sensors=kept_sensors)
    feature_matrix = feature_frame[feature_columns].copy()

    model = MODEL_BUNDLE.get('model')
    if model is None:
        raise HTTPException(status_code=503, detail='The prediction model is unavailable.')
    prediction = float(model.predict(feature_matrix)[-1])

    alert = bool(prediction < float(config.get('alert_threshold', 30)))
    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    logger.info('request_prediction engine_id=%s cycles=%s prediction=%s latency_ms=%s', payload.engine_id, len(payload.cycles), prediction, latency_ms)
    return PredictResponse(predicted_rul=prediction, alert=alert, model_version='1')
