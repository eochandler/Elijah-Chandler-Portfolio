from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.turbofan import config


def load_serving_bundle():
    config_value = config.load_config()
    model_dir = Path(config_value['paths']['model_dir'])
    model_path = model_dir / 'champion_model.joblib'
    feature_columns_path = model_dir / 'feature_columns.json'
    sensor_path = model_dir / 'kept_sensors.json'
    cfg_path = model_dir / 'config.json'

    model = joblib.load(model_path)
    feature_cols = json.loads(feature_columns_path.read_text(encoding='utf-8'))
    sensors = json.loads(sensor_path.read_text(encoding='utf-8'))
    extra = json.loads(cfg_path.read_text(encoding='utf-8'))
    if model is None:
        raise ValueError(f'The serving model bundle at {model_dir} contains no model.')
    if not feature_cols or not sensors:
        raise ValueError(f'The serving model bundle at {model_dir} is missing feature metadata.')

    return {'model': model, 'feature_columns': feature_cols, 'kept_sensors': sensors, 'config': extra}
