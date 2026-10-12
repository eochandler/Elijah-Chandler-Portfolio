from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow

from .config import load_config
from .features import feature_columns_for_config, load_kept_sensors


def export_champion() -> str:
    """Export the registered champion model and metadata to the models folder."""
    config = load_config()
    model_name = config['mlflow']['model_name']
    alias = config['mlflow']['alias']
    models_dir = Path(config['paths']['model_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(config['paths']['mlflow_db'])
    client = mlflow.MlflowClient()
    registered_model = client.get_registered_model(model_name)
    version = registered_model.aliases.get(alias)
    if version is None:
        ready_versions = [
            item for item in client.search_model_versions(f"name='{model_name}'")
            if item.status == 'READY'
        ]
        if not ready_versions:
            raise RuntimeError(f"No READY versions are registered for model '{model_name}'.")
        version = max(ready_versions, key=lambda item: int(item.version)).version
        client.set_registered_model_alias(model_name, alias, version)

    model_uri = f'models:/{model_name}/{version}'
    model = mlflow.lightgbm.load_model(model_uri=model_uri)
    model_version = client.get_model_version(model_name, version)
    validation_rmse = client.get_run(model_version.run_id).data.metrics.get('val_rmse')

    model_path = models_dir / 'champion_model.joblib'
    joblib.dump(model, model_path)

    with (models_dir / 'feature_columns.json').open('w', encoding='utf-8') as file:
        json.dump(feature_columns_for_config(config), file)

    with (models_dir / 'kept_sensors.json').open('w', encoding='utf-8') as file:
        json.dump(load_kept_sensors() or [f'sensor_{id}' for id in config['keep_sensors']], file)

    with (models_dir / 'config.json').open('w', encoding='utf-8') as file:
        json.dump({
            'clip_rul': config.get('clip_rul', 125),
            'min_cycles_required': config.get('min_cycles_required', 30),
            'alert_threshold': config.get('alert_threshold', 30),
            'validation_rmse': validation_rmse,
            'evaluation_split': 'One held-out fold from 3-fold GroupKFold split by engine.',
        }, file)
    return str(model_path)


if __name__ == '__main__':
    export_champion()
