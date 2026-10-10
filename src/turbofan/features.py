from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import load_config

DEFAULT_SENSOR_DROPS = {1, 5, 6, 10, 16, 18, 19}


def _safe_column_list(columns: list[str]) -> list[str]:
    return [c for c in columns if c and not pd.isna(c)]


def select_kept_sensors(train_df: pd.DataFrame, drop_sensor_ids: set[int] | None = None) -> list[str]:
    """Select the FD001 sensors to keep after removing nearly constant ones."""
    drop_sensor_ids = drop_sensor_ids or DEFAULT_SENSOR_DROPS
    sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
    kept: list[str] = []
    for col in sensor_cols:
        sensor_id = int(col.split('_')[-1])
        if sensor_id in drop_sensor_ids:
            continue
        if train_df[col].nunique() <= 1:
            continue
        if train_df[col].var(ddof=0) <= 1e-6:
            continue
        kept.append(col)
    return kept


def engineered_features(
    df: pd.DataFrame,
    kept_sensors: list[str] | None = None,
    baseline_cycles: int = 10,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute lag-safe engineered features per engine."""
    if kept_sensors is None:
        kept_sensors = [c for c in df.columns if c.startswith('sensor_')]
    windows = windows or [5, 10, 30]
    out = df.copy().sort_values(['unit', 'cycle']).reset_index(drop=True)

    for col in kept_sensors:
        for window in windows:
            rolling_mean = out.groupby('unit')[col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
            out[f'{col}_roll_mean_{window}'] = rolling_mean.fillna(out[col])

            rolling_std = out.groupby('unit')[col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).std().fillna(0.0)
            )
            out[f'{col}_roll_std_{window}'] = rolling_std

        baseline = out.groupby('unit')[col].transform(
            lambda s: s.iloc[:baseline_cycles].mean() if len(s) >= baseline_cycles else s.mean()
        )
        out[f'{col}_baseline_diff'] = out[col] - baseline

        out[f'{col}_raw'] = out[col]

    out['cycle_number'] = out['cycle']
    return out


def feature_columns_for_config(feature_config: dict | None = None, kept_sensors: list[str] | None = None) -> list[str]:
    """Return the canonical feature order used by training and serving."""
    config = load_config() if feature_config is None else feature_config

    if kept_sensors is None:
        sensor_ids = config.get('keep_sensors')
        if sensor_ids:
            kept_sensors = [f'sensor_{sensor_id}' for sensor_id in sensor_ids]
        else:
            kept_sensors = [f'sensor_{sensor_id}' for sensor_id in range(1, 22) if sensor_id not in DEFAULT_SENSOR_DROPS]

    feature_names: list[str] = []
    for col in kept_sensors:
        feature_names.append(f'{col}_raw')
        for window in config.get('feature_windows', [5, 10, 30]):
            feature_names.append(f'{col}_roll_mean_{window}')
            feature_names.append(f'{col}_roll_std_{window}')
        feature_names.append(f'{col}_baseline_diff')
    feature_names.append('cycle_number')
    return feature_names


def persist_kept_sensors(sensors: list[str], path: str | Path | None = None) -> str:
    """Persist the selected sensor columns to disk for serving."""
    config = load_config() if path is None else None
    target = Path(path) if path else Path(config['paths']['processed_data_dir']) / 'kept_sensors.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as file:
        json.dump(sensors, file)
    return str(target)


def load_kept_sensors(path: str | Path | None = None) -> list[str]:
    """Load the selected sensor columns from disk."""
    config = load_config() if path is None else None
    target = Path(path) if path else Path(config['paths']['processed_data_dir']) / 'kept_sensors.json'
    if not target.exists():
        return []
    with target.open('r', encoding='utf-8') as file:
        data = json.load(file)
    return _safe_column_list(data)


def transform_cycle_history(cycle_rows: list[dict], engine_id: str | None = None, kept_sensors: list[str] | None = None) -> pd.DataFrame:
    """Convert a JSON cycle history into the same engineered feature matrix used in training."""
    if not cycle_rows:
        raise ValueError('At least one cycle is required to create features.')

    frame = pd.DataFrame(cycle_rows)
    frame['unit'] = engine_id or 'engine-1'
    frame = frame.sort_values('cycle').reset_index(drop=True)

    if kept_sensors is None:
        kept_sensors = [f'sensor_{sensor_id}' for sensor_id in range(1, 22) if sensor_id not in DEFAULT_SENSOR_DROPS]

    final = engineered_features(frame, kept_sensors=kept_sensors, baseline_cycles=10, windows=[5, 10, 30])
    return final
