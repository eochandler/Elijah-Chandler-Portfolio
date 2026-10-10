from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
TRAIN_COLUMNS = ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3', *SENSOR_COLUMNS]


def load_raw_data(data_dir: str | Path, subset: str = 'FD001') -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load NASA C-MAPSS train, test, and RUL text files."""
    base = Path(data_dir)
    train_df = pd.read_csv(base / f'train_{subset}.txt', sep=r'\s+', header=None, engine='python')
    test_df = pd.read_csv(base / f'test_{subset}.txt', sep=r'\s+', header=None, engine='python')
    rul_series = pd.read_csv(base / f'RUL_{subset}.txt', sep=r'\s+', header=None, engine='python').iloc[:, 0]

    train_df = train_df.dropna(axis=1, how='all')
    test_df = test_df.dropna(axis=1, how='all')

    if train_df.shape[1] != 26:
        raise ValueError(f'Expected 26 columns in training data, got {train_df.shape[1]}')
    if test_df.shape[1] != 26:
        raise ValueError(f'Expected 26 columns in test data, got {test_df.shape[1]}')

    train_df.columns = TRAIN_COLUMNS
    test_df.columns = TRAIN_COLUMNS
    rul_series = rul_series.rename('rul')
    return train_df, test_df, rul_series


def add_rul_labels(train_df: pd.DataFrame, clip_rul: int = 125) -> pd.DataFrame:
    """Add clipped RUL labels for the training set."""
    train_df = train_df.copy()
    max_cycle_by_unit = train_df.groupby('unit')['cycle'].transform('max')
    train_df['rul'] = np.maximum(0, max_cycle_by_unit - train_df['cycle'])
    train_df['rul'] = train_df['rul'].clip(upper=clip_rul)
    return train_df


def prepare_dataset(data_dir: str | Path | None = None, subset: str = 'FD001', clip_rul: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load and prepare NASA FD001 data with clipped training labels."""
    config = load_config()
    if data_dir is None:
        data_dir = Path(config['paths']['raw_data_dir'])
    if clip_rul is None:
        clip_rul = int(config.get('clip_rul', 125))
    train_df, test_df, rul_series = load_raw_data(data_dir, subset=subset)
    train_df = add_rul_labels(train_df, clip_rul=clip_rul)
    return train_df, test_df, rul_series


def synthetic_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create a tiny synthetic fixture for tests without the NASA raw files."""
    rng = np.random.default_rng(42)
    rows = []
    for unit in range(1, 6):
        for cycle in range(1, 11):
            base = 600 + unit * 10
            vals = {
                'unit': unit,
                'cycle': cycle,
                'op_setting_1': 0.5 + cycle * 0.01,
                'op_setting_2': 0.8,
                'op_setting_3': 1.0,
            }
            for sensor_idx in range(1, 22):
                sensor_value = base + sensor_idx * 3 + rng.normal(0, 0.5) + cycle * 0.1
                vals[f'sensor_{sensor_idx}'] = sensor_value
            rows.append(vals)

    train_df = pd.DataFrame(rows)
    train_df['rul'] = train_df.groupby('unit')['cycle'].transform('max') - train_df['cycle']
    train_df['rul'] = train_df['rul'].clip(upper=125)

    test_df = train_df.copy()
    test_df['cycle'] = test_df['cycle'].clip(upper=8)
    test_df = test_df.sort_values(['unit', 'cycle']).drop_duplicates(subset=['unit'], keep='last')
    rul = pd.Series(np.linspace(60, 20, len(test_df)), index=test_df.index, name='rul')

    validation_df = train_df.sample(frac=0.3, random_state=42)
    return train_df, test_df, rul, validation_df
