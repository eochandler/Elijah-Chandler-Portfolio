from __future__ import annotations

import pandas as pd

from src.turbofan.data import synthetic_fixture
from src.turbofan.features import engineered_features, feature_columns_for_config, select_kept_sensors
from src.turbofan.train import train_and_evaluate


def test_feature_pipeline_has_expected_columns():
    train_df, _, _, _ = synthetic_fixture()
    kept = select_kept_sensors(train_df)
    features = engineered_features(train_df, kept, baseline_cycles=10, windows=[5, 10, 30])
    columns = feature_columns_for_config({'feature_windows': [5, 10, 30]})
    intersection = [c for c in columns if c in features.columns]
    assert len(intersection) >= 1
    assert 'cycle_number' in features.columns


def test_rolling_features_do_not_use_future_cycles():
    data = pd.DataFrame({
        'unit': [1, 1, 1, 2, 2, 2],
        'cycle': [1, 2, 3, 1, 2, 3],
        'sensor_2': [10, 20, 30, 40, 50, 60]
    })
    features = engineered_features(data, ['sensor_2'], baseline_cycles=2, windows=[2])
    assert features.loc[2, 'sensor_2_roll_mean_2'] == 15.0
    assert features.loc[0, 'sensor_2_roll_mean_2'] == 10.0


def test_training_pipeline_runs_on_synthetic_fixture():
    result = train_and_evaluate()
    assert 'feature_columns' in result
    assert 'kept_sensors' in result
    assert result['feature_columns']
