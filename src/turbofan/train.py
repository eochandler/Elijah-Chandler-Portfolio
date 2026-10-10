from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .config import load_config
from .data import prepare_dataset, synthetic_fixture
from .features import engineered_features, feature_columns_for_config, persist_kept_sensors, select_kept_sensors
from .metrics import rmse


def _git_commit_hash() -> str | None:
    """Return the current git commit hash when a repo is available."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _effective_min_child_samples(train_size: int, requested: int) -> int:
    """Keep LightGBM leaves valid even on small synthetic or offline fixtures."""
    if train_size <= 0:
        return max(2, requested)
    return max(2, min(requested, max(2, train_size // 5)))


def _log_plot(fig: plt.Figure, name: str) -> None:
    """Log a matplotlib figure as an MLflow artifact."""
    mlflow.log_figure(fig, name)
    plt.close(fig)


def raw_data_available(config: dict) -> bool:
    """Check whether the expected NASA raw files exist in the configured raw data directory."""
    raw_dir = Path(config['paths']['raw_data_dir'])
    required = {
        'train_FD001.txt',
        'test_FD001.txt',
        'RUL_FD001.txt',
    }
    if not raw_dir.exists():
        return False
    existing = {item.name for item in raw_dir.iterdir() if item.is_file()}
    return required.issubset(existing)


def train_and_evaluate() -> dict:
    """Train baseline, Ridge, and LightGBM models and log them to MLflow."""
    config = load_config()
    np.random.seed(config['random_seed'])

    if raw_data_available(config):
        train_df, _, _ = prepare_dataset()
    else:
        train_df, _, _, _ = synthetic_fixture()
    kept = select_kept_sensors(train_df)
    persist_kept_sensors(kept, Path(config['paths']['processed_data_dir']) / 'kept_sensors.json')

    X = engineered_features(
        train_df,
        kept_sensors=kept,
        baseline_cycles=config.get('baseline_cycles', 10),
        windows=config.get('feature_windows', [5, 10, 30]),
    )
    feature_columns = feature_columns_for_config(config, kept_sensors=kept)
    X = X[feature_columns]
    y = train_df['rul']

    gkf = GroupKFold(n_splits=3)
    train_idx, val_idx = next(gkf.split(X, y, groups=train_df['unit']))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    naive_predict = np.full(len(y_val), y_train.mean())
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_val)

    train_min_child_samples = _effective_min_child_samples(len(X_train), int(config.get('train', {}).get('min_child_samples', 20)))
    lgbm = LGBMRegressor(
        learning_rate=float(config.get('train', {}).get('learning_rate', 0.05)),
        num_leaves=int(config.get('train', {}).get('num_leaves', 31)),
        n_estimators=int(config.get('train', {}).get('n_estimators', 300)),
        min_child_samples=train_min_child_samples,
        random_state=config['random_seed'],
    )
    lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict(X_val)

    mlflow.set_tracking_uri(config['paths']['mlflow_db'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    commit_hash = _git_commit_hash()
    if commit_hash:
        mlflow.set_tag('git_commit', commit_hash)

    run_metadata = {
        'config.clip_rul': config.get('clip_rul', 125),
        'config.feature_windows': json.dumps(config.get('feature_windows', [5, 10, 30])),
        'config.dropped_sensors': json.dumps(sorted({1, 5, 6, 10, 16, 18, 19})),
        'data.subset': config.get('subset', 'FD001'),
    }

    with mlflow.start_run(run_name='baseline'):
        mlflow.log_params(run_metadata)
        mlflow.log_param('model', 'naive')
        mlflow.log_metric('val_rmse', rmse(y_val, naive_predict))
        fig, ax = plt.subplots()
        ax.scatter(y_val, naive_predict)
        ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red')
        ax.set_xlabel('True RUL')
        ax.set_ylabel('Predicted RUL')
        _log_plot(fig, 'predicted_vs_true.png')

    with mlflow.start_run(run_name='ridge'):
        mlflow.log_params(run_metadata)
        mlflow.log_param('model', 'ridge')
        mlflow.log_metric('val_rmse', rmse(y_val, ridge_pred))
        mlflow.sklearn.log_model(ridge, 'model')

    with mlflow.start_run(run_name='lightgbm'):
        mlflow.log_params(run_metadata)
        mlflow.log_param('model', 'lightgbm')
        mlflow.log_metric('val_rmse', rmse(y_val, lgbm_pred))
        mlflow.log_param('num_leaves', 31)
        mlflow.log_param('learning_rate', 0.05)
        mlflow.log_param('n_estimators', 300)
        mlflow.log_param('min_child_samples', 20)
        mlflow.lightgbm.log_model(lgbm, 'model', registered_model_name=config['mlflow']['model_name'])

        importance = lgbm.feature_importances_
        feature_names = feature_columns
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_names[:20], importance[:20])
        ax.set_title('Top LightGBM feature importances')
        _log_plot(fig, 'feature_importance.png')

    return {'feature_columns': feature_columns, 'kept_sensors': kept}


if __name__ == '__main__':
    train_and_evaluate()
