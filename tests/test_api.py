from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_root_redirects_to_api_docs(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.url.path == '/docs'


def test_model_info_reports_validation_metric_not_test_metric(client):
    response = client.get('/model-info')
    assert response.status_code == 200
    body = response.json()
    assert body['validation_rmse'] is not None
    assert body['evaluation_split'] == 'One held-out fold from 3-fold GroupKFold split by engine.'
    assert 'fd001_test_rmse' not in body


def test_valid_prediction(client):
    payload = {'engine_id': 'eng-1', 'cycles': [{'cycle': i, 'sensor_2': float(640 + i), 'sensor_3': 1589.7, 'sensor_4': 1410.0, 'sensor_7': 100.0, 'sensor_8': 100.0, 'sensor_9': 84.0, 'sensor_11': 0.1, 'sensor_12': 0.5, 'sensor_13': 0.7, 'sensor_14': 0.3, 'sensor_15': 0.2, 'sensor_17': 0.9, 'sensor_20': 1.0, 'sensor_21': 4.0} for i in range(1, 31)]}
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    assert 'predicted_rul' in response.json()


def test_too_few_cycles_returns_422(client):
    payload = {'engine_id': 'eng-1', 'cycles': [{'cycle': i, 'sensor_2': 1.0, 'sensor_3': 2.0} for i in range(1, 10)]}
    response = client.post('/predict', json=payload)
    assert response.status_code == 422


def test_missing_sensor_returns_422(client):
    payload = {'engine_id': 'eng-1', 'cycles': [{'cycle': i, 'sensor_2': float(640 + i)} for i in range(1, 31)]}
    response = client.post('/predict', json=payload)
    assert response.status_code == 422
