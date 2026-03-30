import pytest
from fastapi.testclient import TestClient

import ml_service.app as app_module
from ml_service.app import app
from ml_service.model import ModelData

FULL_REQUEST = {
    'age': 39,
    'workclass': 'State-gov',
    'fnlwgt': 77516,
    'education': 'Bachelors',
    'education.num': 13,
    'marital.status': 'Never-married',
    'occupation': 'Adm-clerical',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'capital.gain': 2174,
    'capital.loss': 0,
    'hours.per.week': 40,
    'native.country': 'United-States',
}


@pytest.fixture
def live_client(fake_pipeline):
    app_module.MODEL.data = ModelData(model=fake_pipeline, run_id='integration-run')
    with TestClient(app) as c:
        yield c


def test_service_starts_and_health_is_ok(live_client):
    resp = live_client.get('/health')
    assert resp.status_code == 200
    assert resp.json()['status'] == 'ok'


def test_service_starts_without_model_when_env_vars_missing():
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get('/health')
    assert resp.status_code == 200


def test_full_prediction_returns_valid_response(live_client):
    resp = live_client.post('/predict', json=FULL_REQUEST)
    assert resp.status_code == 200
    body = resp.json()
    assert body['prediction'] in (0, 1)
    assert 0.0 <= body['probability'] <= 1.0
