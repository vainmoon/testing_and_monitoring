from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException

import ml_service.app as app_module
from ml_service.app import app
from ml_service.model import ModelData


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def client_with_model(fake_pipeline):
    app_module.MODEL.data = ModelData(model=fake_pipeline, run_id='test-run')
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_predict_without_model_returns_503(client, full_request_dict):
    resp = client.post('/predict', json=full_request_dict)
    assert resp.status_code == 503


def test_predict_invalid_type_for_age_returns_422(client_with_model):
    resp = client_with_model.post('/predict', json={'age': 'not-a-number'})
    assert resp.status_code == 422


def test_predict_negative_age_returns_422(client_with_model):
    resp = client_with_model.post('/predict', json={'age': -1})
    assert resp.status_code == 422


def test_predict_negative_fnlwgt_returns_422(client_with_model):
    resp = client_with_model.post('/predict', json={'fnlwgt': -100})
    assert resp.status_code == 422


def test_predict_hours_per_week_over_168_returns_422(client_with_model):
    resp = client_with_model.post('/predict', json={'hours.per.week': 200})
    assert resp.status_code == 422


def test_predict_negative_capital_gain_returns_422(client_with_model):
    resp = client_with_model.post('/predict', json={'capital.gain': -1})
    assert resp.status_code == 422


def test_predict_returns_200(client_with_model, full_request_dict):
    resp = client_with_model.post('/predict', json=full_request_dict)
    assert resp.status_code == 200


def test_predict_response_has_prediction_and_probability(client_with_model, full_request_dict):
    body = client_with_model.post('/predict', json=full_request_dict).json()
    assert 'prediction' in body
    assert 'probability' in body


def test_predict_prediction_is_binary(client_with_model, full_request_dict):
    body = client_with_model.post('/predict', json=full_request_dict).json()
    assert body['prediction'] in (0, 1)


def test_predict_probability_in_unit_interval(client_with_model, full_request_dict):
    body = client_with_model.post('/predict', json=full_request_dict).json()
    assert 0.0 <= body['probability'] <= 1.0


def test_update_model_empty_run_id_returns_422(client_with_model):
    resp = client_with_model.post('/updateModel', json={'run_id': ''})
    assert resp.status_code == 422


def test_update_model_whitespace_run_id_returns_422(client_with_model):
    resp = client_with_model.post('/updateModel', json={'run_id': '   '})
    assert resp.status_code == 422


def test_update_model_missing_run_id_returns_422(client_with_model):
    resp = client_with_model.post('/updateModel', json={})
    assert resp.status_code == 422


def test_update_model_invalid_run_id_returns_404(client_with_model):
    with patch.object(app_module.MODEL, 'set', side_effect=MlflowException('not found')):
        resp = client_with_model.post('/updateModel', json={'run_id': 'bad-id'})
    assert resp.status_code == 404


def test_update_model_connection_error_returns_503(client_with_model):
    with patch.object(app_module.MODEL, 'set', side_effect=ConnectionError('timeout')):
        resp = client_with_model.post('/updateModel', json={'run_id': 'some-id'})
    assert resp.status_code == 503


def test_update_model_success_returns_run_id(client_with_model):
    with patch.object(app_module.MODEL, 'set'):
        resp = client_with_model.post('/updateModel', json={'run_id': 'new-run-123'})
    assert resp.status_code == 200
    assert resp.json()['run_id'] == 'new-run-123'
