from unittest.mock import MagicMock

import numpy as np
import pytest

import ml_service.app as app_module
from ml_service.features import FEATURE_COLUMNS
from ml_service.model import ModelData

FULL_REQUEST_DICT = {
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
def full_request_dict():
    return FULL_REQUEST_DICT.copy()


@pytest.fixture(scope='session')
def fake_pipeline():
    mock = MagicMock()
    mock.feature_names_in_ = np.array(FEATURE_COLUMNS)
    mock.predict_proba.return_value = np.array([[0.5, 0.5]])
    return mock


@pytest.fixture(autouse=True)
def reset_model():
    app_module.MODEL.data = ModelData(model=None, run_id=None)
    yield
    app_module.MODEL.data = ModelData(model=None, run_id=None)


@pytest.fixture(autouse=True)
def no_mlflow_env(monkeypatch):
    monkeypatch.delenv('MLFLOW_TRACKING_URI', raising=False)
    monkeypatch.delenv('DEFAULT_RUN_ID', raising=False)
