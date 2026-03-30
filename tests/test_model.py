from unittest.mock import MagicMock, patch

import pytest

from ml_service.model import Model, ModelData


def test_initial_state_is_empty():
    model = Model()
    data = model.get()
    assert data.model is None
    assert data.run_id is None


def test_get_returns_model_data_namedtuple():
    model = Model()
    assert isinstance(model.get(), ModelData)


def test_features_raises_when_model_is_none():
    model = Model()
    with pytest.raises(RuntimeError, match='Model is not loaded'):
        _ = model.features


def test_features_returns_list_when_model_is_loaded(fake_pipeline):
    model = Model()
    model.data = ModelData(model=fake_pipeline, run_id='run1')
    features = model.features
    assert isinstance(features, list)
    assert len(features) > 0


def test_set_updates_data(fake_pipeline):
    model = Model()
    with patch('ml_service.model.load_model', return_value=fake_pipeline):
        model.set(run_id='new-run')
    data = model.get()
    assert data.run_id == 'new-run'
    assert data.model is fake_pipeline
