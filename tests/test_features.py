import pytest

from ml_service.features import FEATURE_COLUMNS, to_dataframe
from ml_service.schemas import PredictRequest


def test_to_dataframe_returns_all_feature_columns_by_default(full_request_dict):
    req = PredictRequest.model_validate(full_request_dict)
    df = to_dataframe(req)
    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 1


def test_to_dataframe_filters_to_needed_columns(full_request_dict):
    req = PredictRequest.model_validate(full_request_dict)
    needed = ['age', 'education', 'sex']
    df = to_dataframe(req, needed_columns=needed)
    assert list(df.columns) == needed


def test_to_dataframe_ignores_unknown_needed_columns(full_request_dict):
    req = PredictRequest.model_validate(full_request_dict)
    df = to_dataframe(req, needed_columns=['age', 'nonexistent_column'])
    assert list(df.columns) == ['age']


def test_to_dataframe_preserves_values(full_request_dict):
    req = PredictRequest.model_validate(full_request_dict)
    df = to_dataframe(req, needed_columns=['age', 'sex'])
    assert df['age'].iloc[0] == full_request_dict['age']
    assert df['sex'].iloc[0] == full_request_dict['sex']


def test_to_dataframe_none_values_preserved():
    req = PredictRequest()
    df = to_dataframe(req, needed_columns=['age', 'sex'])
    assert df['age'].iloc[0] is None
    assert df['sex'].iloc[0] is None


def test_to_dataframe_empty_needed_columns(full_request_dict):
    req = PredictRequest.model_validate(full_request_dict)
    df = to_dataframe(req, needed_columns=[])
    assert df.shape == (1, 0)
