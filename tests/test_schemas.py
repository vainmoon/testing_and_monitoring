import pytest
from pydantic import ValidationError

from ml_service.schemas import PredictRequest, UpdateModelRequest


def test_predict_request_full_valid(full_request_dict):
    req = PredictRequest.model_validate(full_request_dict)
    assert req.age == full_request_dict['age']
    assert req.education_num == full_request_dict['education.num']
    assert req.hours_per_week == full_request_dict['hours.per.week']
    assert req.capital_gain == full_request_dict['capital.gain']
    assert req.marital_status == full_request_dict['marital.status']


def test_predict_request_all_fields_optional():
    req = PredictRequest()
    assert req.age is None
    assert req.sex is None
    assert req.capital_gain is None


def test_predict_request_boundary_hours_168():
    req = PredictRequest.model_validate({'hours.per.week': 168})
    assert req.hours_per_week == 168


def test_predict_request_boundary_age_zero():
    req = PredictRequest(age=0)
    assert req.age == 0


def test_predict_request_negative_age():
    with pytest.raises(ValidationError):
        PredictRequest(age=-1)


def test_predict_request_negative_fnlwgt():
    with pytest.raises(ValidationError):
        PredictRequest(fnlwgt=-1)


def test_predict_request_negative_education_num():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({'education.num': -1})


def test_predict_request_negative_capital_gain():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({'capital.gain': -1})


def test_predict_request_negative_capital_loss():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({'capital.loss': -1})


def test_predict_request_hours_per_week_above_168():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({'hours.per.week': 169})


def test_predict_request_hours_per_week_negative():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({'hours.per.week': -1})


def test_predict_request_wrong_type_for_int_field():
    with pytest.raises(ValidationError):
        PredictRequest(age='not-a-number')


def test_update_model_request_valid():
    req = UpdateModelRequest(run_id='abc123')
    assert req.run_id == 'abc123'


def test_update_model_request_strips_whitespace():
    req = UpdateModelRequest(run_id='  abc123  ')
    assert req.run_id == 'abc123'


def test_update_model_request_empty_string():
    with pytest.raises(ValidationError):
        UpdateModelRequest(run_id='')


def test_update_model_request_whitespace_only():
    with pytest.raises(ValidationError):
        UpdateModelRequest(run_id='   ')
