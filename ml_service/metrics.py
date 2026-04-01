from prometheus_client import Counter, Gauge, Histogram, Info, Summary

PREPROCESSING_DURATION = Histogram(
    'preprocessing_duration_seconds',
    'Time spent building the feature DataFrame in to_dataframe()',
    buckets=(.0001, .0005, .001, .0025, .005, .01, .025, .05),
)


INPUT_FEATURE_VALUE = Summary(
    'input_feature_value',
    'Observed values of numeric input features',
    ['feature'],
)

INFERENCE_DURATION = Histogram(
    'model_inference_duration_seconds',
    'Time spent in model.predict_proba()',
    buckets=(.0001, .0005, .001, .0025, .005, .01, .025, .05, .1),
)

PREDICTION_PROBABILITY = Histogram(
    'model_prediction_probability',
    'Distribution of model output probability P(class=1)',
    buckets=(.1, .2, .3, .4, .5, .6, .7, .8, .9),
)

PREDICTIONS_TOTAL = Counter(
    'model_predictions_total',
    'Total predictions made, labelled by predicted class',
    ['prediction'],
)

MODEL_UPDATES_TOTAL = Counter(
    'model_updates_total',
    'Total number of successful model updates via /updateModel',
)

MODEL_CURRENT_INFO = Gauge(
    'model_current_info',
    'Indicates the active model run (1 = currently loaded)',
    ['run_id'],
)

MODEL_FEATURES_TOTAL = Gauge(
    'model_features_total',
    'Number of input features required by the currently loaded model',
)

MODEL_FEATURE_REQUIRED = Gauge(
    'model_feature_required',
    'Feature is required by the currently loaded model (1 = required)',
    ['feature'],
)

MODEL_INFO = Info(
    'model',
    'Metadata of the currently loaded model: run_id, type, required features',
)

_NUMERIC_FEATURES = (
    'age',
    'fnlwgt',
    'education.num',
    'capital.gain',
    'capital.loss',
    'hours.per.week',
)


def record_model_loaded(
    run_id: str,
    features: list[str],
    model_type: str | None = None,
    old_run_id: str | None = None,
) -> None:
    if old_run_id and old_run_id != run_id:
        MODEL_CURRENT_INFO.labels(run_id=old_run_id).set(0)
    MODEL_CURRENT_INFO.labels(run_id=run_id).set(1)
    MODEL_FEATURES_TOTAL.set(len(features))
    for feature in features:
        MODEL_FEATURE_REQUIRED.labels(feature=feature).set(1)
    MODEL_INFO.info({
        'run_id': run_id,
        'type': model_type or 'unknown',
        'features': ','.join(features),
    })


def record_input_features(request) -> None:
    for feature in _NUMERIC_FEATURES:
        value = getattr(request, feature.replace('.', '_'), None)
        if value is not None:
            INPUT_FEATURE_VALUE.labels(feature=feature).observe(value)
