import logging
import time
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from prometheus_fastapi_instrumentator import Instrumentator

from ml_service import config
from ml_service.features import to_dataframe
from ml_service.metrics import (
    INFERENCE_DURATION,
    MODEL_UPDATES_TOTAL,
    PREDICTION_PROBABILITY,
    PREDICTIONS_TOTAL,
    PREPROCESSING_DURATION,
    record_input_features,
    record_model_loaded,
)
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)

logger = logging.getLogger(__name__)

MODEL = Model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    Loads the initial model from MLflow on startup.
    """
    try:
        configure_mlflow()
        run_id = config.default_run_id()
        MODEL.set(run_id=run_id)
        pipeline = MODEL.get().model
        model_type = type(pipeline.steps[-1][1]).__name__ if hasattr(pipeline, 'steps') else type(pipeline).__name__
        record_model_loaded(run_id=run_id, features=MODEL.features, model_type=model_type)
        logger.info('Model loaded successfully on startup (run_id=%s)', run_id)
    except RuntimeError as e:
        logger.warning('Startup configuration error: %s. Service will run without a model.', e)
    except MlflowException as e:
        logger.warning('Failed to load initial model from MLflow: %s. Service will run without a model.', e)
    except Exception as e:
        logger.warning('Unexpected error during startup: %s. Service will run without a model.', e)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)

    Instrumentator().instrument(app).expose(app)

    @app.get('/health')
    def health() -> dict[str, Any]:
        model_state = MODEL.get()
        run_id = model_state.run_id
        return {'status': 'ok', 'run_id': run_id}

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        model_data = MODEL.get()
        if model_data.model is None:
            raise HTTPException(status_code=503, detail='Model is not loaded yet')

        try:
            features = MODEL.features
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        record_input_features(request)

        preprocess_start = time.perf_counter()
        try:
            df = to_dataframe(request, needed_columns=features)
        except Exception as e:
            logger.exception('Failed to build feature DataFrame')
            raise HTTPException(status_code=422, detail=f'Failed to build feature DataFrame: {e}')
        finally:
            PREPROCESSING_DURATION.observe(time.perf_counter() - preprocess_start)

        inference_start = time.perf_counter()
        try:
            probability = float(model_data.model.predict_proba(df)[0][1])
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f'Invalid feature values: {e}')
        except Exception as e:
            logger.exception('Prediction failed')
            raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')
        finally:
            INFERENCE_DURATION.observe(time.perf_counter() - inference_start)

        prediction = int(probability >= 0.5)
        PREDICTION_PROBABILITY.observe(probability)
        PREDICTIONS_TOTAL.labels(prediction=str(prediction)).inc()

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        old_run_id = MODEL.get().run_id
        try:
            MODEL.set(run_id=req.run_id)
        except MlflowException as e:
            raise HTTPException(
                status_code=404,
                detail=f'MLflow run not found or model artifact missing: {e}',
            )
        except Exception as e:
            logger.exception('Failed to update model with run_id=%s', req.run_id)
            raise HTTPException(status_code=503, detail=f'Failed to load model: {e}')

        MODEL_UPDATES_TOTAL.inc()
        try:
            pipeline = MODEL.get().model
            model_type = type(pipeline.steps[-1][1]).__name__ if hasattr(pipeline, 'steps') else type(pipeline).__name__
            record_model_loaded(run_id=req.run_id, features=MODEL.features, model_type=model_type, old_run_id=old_run_id)
        except Exception as e:
            logger.warning('Failed to record model info metrics: %s', e)
        logger.info('Model updated successfully (run_id=%s)', req.run_id)
        return UpdateModelResponse(run_id=req.run_id)

    return app


app = create_app()
