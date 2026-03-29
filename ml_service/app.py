import logging
from typing import Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException

from ml_service import config
from ml_service.features import to_dataframe
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

        try:
            df = to_dataframe(request, needed_columns=features)
        except Exception as e:
            logger.exception('Failed to build feature DataFrame')
            raise HTTPException(status_code=422, detail=f'Failed to build feature DataFrame: {e}')

        try:
            probability = float(model_data.model.predict_proba(df)[0][1])
        except Exception as e:
            logger.exception('Prediction failed')
            raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')

        prediction = int(probability >= 0.5)
        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
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
        logger.info('Model updated successfully (run_id=%s)', req.run_id)
        return UpdateModelResponse(run_id=req.run_id)

    return app


app = create_app()
