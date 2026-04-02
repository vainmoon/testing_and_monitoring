import os

MODEL_ARTIFACT_PATH = 'model'


def tracking_uri() -> str:
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not tracking_uri:
        raise RuntimeError('Please set MLFLOW_TRACKING_URI')
    return tracking_uri


def default_run_id() -> str:
    """
    Returns model URI for startup.
    """

    default_run_id = os.getenv('DEFAULT_RUN_ID')
    if not default_run_id:
        raise RuntimeError('Set DEFAULT_RUN_ID to load model on startup')
    return default_run_id


def evidently_url() -> str:
    url = os.getenv('EVIDENTLY_URL')
    if not url:
        raise RuntimeError('Please set EVIDENTLY_URL')
    return url


def evidently_project_id() -> str:
    project_id = os.getenv('EVIDENTLY_PROJECT_ID')
    if not project_id:
        raise RuntimeError('Please set EVIDENTLY_PROJECT_ID')
    return project_id


def drift_report_interval() -> int:
    return int(os.getenv('DRIFT_REPORT_INTERVAL_SECONDS', '60'))


def drift_window_size() -> int:
    return int(os.getenv('DRIFT_WINDOW_SIZE', '100'))
