import asyncio
import logging
import os
import threading

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import RemoteWorkspace

from ml_service import config

logger = logging.getLogger(__name__)


class DriftMonitor:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffer: list[dict] = []

    def add(self, features: pd.DataFrame, prediction: int, probability: float) -> None:
        row = features.iloc[0].to_dict()
        row['prediction'] = prediction
        row['probability'] = probability
        with self._lock:
            self._buffer.append(row)

    def pop_current_data(self) -> pd.DataFrame:
        with self._lock:
            data = self._buffer.copy()
            self._buffer.clear()
        return pd.DataFrame(data) if data else pd.DataFrame()


def _load_reference_data(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        logger.warning('Reference data file not found: %s', path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.warning('Failed to load reference data from %s: %s', path, e)
        return None


async def run_drift_monitoring(monitor: DriftMonitor) -> None:

    try:
        evidently_url = config.evidently_url()
        project_id = config.evidently_project_id()
        reference_path = config.reference_data_path()
        interval = config.drift_report_interval()
        min_window = config.drift_window_size()
    except RuntimeError as e:
        logger.warning('Drift monitoring disabled: %s', e)
        return

    reference_data = _load_reference_data(reference_path)
    if reference_data is None:
        logger.warning('Drift monitoring disabled: no reference data available')
        return

    workspace = RemoteWorkspace(evidently_url)
    logger.info(
        'Drift monitoring started (interval=%ds, min_window=%d, project=%s)',
        interval,
        min_window,
        project_id,
    )

    while True:
        await asyncio.sleep(interval)
        current_data = monitor.pop_current_data()
        if len(current_data) < min_window:
            logger.debug(
                'Skipping drift report: only %d samples collected (min %d required)',
                len(current_data),
                min_window,
            )
            continue
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_data, current_data=current_data)
            workspace.add_run(project_id, report)
            logger.info('Drift report uploaded (%d samples)', len(current_data))
        except Exception as e:
            logger.exception('Failed to generate or upload drift report: %s', e)
