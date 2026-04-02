import asyncio
import logging
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


async def run_drift_monitoring(monitor: DriftMonitor) -> None:
    """
    Async coroutine that periodically builds an Evidently drift report
    from buffered predictions and uploads it to the RemoteWorkspace.

    The first full window of predictions is used as reference data.
    Subsequent windows are compared against it.

    Start at app launch with asyncio.ensure_future(run_drift_monitoring(monitor)).
    """
    try:
        evidently_url = config.evidently_url()
        project_id = config.evidently_project_id()
        interval = config.drift_report_interval()
        min_window = config.drift_window_size()
    except RuntimeError as e:
        logger.warning('Drift monitoring disabled: %s', e)
        return

    workspace = RemoteWorkspace(evidently_url)
    reference_data: pd.DataFrame | None = None
    logger.info('Drift monitoring started (interval=%ds, min_window=%d)', interval, min_window)

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

        if reference_data is None:
            reference_data = current_data
            logger.info(
                'Reference data initialised from first window (%d samples)', len(reference_data)
            )
            continue

        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_data, current_data=current_data)
            workspace.add_run(project_id, report)
            logger.info('Drift report uploaded (%d samples)', len(current_data))
        except Exception as e:
            logger.exception('Failed to generate or upload drift report: %s', e)
