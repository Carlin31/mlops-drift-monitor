from datetime import datetime, timezone
from typing import Optional


class MetricsStore:
    def __init__(self) -> None:
        self._report: Optional[dict] = None
        self._last_updated: Optional[datetime] = None

    def set(self, report: dict) -> None:
        self._report = report
        self._last_updated = datetime.now(timezone.utc)

    def get(self) -> tuple[Optional[dict], Optional[datetime]]:
        return self._report, self._last_updated

    def reset(self) -> None:
        self._report = None
        self._last_updated = None


metrics_store = MetricsStore()
