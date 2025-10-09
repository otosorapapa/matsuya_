"""Utilities for orchestrating scheduled and event-driven data synchronisation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .manager import IntegrationResult, fetch_datasets


@dataclass
class BatchJob:
    """Represents a periodic batch job that synchronises datasets via the API."""

    provider: str
    run_time: time = time(hour=5, minute=0)
    frequency: str = "daily"
    lookback_days: int = 1
    enabled: bool = True
    last_run: Optional[date] = None

    def is_due(self, *, now: Optional[datetime] = None) -> bool:
        """Return ``True`` when the batch job should execute."""

        if not self.enabled:
            return False
        moment = now or datetime.utcnow()
        if self.frequency != "daily":
            return False
        if self.last_run == moment.date():
            return False
        return moment.time() >= self.run_time

    def execution_window(self, *, reference: Optional[datetime] = None) -> tuple[date, date]:
        """Return the start/end date for the next fetch window."""

        moment = reference or datetime.utcnow()
        end = moment.date() - timedelta(days=1)
        start = end - timedelta(days=max(self.lookback_days - 1, 0))
        return start, end


@dataclass
class WebhookEvent:
    """Payload captured from a webhook callback."""

    provider: str
    payload: Dict[str, object]
    received_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RPAJob:
    """Definition of an automated CSV retrieval job via simple RPA."""

    name: str
    url: str
    dataset: str
    last_fetched: Optional[datetime] = None


class IntegrationSyncManager:
    """Stateful helper that encapsulates batch, webhook, and RPA ingestion flows."""

    def __init__(self) -> None:
        self.batch_jobs: Dict[str, BatchJob] = {}
        self.webhook_queue: List[WebhookEvent] = []
        self.rpa_jobs: Dict[str, RPAJob] = {}

    # ------------------------------------------------------------------
    # Batch scheduling
    # ------------------------------------------------------------------
    def enable_daily_batch(
        self,
        provider: str,
        *,
        run_time: time = time(hour=5, minute=0),
        lookback_days: int = 1,
    ) -> None:
        """Register or update a daily batch job."""

        job = self.batch_jobs.get(provider) or BatchJob(provider=provider)
        job.run_time = run_time
        job.lookback_days = max(1, int(lookback_days))
        job.frequency = "daily"
        job.enabled = True
        self.batch_jobs[provider] = job

    def disable_batch(self, provider: str) -> None:
        """Disable the batch job for ``provider`` if it exists."""

        job = self.batch_jobs.get(provider)
        if job:
            job.enabled = False

    def run_pending_batches(
        self,
        *,
        credentials: Dict[str, Dict[str, str]],
        now: Optional[datetime] = None,
    ) -> List[IntegrationResult]:
        """Execute all due batch jobs and return the resulting integrations."""

        executed: List[IntegrationResult] = []
        moment = now or datetime.utcnow()
        for provider, job in list(self.batch_jobs.items()):
            if not job.is_due(now=moment):
                continue
            start, end = job.execution_window(reference=moment)
            provider_credentials = credentials.get(provider, {})
            result = fetch_datasets(provider, start, end, provider_credentials)
            job.last_run = moment.date()
            executed.append(result)
        return executed

    # ------------------------------------------------------------------
    # Webhook processing
    # ------------------------------------------------------------------
    def queue_webhook(self, provider: str, payload: Dict[str, object]) -> None:
        """Store an inbound webhook payload for asynchronous processing."""

        self.webhook_queue.append(WebhookEvent(provider=provider, payload=payload))

    def process_webhooks(
        self,
        *,
        credentials: Dict[str, Dict[str, str]],
    ) -> List[IntegrationResult]:
        """Process queued webhook events and return executed integrations."""

        results: List[IntegrationResult] = []
        remaining: List[WebhookEvent] = []
        while self.webhook_queue:
            event = self.webhook_queue.pop(0)
            try:
                start_raw = event.payload.get("start_date")
                end_raw = event.payload.get("end_date")
                if start_raw and end_raw:
                    start = date.fromisoformat(str(start_raw))
                    end = date.fromisoformat(str(end_raw))
                else:
                    # When no explicit range is provided we fallback to yesterday.
                    today = event.received_at.date()
                    end = today
                    start = today
                provider_credentials = credentials.get(event.provider, {})
                result = fetch_datasets(event.provider, start, end, provider_credentials)
                results.append(result)
            except Exception:  # pragma: no cover - defensive branch
                remaining.append(event)
        self.webhook_queue = remaining
        return results

    # ------------------------------------------------------------------
    # RPA automation
    # ------------------------------------------------------------------
    def register_rpa_job(self, name: str, url: str, dataset: str) -> None:
        """Register a simple RPA workflow that downloads a CSV and feeds it into the app."""

        self.rpa_jobs[name] = RPAJob(name=name, url=url, dataset=dataset)

    def run_rpa_jobs(self) -> Dict[str, pd.DataFrame]:
        """Execute registered RPA CSV fetches and return parsed dataframes."""

        from io import StringIO

        import requests

        datasets: Dict[str, pd.DataFrame] = {}
        for job in self.rpa_jobs.values():
            try:
                response = requests.get(job.url, timeout=15)
                response.raise_for_status()
                buffer = StringIO(response.text)
                datasets[job.dataset] = pd.read_csv(buffer)
                job.last_fetched = datetime.utcnow()
            except Exception:  # pragma: no cover - network guard
                continue
        return datasets


def merge_results(results: Iterable[IntegrationResult]) -> Optional[IntegrationResult]:
    """Merge multiple integration results into a single consolidated object."""

    results = list(results)
    if not results:
        return None
    primary = results[0]
    merged_datasets: Dict[str, pd.DataFrame] = {k: v.copy() for k, v in primary.datasets.items()}
    message_parts = [primary.message]
    for result in results[1:]:
        message_parts.append(result.message)
        for name, df in result.datasets.items():
            if name in merged_datasets:
                merged_datasets[name] = pd.concat(
                    [merged_datasets[name], df], ignore_index=True, sort=False
                ).drop_duplicates()
            else:
                merged_datasets[name] = df.copy()
    return IntegrationResult(
        provider=primary.provider,
        start_date=min(result.start_date for result in results),
        end_date=max(result.end_date for result in results),
        datasets=merged_datasets,
        message=" / ".join(filter(None, message_parts)),
        retrieved_at=max(result.retrieved_at for result in results),
    )


__all__ = [
    "BatchJob",
    "IntegrationSyncManager",
    "WebhookEvent",
    "RPAJob",
    "merge_results",
]
