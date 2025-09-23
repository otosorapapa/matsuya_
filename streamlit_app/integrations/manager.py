"""High level helpers for orchestrating external data integrations."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

from .clients import DatasetBundle, PROVIDER_CLIENTS, get_client


@dataclass
class IntegrationResult:
    provider: str
    start_date: date
    end_date: date
    datasets: DatasetBundle
    message: str
    retrieved_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def row_counts(self) -> Dict[str, int]:
        return {name: len(df) for name, df in self.datasets.items()}

    def period_label(self) -> str:
        return f"{self.start_date:%Y-%m-%d} 〜 {self.end_date:%Y-%m-%d}"


def available_providers() -> List[str]:
    """Return the list of supported integration providers."""

    return list(PROVIDER_CLIENTS.keys())


def fetch_datasets(
    provider_label: str,
    start_date: date,
    end_date: date,
    credentials: Optional[Dict[str, str]] = None,
) -> IntegrationResult:
    """Fetch datasets from the given provider."""

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    client = get_client(provider_label)
    datasets, message = client.fetch(start_date, end_date, credentials)
    sanitized = {name: df.copy() for name, df in datasets.items()}
    return IntegrationResult(
        provider=client.provider_label,
        start_date=start_date,
        end_date=end_date,
        datasets=sanitized,
        message=message,
    )
