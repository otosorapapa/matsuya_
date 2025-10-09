"""Integration helpers for external POS, accounting systems, and benchmarks."""

from .benchmarks import DEFAULT_BENCHMARKS, fetch_benchmark_indicators
from .manager import IntegrationResult, available_providers, fetch_datasets
from .sync import (
    BatchJob,
    IntegrationSyncManager,
    RPAJob,
    WebhookEvent,
    merge_results,
)

__all__ = [
    "DEFAULT_BENCHMARKS",
    "IntegrationResult",
    "available_providers",
    "fetch_benchmark_indicators",
    "fetch_datasets",
    "BatchJob",
    "IntegrationSyncManager",
    "WebhookEvent",
    "RPAJob",
    "merge_results",
]
