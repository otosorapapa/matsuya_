"""Integration helpers for external POS and accounting systems."""
from .manager import IntegrationResult, available_providers, fetch_datasets

__all__ = [
    "IntegrationResult",
    "available_providers",
    "fetch_datasets",
]
