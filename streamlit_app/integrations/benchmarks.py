"""Helper functions for retrieving external benchmark datasets."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - graceful degradation
    requests = None  # type: ignore[assignment]


DEFAULT_BENCHMARKS = pd.DataFrame(
    [
        {
            "metric": "operating_margin",
            "industry_avg": 0.085,
            "top_quartile": 0.12,
            "unit": "ratio",
        },
        {
            "metric": "sales_growth",
            "industry_avg": 0.031,
            "top_quartile": 0.056,
            "unit": "ratio",
        },
        {
            "metric": "inventory_turnover",
            "industry_avg": 10.5,
            "top_quartile": 14.2,
            "unit": "times",
        },
    ]
)


def fetch_benchmark_indicators(
    *,
    api_url: str,
    industry: str,
    region: Optional[str] = None,
    metrics: Optional[Iterable[str]] = None,
    api_key: Optional[str] = None,
    timeout: float = 5.0,
) -> pd.DataFrame:
    """Fetch benchmark metrics from an external API with sensible fallbacks."""

    params: Dict[str, object] = {"industry": industry}
    if region:
        params["region"] = region
    if metrics:
        params["metrics"] = ",".join(metrics)

    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if not api_url:
        return DEFAULT_BENCHMARKS.copy()

    if requests is None:
        return DEFAULT_BENCHMARKS.copy()

    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return DEFAULT_BENCHMARKS.copy()

    data = payload.get("benchmarks") if isinstance(payload, dict) else None
    if not data:
        return DEFAULT_BENCHMARKS.copy()

    df = pd.DataFrame(data)
    if "metric" not in df.columns:
        return DEFAULT_BENCHMARKS.copy()
    return df


__all__ = ["DEFAULT_BENCHMARKS", "fetch_benchmark_indicators"]
