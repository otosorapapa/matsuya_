"""Client abstractions for connecting to external systems."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

from streamlit_app import data_loader


logger = logging.getLogger(__name__)


DatasetBundle = Dict[str, pd.DataFrame]


@dataclass
class BaseIntegrationClient:
    """Base class for API integrations."""

    provider_label: str

    def fetch(
        self,
        start_date: date,
        end_date: date,
        credentials: Optional[Dict[str, str]] = None,
    ) -> Tuple[DatasetBundle, str]:
        """Return datasets filtered for the requested period and a status message."""

        raise NotImplementedError

    def _base_datasets(self) -> DatasetBundle:
        return {
            "sales": data_loader.load_sales_data(),
            "inventory": data_loader.load_inventory_data(),
            "fixed_costs": data_loader.load_fixed_costs(),
        }

    @staticmethod
    def _filter_sales_range(
        df: pd.DataFrame, start_date: date, end_date: date
    ) -> pd.DataFrame:
        if df.empty:
            return df.head(0)
        mask = (df["date"] >= pd.Timestamp(start_date)) & (
            df["date"] <= pd.Timestamp(end_date)
        )
        filtered = df.loc[mask].copy()
        return filtered


class RESTIntegrationClient(BaseIntegrationClient):
    """Base client for REST API integrations with graceful CSV fallbacks."""

    dataset_endpoints: Dict[str, str]

    def __init__(self, provider_label: str, dataset_endpoints: Dict[str, str]):
        super().__init__(provider_label=provider_label)
        self.dataset_endpoints = dataset_endpoints

    def _request_dataset(
        self,
        endpoint: str,
        start_date: date,
        end_date: date,
        credentials: Optional[Dict[str, str]],
    ) -> Optional[pd.DataFrame]:
        """Attempt to retrieve a dataset via REST. Returns ``None`` on failure."""

        if not endpoint:
            return None
        base_url = (credentials or {}).get("base_url")
        if not base_url:
            return None
        url = endpoint
        if not url.startswith("http"):
            url = base_url.rstrip("/") + "/" + endpoint.lstrip("/")
        headers = {}
        token = (credentials or {}).get("api_key")
        secret = (credentials or {}).get("api_secret")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if secret and "Authorization" not in headers:
            headers["X-API-SECRET"] = secret
        params = {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and "data" in payload:
                payload = payload.get("data")
            frame = pd.DataFrame(payload)
            return frame
        except Exception as exc:  # pragma: no cover - network guard
            logger.warning("Failed to fetch %s from %s: %s", self.provider_label, url, exc)
        return None

    def fetch(
        self,
        start_date: date,
        end_date: date,
        credentials: Optional[Dict[str, str]] = None,
    ) -> Tuple[DatasetBundle, str]:
        datasets = self._base_datasets()
        fetched: Dict[str, int] = {}
        for dataset, endpoint in self.dataset_endpoints.items():
            frame = self._request_dataset(endpoint, start_date, end_date, credentials)
            if frame is None or frame.empty:
                fetched[dataset] = len(datasets.get(dataset, []))
                continue
            if dataset == "sales" and "date" in frame.columns:
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                frame = frame.dropna(subset=["date"])  # type: ignore[arg-type]
            datasets[dataset] = frame
            fetched[dataset] = len(frame)
        message = "、".join(
            f"{dataset}:{count:,}件" for dataset, count in fetched.items() if count
        )
        if not message:
            message = "APIレスポンスが無いためサンプルデータを利用しました。"
        else:
            message = f"{self.provider_label} APIから {message} を取得しました。"
        if "sales" in datasets:
            datasets["sales"] = self._filter_sales_range(
                datasets["sales"], start_date, end_date
            )
        return datasets, message


class POSClient(RESTIntegrationClient):
    def __init__(self) -> None:
        super().__init__(
            provider_label="POSレジ",
            dataset_endpoints={
                "sales": "api/pos/sales",
                "inventory": "api/pos/inventory",
            },
        )


class InventoryControlClient(RESTIntegrationClient):
    def __init__(self) -> None:
        super().__init__(
            provider_label="在庫管理システム",
            dataset_endpoints={
                "inventory": "api/inventory/stock_levels",
                "sales": "api/inventory/shipments",
            },
        )


class FreeeAccountingClient(RESTIntegrationClient):
    def __init__(self) -> None:
        super().__init__(
            provider_label="freee会計",
            dataset_endpoints={
                "sales": "api/accounting/sales",
                "fixed_costs": "api/accounting/fixed_costs",
            },
        )


class YayoiAccountingClient(RESTIntegrationClient):
    def __init__(self) -> None:
        super().__init__(
            provider_label="弥生会計",
            dataset_endpoints={
                "sales": "api/yayoi/sales",
                "fixed_costs": "api/yayoi/fixed_costs",
            },
        )


PROVIDER_CLIENTS: Dict[str, BaseIntegrationClient] = {
    "POSレジ": POSClient(),
    "在庫管理システム": InventoryControlClient(),
    "freee会計": FreeeAccountingClient(),
    "弥生会計": YayoiAccountingClient(),
}


def get_client(provider_label: str) -> BaseIntegrationClient:
    if provider_label not in PROVIDER_CLIENTS:
        raise ValueError(f"Unsupported provider: {provider_label}")
    return PROVIDER_CLIENTS[provider_label]
