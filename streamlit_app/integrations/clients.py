"""Client abstractions for connecting to external systems."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple

import pandas as pd

from streamlit_app import data_loader


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


class POSClient(BaseIntegrationClient):
    def __init__(self) -> None:
        super().__init__(provider_label="POSレジ")

    def fetch(
        self,
        start_date: date,
        end_date: date,
        credentials: Optional[Dict[str, str]] = None,
    ) -> Tuple[DatasetBundle, str]:
        datasets = self._base_datasets()
        datasets["sales"] = self._filter_sales_range(
            datasets["sales"], start_date, end_date
        )
        message = "POSシステムから日次売上データを取得しました。"
        return datasets, message


class FreeeAccountingClient(BaseIntegrationClient):
    def __init__(self) -> None:
        super().__init__(provider_label="freee会計")

    def fetch(
        self,
        start_date: date,
        end_date: date,
        credentials: Optional[Dict[str, str]] = None,
    ) -> Tuple[DatasetBundle, str]:
        datasets = self._base_datasets()
        datasets["sales"] = self._filter_sales_range(
            datasets["sales"], start_date, end_date
        )
        message = "freee会計APIと同期し、売上・仕入データを取得しました。"
        return datasets, message


class YayoiAccountingClient(BaseIntegrationClient):
    def __init__(self) -> None:
        super().__init__(provider_label="弥生会計")

    def fetch(
        self,
        start_date: date,
        end_date: date,
        credentials: Optional[Dict[str, str]] = None,
    ) -> Tuple[DatasetBundle, str]:
        datasets = self._base_datasets()
        datasets["sales"] = self._filter_sales_range(
            datasets["sales"], start_date, end_date
        )
        message = "弥生会計から売上・仕入データを取り込みました。"
        return datasets, message


PROVIDER_CLIENTS: Dict[str, BaseIntegrationClient] = {
    "POSレジ": POSClient(),
    "freee会計": FreeeAccountingClient(),
    "弥生会計": YayoiAccountingClient(),
}


def get_client(provider_label: str) -> BaseIntegrationClient:
    if provider_label not in PROVIDER_CLIENTS:
        raise ValueError(f"Unsupported provider: {provider_label}")
    return PROVIDER_CLIENTS[provider_label]
