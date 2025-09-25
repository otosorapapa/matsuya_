"""Data transformation helpers shared across the Streamlit application."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Sequence

import pandas as pd


ALL_STORES = "全店舗"
ALL_CATEGORIES = "全カテゴリ"
ALL_REGIONS = "全地域"
ALL_CHANNELS = "全チャネル"

DEFAULT_REGION = "未設定"
DEFAULT_CHANNEL = "店頭販売"

STORE_DIMENSIONS = {
    "本店": {"region": "関東", "channel": "直営"},
    "福岡西店": {"region": "九州", "channel": "直営"},
    "唐津店": {"region": "九州", "channel": "フランチャイズ"},
}


@dataclass
class FilterState:
    stores: Sequence[str]
    start_date: date
    end_date: date
    categories: Sequence[str]
    regions: Sequence[str] = field(default_factory=list)
    channels: Sequence[str] = field(default_factory=list)
    period_granularity: str = "daily"
    breakdown_dimension: str = "store"

    @property
    def store(self) -> str:
        """Return a representative store selection for legacy consumers."""

        if not self.stores or len(self.stores) != 1:
            return ALL_STORES
        return self.stores[0]

    @property
    def category(self) -> str:
        """Return a representative category selection for legacy consumers."""

        if not self.categories or len(self.categories) != 1:
            return ALL_CATEGORIES
        return self.categories[0]


NUMERIC_COLUMNS = [
    "sales_amount",
    "sales_qty",
    "cogs_amount",
    "gross_profit",
    "gross_margin",
]


def prepare_sales_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric types and derive helper columns for analysis."""
    dataset = df.copy()
    dataset[NUMERIC_COLUMNS] = dataset[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    dataset[NUMERIC_COLUMNS] = dataset[NUMERIC_COLUMNS].fillna(0)

    dataset["avg_unit_price"] = dataset.apply(
        lambda row: row["sales_amount"] / row["sales_qty"] if row["sales_qty"] else 0,
        axis=1,
    )

    def _store_attribute(store: str, key: str, default: str) -> str:
        attributes = STORE_DIMENSIONS.get(store, {})
        return str(attributes.get(key, default))

    if "region" not in dataset.columns:
        dataset["region"] = dataset["store"].map(
            lambda store: _store_attribute(store, "region", DEFAULT_REGION)
        )
    else:
        dataset["region"] = dataset["region"].fillna(
            dataset["store"].map(lambda store: _store_attribute(store, "region", DEFAULT_REGION))
        )
        dataset["region"] = dataset["region"].replace({"": DEFAULT_REGION}).astype(str)

    if "channel" not in dataset.columns:
        dataset["channel"] = dataset["store"].map(
            lambda store: _store_attribute(store, "channel", DEFAULT_CHANNEL)
        )
    else:
        dataset["channel"] = dataset["channel"].fillna(
            dataset["store"].map(lambda store: _store_attribute(store, "channel", DEFAULT_CHANNEL))
        )
        dataset["channel"] = dataset["channel"].replace({"": DEFAULT_CHANNEL}).astype(str)

    dataset["year"] = dataset["date"].dt.year
    dataset["month"] = dataset["date"].dt.month
    dataset["year_month"] = dataset["date"].dt.to_period("M").dt.to_timestamp()
    return dataset


def apply_filters(df: pd.DataFrame, filters: FilterState) -> pd.DataFrame:
    """Filter dataset by store, period and category."""
    dataset = df.copy()
    if filters.stores:
        dataset = dataset[dataset["store"].isin(filters.stores)]
    if filters.categories:
        dataset = dataset[dataset["category"].isin(filters.categories)]
    if filters.regions and "region" in dataset.columns:
        dataset = dataset[dataset["region"].isin(filters.regions)]
    if filters.channels and "channel" in dataset.columns:
        dataset = dataset[dataset["channel"].isin(filters.channels)]

    mask = (dataset["date"] >= pd.Timestamp(filters.start_date)) & (
        dataset["date"] <= pd.Timestamp(filters.end_date)
    )
    dataset = dataset.loc[mask]
    return dataset


def extract_stores(df: pd.DataFrame) -> List[str]:
    if "store" not in df.columns:
        return []
    stores = (
        df["store"].dropna().astype(str).unique().tolist()
        if not df.empty
        else []
    )
    return sorted(stores)


def extract_categories(df: pd.DataFrame) -> List[str]:
    if "category" not in df.columns:
        return []
    categories = (
        df["category"].dropna().astype(str).unique().tolist()
        if not df.empty
        else []
    )
    return sorted(categories)


def extract_regions(df: pd.DataFrame) -> List[str]:
    if "region" not in df.columns:
        return []
    regions = (
        df["region"].dropna().astype(str).replace("", pd.NA).dropna().unique().tolist()
        if not df.empty
        else []
    )
    return sorted(regions)


def extract_channels(df: pd.DataFrame) -> List[str]:
    if "channel" not in df.columns:
        return []
    channels = (
        df["channel"].dropna().astype(str).replace("", pd.NA).dropna().unique().tolist()
        if not df.empty
        else []
    )
    return sorted(channels)


def compute_comparison_period(filters: FilterState) -> FilterState:
    """Return the comparison period (one year earlier) for YoY calculations."""
    start_prev = (
        pd.Timestamp(filters.start_date) - pd.DateOffset(years=1)
    ).date()
    end_prev = (pd.Timestamp(filters.end_date) - pd.DateOffset(years=1)).date()
    return FilterState(
        stores=list(filters.stores),
        start_date=start_prev,
        end_date=end_prev,
        categories=list(filters.categories),
        regions=list(filters.regions),
        channels=list(filters.channels),
        period_granularity=filters.period_granularity,
        breakdown_dimension=filters.breakdown_dimension,
    )


def ensure_not_empty(df: pd.DataFrame) -> pd.DataFrame:
    """Return an empty dataframe with the same columns when df is empty."""
    return df if not df.empty else df.head(0)


def to_download_csv(df: pd.DataFrame, *, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode("utf-8-sig")
