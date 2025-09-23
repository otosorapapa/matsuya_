"""Data transformation helpers shared across the Streamlit application."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import pandas as pd


@dataclass
class FilterState:
    store: str
    start_date: date
    end_date: date
    category: str


ALL_STORES = "全店舗"
ALL_CATEGORIES = "全カテゴリ"


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
    dataset["year"] = dataset["date"].dt.year
    dataset["month"] = dataset["date"].dt.month
    dataset["year_month"] = dataset["date"].dt.to_period("M").dt.to_timestamp()
    return dataset


def apply_filters(df: pd.DataFrame, filters: FilterState) -> pd.DataFrame:
    """Filter dataset by store, period and category."""
    dataset = df.copy()
    if filters.store != ALL_STORES:
        dataset = dataset[dataset["store"] == filters.store]
    if filters.category != ALL_CATEGORIES:
        dataset = dataset[dataset["category"] == filters.category]

    mask = (dataset["date"] >= pd.Timestamp(filters.start_date)) & (
        dataset["date"] <= pd.Timestamp(filters.end_date)
    )
    dataset = dataset.loc[mask]
    return dataset


def extract_stores(df: pd.DataFrame) -> list[str]:
    stores = sorted(df["store"].unique().tolist())
    return [ALL_STORES, *stores]


def extract_categories(df: pd.DataFrame) -> list[str]:
    categories = sorted(df["category"].unique().tolist())
    return [ALL_CATEGORIES, *categories]


def compute_comparison_period(filters: FilterState) -> FilterState:
    """Return the comparison period (one year earlier) for YoY calculations."""
    start_prev = filters.start_date.replace(year=filters.start_date.year - 1)
    end_prev = filters.end_date.replace(year=filters.end_date.year - 1)
    return FilterState(
        store=filters.store,
        start_date=start_prev,
        end_date=end_prev,
        category=filters.category,
    )


def ensure_not_empty(df: pd.DataFrame) -> pd.DataFrame:
    """Return an empty dataframe with the same columns when df is empty."""
    return df if not df.empty else df.head(0)


def to_download_csv(df: pd.DataFrame, *, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode("utf-8-sig")
