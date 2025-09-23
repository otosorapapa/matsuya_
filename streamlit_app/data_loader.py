"""Utility functions to load and validate CSV datasets for the Streamlit app."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import IO, Optional, Union

import pandas as pd


SAMPLE_DATA_DIR = Path(__file__).resolve().parent / "assets" / "sample_data"

REQUIRED_SALES_COLUMNS = {
    "date",
    "store",
    "category",
    "product",
    "sales_amount",
    "sales_qty",
    "cogs_amount",
}

REQUIRED_INVENTORY_COLUMNS = {
    "store",
    "product",
    "category",
    "opening_stock",
    "planned_purchase",
    "safety_stock",
}

REQUIRED_FIXED_COST_COLUMNS = {
    "store",
    "rent",
    "payroll",
    "utilities",
    "marketing",
    "other_fixed",
}

CsvSource = Union[str, Path, IO[str], IO[bytes], bytes, bytearray]


def _coerce_to_dataframe(source: Optional[CsvSource], *, default_path: Path) -> pd.DataFrame:
    """Return a dataframe from a CSV source or fallback to a default path."""
    path = default_path
    if source is None:
        df = pd.read_csv(path)
    elif isinstance(source, (bytes, bytearray)):
        df = pd.read_csv(BytesIO(source))
    else:
        df = pd.read_csv(source)
    return df


def load_sales_data(source: Optional[CsvSource] = None) -> pd.DataFrame:
    """Load and validate the sales dataset.

    Args:
        source: CSV path or file-like object. When ``None`` the bundled sample
            dataset is used.

    Returns:
        A dataframe with parsed dates and computed gross profit/margin.
    """
    df = _coerce_to_dataframe(source, default_path=SAMPLE_DATA_DIR / "sales.csv")

    missing = REQUIRED_SALES_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "売上データの列が不足しています。必要な列: "
            + ", ".join(sorted(missing))
        )

    df["date"] = pd.to_datetime(df["date"])

    if "gross_profit" not in df.columns:
        df["gross_profit"] = df["sales_amount"] - df["cogs_amount"]
    if "gross_margin" not in df.columns:
        df["gross_margin"] = (
            df["gross_profit"] / df["sales_amount"].replace(0, pd.NA)
        ).fillna(0.0)
    return df


def load_inventory_data(source: Optional[CsvSource] = None) -> pd.DataFrame:
    """Load the inventory master dataset."""
    df = _coerce_to_dataframe(source, default_path=SAMPLE_DATA_DIR / "inventory.csv")
    missing = REQUIRED_INVENTORY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "在庫データの列が不足しています。必要な列: " + ", ".join(sorted(missing))
        )
    numeric_cols = ["opening_stock", "planned_purchase", "safety_stock"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def load_fixed_costs(source: Optional[CsvSource] = None) -> pd.DataFrame:
    """Load the fixed cost master dataset."""
    df = _coerce_to_dataframe(source, default_path=SAMPLE_DATA_DIR / "fixed_costs.csv")
    missing = REQUIRED_FIXED_COST_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "固定費データの列が不足しています。必要な列: " + ", ".join(sorted(missing))
        )
    numeric_cols = ["rent", "payroll", "utilities", "marketing", "other_fixed"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def available_sample_files() -> dict[str, Path]:
    """Return paths to the built-in sample CSV files."""
    return {
        "sales": SAMPLE_DATA_DIR / "sales.csv",
        "inventory": SAMPLE_DATA_DIR / "inventory.csv",
        "fixed_costs": SAMPLE_DATA_DIR / "fixed_costs.csv",
    }
