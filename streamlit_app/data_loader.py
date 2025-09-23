"""Utility functions to load and validate CSV datasets for the Streamlit app."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import IO, Callable, Dict, Iterable, Optional, Union

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

SALES_TEMPLATE_COLUMNS = [
    "date",
    "store",
    "category",
    "product",
    "sales_amount",
    "sales_qty",
    "cogs_amount",
    "gross_profit",
    "gross_margin",
]

INVENTORY_TEMPLATE_COLUMNS = [
    "store",
    "product",
    "category",
    "opening_stock",
    "planned_purchase",
    "safety_stock",
]

FIXED_COST_TEMPLATE_COLUMNS = [
    "store",
    "rent",
    "payroll",
    "utilities",
    "marketing",
    "other_fixed",
]

CsvSource = Union[str, Path, IO[str], IO[bytes], bytes, bytearray, pd.DataFrame]


@dataclass
class ValidationResult:
    """Result returned by CSV validation helpers."""

    dataframe: pd.DataFrame
    errors: pd.DataFrame
    valid: bool
    total_rows: int
    dropped_rows: int

    @property
    def has_errors(self) -> bool:
        """Return ``True`` when row-level validation errors exist."""

        return not self.errors.empty


def _coerce_to_dataframe(source: Optional[CsvSource], *, default_path: Path) -> pd.DataFrame:
    """Return a dataframe from a CSV source or fallback to a default path."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif source is None:
        df = pd.read_csv(default_path)
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


def available_templates() -> Dict[str, bytes]:
    """Return downloadable CSV templates for manual uploads."""

    return {
        "sales": generate_template_csv("sales"),
        "inventory": generate_template_csv("inventory"),
        "fixed_costs": generate_template_csv("fixed_costs"),
    }


def generate_template_csv(dataset: str) -> bytes:
    """Generate an empty CSV template that contains the required headers."""

    if dataset == "sales":
        columns = SALES_TEMPLATE_COLUMNS
    elif dataset == "inventory":
        columns = INVENTORY_TEMPLATE_COLUMNS
    elif dataset == "fixed_costs":
        columns = FIXED_COST_TEMPLATE_COLUMNS
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")

    template = pd.DataFrame(columns=columns)
    return template.to_csv(index=False).encode("utf-8-sig")


def validate_sales_csv(source: Optional[CsvSource]) -> ValidationResult:
    """Validate and clean an uploaded sales CSV."""

    return _validate_dataset(
        source,
        default_path=SAMPLE_DATA_DIR / "sales.csv",
        required_columns=REQUIRED_SALES_COLUMNS,
        numeric_columns=["sales_amount", "sales_qty", "cogs_amount"],
        date_columns=["date"],
        loader=load_sales_data,
        dataset_label="売上",
    )


def validate_inventory_csv(source: Optional[CsvSource]) -> ValidationResult:
    """Validate and clean an uploaded inventory CSV."""

    return _validate_dataset(
        source,
        default_path=SAMPLE_DATA_DIR / "inventory.csv",
        required_columns=REQUIRED_INVENTORY_COLUMNS,
        numeric_columns=["opening_stock", "planned_purchase", "safety_stock"],
        date_columns=None,
        loader=load_inventory_data,
        dataset_label="在庫/仕入",
    )


def validate_fixed_costs_csv(source: Optional[CsvSource]) -> ValidationResult:
    """Validate and clean an uploaded fixed cost CSV."""

    return _validate_dataset(
        source,
        default_path=SAMPLE_DATA_DIR / "fixed_costs.csv",
        required_columns=REQUIRED_FIXED_COST_COLUMNS,
        numeric_columns=["rent", "payroll", "utilities", "marketing", "other_fixed"],
        date_columns=None,
        loader=load_fixed_costs,
        dataset_label="固定費",
    )


ERROR_COLUMNS = ["行番号", "列名", "内容"]


def _validate_dataset(
    source: Optional[CsvSource],
    *,
    default_path: Path,
    required_columns: Iterable[str],
    numeric_columns: Optional[Iterable[str]],
    date_columns: Optional[Iterable[str]],
    loader: Callable[[Optional[CsvSource]], pd.DataFrame],
    dataset_label: str,
) -> ValidationResult:
    df_raw = _coerce_to_dataframe(source, default_path=default_path)
    total_rows = len(df_raw)
    errors: list[Dict[str, object]] = []

    missing = set(required_columns) - set(df_raw.columns)
    if missing:
        errors.append(
            {
                "行番号": "全体",
                "列名": "列定義",
                "内容": f"{dataset_label}データに必要な列が不足しています: "
                + ", ".join(sorted(missing)),
            }
        )
        empty = pd.DataFrame(columns=sorted(required_columns))
        processed = loader(empty)
        return ValidationResult(
            dataframe=processed,
            errors=_build_error_frame(errors),
            valid=False,
            total_rows=total_rows,
            dropped_rows=total_rows,
        )

    invalid_mask = pd.Series(False, index=df_raw.index)
    converted: Dict[str, pd.Series] = {}

    if date_columns:
        for column in date_columns:
            parsed = pd.to_datetime(df_raw[column], errors="coerce")
            invalid = parsed.isna()
            for idx in df_raw.index[invalid]:
                errors.append(
                    {
                        "行番号": int(idx) + 2,
                        "列名": column,
                        "内容": "日付形式が不正です。YYYY-MM-DD形式で入力してください。",
                    }
                )
            converted[column] = parsed
            invalid_mask |= invalid

    if numeric_columns:
        for column in numeric_columns:
            numeric = pd.to_numeric(df_raw[column], errors="coerce")
            invalid = numeric.isna()
            for idx in df_raw.index[invalid]:
                errors.append(
                    {
                        "行番号": int(idx) + 2,
                        "列名": column,
                        "内容": "数値として認識できません。空欄または数値を入力してください。",
                    }
                )
            converted[column] = numeric
            invalid_mask |= invalid

    clean_df = df_raw.loc[~invalid_mask].copy()
    for column, values in converted.items():
        clean_df.loc[:, column] = values.loc[~invalid_mask]

    processed_df = loader(clean_df)
    errors_df = _build_error_frame(errors)
    dropped_rows = total_rows - len(clean_df)

    return ValidationResult(
        dataframe=processed_df,
        errors=errors_df,
        valid=True,
        total_rows=total_rows,
        dropped_rows=dropped_rows,
    )


def _build_error_frame(errors: Iterable[Dict[str, object]]) -> pd.DataFrame:
    if not errors:
        return pd.DataFrame(columns=ERROR_COLUMNS)
    return pd.DataFrame(errors, columns=ERROR_COLUMNS)
