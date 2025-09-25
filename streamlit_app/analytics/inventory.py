"""Inventory related analytics."""
from __future__ import annotations

from datetime import date
from typing import Dict, Optional, Tuple

import pandas as pd


DEFAULT_ROLLING_WINDOW = 7
DEFAULT_SAFETY_BUFFER_DAYS = 3


def _coerce_float(value: object, default: float = 0.0) -> float:
    """Safely convert ``value`` to ``float`` while handling ``pd.NA`` values."""

    if pd.isna(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: object, default: int) -> int:
    """Safely convert ``value`` to ``int`` with sensible fallbacks."""

    if pd.isna(value):
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _resolve_date_range(
    sales: pd.DataFrame, start: Optional[date], end: Optional[date]
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if start is not None and end is not None:
        return pd.Timestamp(start), pd.Timestamp(end)
    if sales.empty:
        today = pd.Timestamp.today().normalize()
        return today - pd.Timedelta(days=30), today
    return sales["date"].min().normalize(), sales["date"].max().normalize()


def inventory_overview(
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    safety_buffer_days: int = DEFAULT_SAFETY_BUFFER_DAYS,
) -> pd.DataFrame:
    """Estimate inventory KPIs using moving averages and sales cost."""

    if inventory.empty:
        return inventory.head(0)

    working_sales = sales.copy()
    if "date" in working_sales.columns:
        working_sales["date"] = pd.to_datetime(
            working_sales["date"], errors="coerce"
        )
    else:
        working_sales["date"] = pd.NaT

    numeric_columns = ["sales_qty", "cogs_amount"]
    for column in numeric_columns:
        if column in working_sales.columns:
            working_sales[column] = pd.to_numeric(
                working_sales[column], errors="coerce"
            ).fillna(0.0)
        else:
            working_sales[column] = 0.0

    for column in ("store", "product"):
        if column not in working_sales.columns:
            working_sales[column] = None

    working_sales = working_sales.dropna(subset=["date"]).copy()

    window = max(_coerce_int(rolling_window, DEFAULT_ROLLING_WINDOW), 1)
    buffer_days = max(_coerce_int(safety_buffer_days, DEFAULT_SAFETY_BUFFER_DAYS), 0)

    period_start, period_end = _resolve_date_range(working_sales, start_date, end_date)

    daily_sales = (
        working_sales.assign(date=working_sales["date"].dt.floor("D"))
        .groupby(["store", "product", "date"], as_index=False)[
            ["sales_qty", "cogs_amount"]
        ]
        .sum()
    )

    records = []
    for _, row in inventory.iterrows():
        store_name = row.get("store")
        product_name = row.get("product")
        opening = _coerce_float(row.get("opening_stock", 0))
        purchase = _coerce_float(row.get("planned_purchase", 0))
        safety_stock = _coerce_float(row.get("safety_stock", 0))
        available = max(opening + purchase, 0)

        product_sales = daily_sales[
            (daily_sales["store"] == store_name)
            & (daily_sales["product"] == product_name)
        ].copy()

        timeline = pd.date_range(period_start, period_end, freq="D")
        if product_sales.empty:
            profile = pd.DataFrame(
                {
                    "date": timeline,
                    "sales_qty": 0.0,
                    "cogs_amount": 0.0,
                }
            )
        else:
            product_sales = product_sales.set_index("date").reindex(timeline, fill_value=0.0)
            product_sales = product_sales.reset_index().rename(columns={"index": "date"})
            profile = product_sales

        profile = profile.sort_values("date")
        profile["cumulative_qty"] = profile["sales_qty"].cumsum()
        profile["estimated_stock"] = (available - profile["cumulative_qty"]).clip(lower=0.0)
        profile["moving_stock"] = (
            profile["estimated_stock"].rolling(window=window, min_periods=1).mean()
        )
        profile["moving_sales_qty"] = (
            profile["sales_qty"].rolling(window=window, min_periods=1).mean()
        )
        profile["moving_cogs"] = (
            profile["cogs_amount"].rolling(window=window, min_periods=1).mean()
        )

        avg_inventory = float(profile["moving_stock"].mean()) if not profile.empty else available
        avg_inventory = max(avg_inventory, 0.0)
        total_cogs = float(profile["cogs_amount"].sum())
        total_sales_qty = float(profile["sales_qty"].sum())
        avg_daily_qty = float(profile["moving_sales_qty"].iloc[-1]) if not profile.empty else 0.0
        avg_daily_cogs = float(profile["moving_cogs"].iloc[-1]) if not profile.empty else 0.0
        ending_stock = float(profile["estimated_stock"].iloc[-1]) if not profile.empty else available

        turnover = total_cogs / avg_inventory if avg_inventory > 0 else 0.0
        turnover = max(turnover, 0.0)

        safety_lower = max(safety_stock - avg_daily_qty * buffer_days, 0.0)
        safety_upper = safety_stock + avg_daily_qty * buffer_days
        coverage_days = (ending_stock / avg_daily_qty) if avg_daily_qty > 0 else None

        status = "適正"
        if ending_stock <= 0:
            status = "在庫切れ"
        elif ending_stock <= max(safety_lower, safety_stock * 0.5):
            status = "在庫少"
        elif ending_stock >= safety_upper and safety_upper > 0:
            status = "在庫過多"

        records.append(
            {
                **row,
                "sold_qty": total_sales_qty,
                "estimated_stock": ending_stock,
                "avg_inventory": avg_inventory,
                "avg_daily_qty": avg_daily_qty,
                "avg_daily_cogs": avg_daily_cogs,
                "total_cogs": total_cogs,
                "turnover": round(turnover, 2),
                "safety_lower": safety_lower,
                "safety_upper": safety_upper,
                "coverage_days": coverage_days,
                "stock_status": status,
                "safety_buffer_days": buffer_days,
                "analysis_window": window,
            }
        )

    overview_df = pd.DataFrame(records)
    if "coverage_days" in overview_df.columns:
        overview_df["coverage_days"] = overview_df["coverage_days"].round(1)
    return overview_df


def turnover_by_category(overview: pd.DataFrame, *, period_days: int) -> pd.DataFrame:
    """Calculate inventory turnover ratio per category using COGS."""

    if overview.empty:
        return overview.head(0)

    if period_days <= 0:
        period_days = 1

    df = overview.copy()
    df["avg_inventory"] = df["avg_inventory"].replace(0, pd.NA)
    df["category_turnover"] = (
        df["total_cogs"].fillna(0) * (365 / period_days)
    ) / df["avg_inventory"]
    df["category_turnover"] = df["category_turnover"].fillna(0.0)

    grouped = (
        df.groupby("category")
        .agg(
            turnover=("category_turnover", "mean"),
            avg_inventory=("avg_inventory", "mean"),
            cogs=("total_cogs", "sum"),
        )
        .reset_index()
    )
    grouped["turnover"] = grouped["turnover"].round(2)
    return grouped.sort_values("turnover", ascending=False)


def inventory_advice(
    overview: pd.DataFrame, rank_lookup: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Generate qualitative advice per product with lead time context."""

    def create_message(row: pd.Series) -> str:
        rank = None
        if rank_lookup:
            rank = rank_lookup.get(row["product"])
        status = row.get("stock_status", "")
        lead = row.get("coverage_days")
        lead_text = (
            f"想定残日数は約{lead:.1f}日です。" if isinstance(lead, (int, float)) else ""
        )

        if status == "在庫切れ":
            return "在庫が枯渇しています。即時の補充が必要です。"
        if status == "在庫少":
            if rank == "A":
                return "主力商品の在庫が不足しています。優先的に補充してください。" + lead_text
            return "在庫が少ないため、次回仕入での補充を検討してください。" + lead_text
        if status == "在庫過多":
            if rank == "C":
                return "回転の遅い商品の過剰在庫です。販促や仕入抑制を検討。"
            return "在庫が多めです。販売動向をモニタリングしましょう。"
        return "適正在庫を維持しています。" + (" " + lead_text if lead_text else "")

    advice_df = overview.copy()
    advice_df["advice"] = advice_df.apply(create_message, axis=1)
    return advice_df
