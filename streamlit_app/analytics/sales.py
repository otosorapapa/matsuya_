"""Sales related aggregations used in the dashboard."""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def kpi_summary(current: pd.DataFrame, comparison: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Compute headline KPIs for the sales dashboard."""
    total_sales = float(current["sales_amount"].sum())
    total_gross_profit = float(current["gross_profit"].sum())
    total_customers = float(current["sales_qty"].sum())
    avg_unit_price = float(total_sales / total_customers) if total_customers else 0.0
    gross_margin = float(total_gross_profit / total_sales) if total_sales else 0.0

    yoy_rate = None
    if comparison is not None and not comparison.empty:
        base = float(comparison["sales_amount"].sum())
        if base:
            yoy_rate = (total_sales - base) / base

    return {
        "total_sales": total_sales,
        "total_gross_profit": total_gross_profit,
        "total_customers": total_customers,
        "avg_unit_price": avg_unit_price,
        "gross_margin": gross_margin,
        "yoy_rate": yoy_rate,
    }


def daily_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily sales and gross profit for charts."""
    aggregated = (
        df.groupby("date")[
            ["sales_amount", "sales_qty", "gross_profit"]
        ]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    return aggregated


def monthly_performance(current: pd.DataFrame, comparison: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return a dataframe summarising monthly sales with YoY comparison."""
    current_monthly = (
        current.groupby("year_month")[
            ["sales_amount", "gross_profit", "sales_qty"]
        ]
        .sum()
        .reset_index()
        .rename(columns={"sales_amount": "sales_amount", "gross_profit": "gross_profit"})
    )
    current_monthly["month_label"] = current_monthly["year_month"].dt.strftime("%Y-%m")

    if comparison is None or comparison.empty:
        current_monthly["yoy_sales"] = pd.NA
        current_monthly["yoy_rate"] = pd.NA
        return current_monthly

    prev_monthly = (
        comparison.groupby("year_month")["sales_amount"].sum().reset_index()
    )
    prev_monthly = prev_monthly.rename(
        columns={"year_month": "comparison_month", "sales_amount": "sales_prev"}
    )
    prev_monthly["month_label"] = (
        prev_monthly["comparison_month"] + pd.offsets.DateOffset(years=1)
    ).dt.strftime("%Y-%m")

    merged = current_monthly.merge(
        prev_monthly[["month_label", "sales_prev"]],
        on="month_label",
        how="left",
    )
    merged["yoy_sales"] = merged["sales_prev"]
    merged["yoy_rate"] = (merged["sales_amount"] - merged["sales_prev"]) / merged["sales_prev"]
    merged.loc[merged["sales_prev"].isna() | (merged["sales_prev"] == 0), "yoy_rate"] = pd.NA
    return merged.sort_values("year_month")


def sales_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Return category level aggregation for pie charts."""
    return (
        df.groupby("category")[
            ["sales_amount", "gross_profit", "sales_qty"]
        ]
        .sum()
        .reset_index()
        .sort_values("sales_amount", ascending=False)
    )


def sales_by_store(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("store")[["sales_amount", "gross_profit"]]
        .sum()
        .reset_index()
        .sort_values("sales_amount", ascending=False)
    )
