"""Profitability analytics for store level reporting."""
from __future__ import annotations

from typing import Dict

import pandas as pd


def store_profitability(sales: pd.DataFrame, fixed_costs: pd.DataFrame) -> pd.DataFrame:
    """Combine sales and fixed costs to build a store level P&L table."""
    if sales.empty:
        merged = fixed_costs.copy()
        merged["sales_amount"] = 0
        merged["gross_profit"] = 0
    else:
        aggregated = (
            sales.groupby("store")[["sales_amount", "gross_profit"]]
            .sum()
            .reset_index()
        )
        merged = aggregated.merge(fixed_costs, on="store", how="left")

    cost_columns = ["rent", "payroll", "utilities", "marketing", "other_fixed"]
    merged[cost_columns] = merged[cost_columns].fillna(0)
    merged["total_fixed_cost"] = merged[cost_columns].sum(axis=1)
    merged["operating_profit"] = merged["gross_profit"] - merged["total_fixed_cost"]
    merged["gross_margin"] = merged["gross_profit"] / merged["sales_amount"].replace(0, pd.NA)
    merged["gross_margin"] = merged["gross_margin"].fillna(0.0)
    merged["operating_margin"] = merged["operating_profit"] / merged["sales_amount"].replace(0, pd.NA)
    merged["operating_margin"] = merged["operating_margin"].fillna(0.0)
    return merged


def profitability_chart_data(pnl_df: pd.DataFrame) -> pd.DataFrame:
    return pnl_df[["store", "sales_amount", "gross_profit", "operating_profit"]]


def breakeven_sales(gross_margin: float, fixed_cost: float) -> float:
    if gross_margin <= 0:
        return float("inf")
    return fixed_cost / gross_margin


def total_fixed_costs(row: pd.Series) -> float:
    return float(
        row[["rent", "payroll", "utilities", "marketing", "other_fixed"]].sum()
    )
