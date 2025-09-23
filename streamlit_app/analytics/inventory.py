"""Inventory related analytics."""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def inventory_overview(sales: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Estimate closing inventory using sales consumption."""
    sold = (
        sales.groupby(["store", "product"])["sales_qty"]
        .sum()
        .reset_index()
        .rename(columns={"sales_qty": "sold_qty"})
    )
    merged = inventory.merge(sold, on=["store", "product"], how="left")
    merged["sold_qty"] = merged["sold_qty"].fillna(0)
    merged["estimated_stock"] = (
        merged["opening_stock"] + merged["planned_purchase"] - merged["sold_qty"]
    )
    merged["stock_status"] = "適正"
    merged.loc[merged["estimated_stock"] <= 0, "stock_status"] = "在庫切れ"
    merged.loc[
        (merged["estimated_stock"] > 0)
        & (merged["estimated_stock"] <= merged["safety_stock"] * 0.7),
        "stock_status",
    ] = "在庫少"
    merged.loc[merged["estimated_stock"] > merged["safety_stock"] * 1.3, "stock_status"] = "在庫過多"
    return merged


def turnover_by_category(overview: pd.DataFrame, *, period_days: int) -> pd.DataFrame:
    """Calculate inventory turnover ratio per category."""
    if period_days <= 0:
        period_days = 1
    df = overview.copy()
    df["avg_inventory"] = (df["opening_stock"] + df["estimated_stock"]) / 2
    annualised_factor = 365 / period_days
    df["turnover"] = (
        df["sold_qty"].fillna(0) * annualised_factor
    ) / df["avg_inventory"].replace(0, pd.NA)
    df["turnover"] = df["turnover"].fillna(0.0)
    return (
        df.groupby("category")[["turnover", "avg_inventory"]]
        .mean()
        .reset_index()
        .sort_values("turnover", ascending=False)
    )


def inventory_advice(overview: pd.DataFrame, rank_lookup: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Generate qualitative advice per product."""
    def create_message(row: pd.Series) -> str:
        rank = None
        if rank_lookup:
            rank = rank_lookup.get(row["product"])
        status = row.get("stock_status", "")
        if status == "在庫切れ":
            return "在庫が枯渇しています。即時の補充が必要です。"
        if status == "在庫少":
            if rank == "A":
                return "主力商品の在庫が不足しています。優先的に補充してください。"
            return "在庫が少ないため、次回仕入での補充を検討してください。"
        if status == "在庫過多":
            if rank == "C":
                return "回転の遅い商品の過剰在庫です。販促や仕入抑制を検討。"
            return "在庫が多めです。販売動向をモニタリングしましょう。"
        return "適正在庫を維持しています。"

    advice_df = overview.copy()
    advice_df["advice"] = advice_df.apply(create_message, axis=1)
    return advice_df
