"""Product level analytics including ABC classification."""
from __future__ import annotations

from typing import Optional

import pandas as pd


ABC_THRESHOLDS = {
    "A": 0.8,
    "B": 0.9,
}


def abc_analysis(current: pd.DataFrame, comparison: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return ABC ranking with cumulative contribution."""
    grouped = (
        current.groupby("product")[["sales_amount", "gross_profit", "sales_qty"]]
        .sum()
        .reset_index()
    )
    if grouped.empty:
        return grouped

    grouped = grouped.sort_values("sales_amount", ascending=False).reset_index(drop=True)
    grouped["share"] = grouped["sales_amount"] / grouped["sales_amount"].sum()
    grouped["cumulative_share"] = grouped["share"].cumsum()

    def classify(value: float) -> str:
        if value <= ABC_THRESHOLDS["A"]:
            return "A"
        if value <= ABC_THRESHOLDS["B"]:
            return "B"
        return "C"

    grouped["rank"] = grouped["cumulative_share"].apply(classify)

    if comparison is not None and not comparison.empty:
        prev_grouped = (
            comparison.groupby("product")["sales_amount"].sum().reset_index()
        )
        grouped = grouped.merge(prev_grouped, on="product", how="left", suffixes=("", "_prev"))
        grouped["yoy_growth"] = (
            (grouped["sales_amount"] - grouped["sales_amount_prev"]) / grouped["sales_amount_prev"]
        )
        grouped.loc[
            grouped["sales_amount_prev"].isna() | (grouped["sales_amount_prev"] == 0),
            "yoy_growth",
        ] = pd.NA
    else:
        grouped["sales_amount_prev"] = pd.NA
        grouped["yoy_growth"] = pd.NA

    return grouped


def top_growth_products(abc_df: pd.DataFrame, *, limit: int = 3) -> pd.DataFrame:
    """Return the fastest growing A-rank products."""
    if "rank" not in abc_df.columns or "yoy_growth" not in abc_df.columns:
        return abc_df.head(0)
    filtered = abc_df[(abc_df["rank"] == "A") & abc_df["yoy_growth"].notna()]
    filtered = filtered.sort_values("yoy_growth", ascending=False)
    return filtered.head(limit)


def pareto_chart_data(abc_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for a Pareto chart."""
    if abc_df.empty:
        return abc_df
    columns = ["product", "sales_amount", "cumulative_share", "rank"]
    available = [column for column in columns if column in abc_df.columns]
    return abc_df[available]
