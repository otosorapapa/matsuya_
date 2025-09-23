"""Sales analytics helpers."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
from pandas.tseries.offsets import MonthEnd, YearEnd

GRANULARITY_CONFIG = {
    "daily": {"label": "日次", "format": "%Y-%m-%d"},
    "weekly": {"label": "週次", "format": "%Y-%m-%d週"},
    "monthly": {"label": "月次", "format": "%Y-%m"},
    "yearly": {"label": "年次", "format": "%Y年"},
}

BREAKDOWN_LABELS = {
    "store": "店舗別",
    "category": "カテゴリ別",
    "region": "地域別",
    "channel": "チャネル別",
}

SUPPORTED_BREAKDOWNS = set(BREAKDOWN_LABELS.keys())
COMPARISON_OFFSET = pd.DateOffset(years=1)


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


def granularity_label(key: str) -> str:
    return GRANULARITY_CONFIG.get(key, {}).get("label", key)


def breakdown_label(key: str) -> str:
    return BREAKDOWN_LABELS.get(key, key)


def resolve_breakdown_column(breakdown: Optional[str]) -> Optional[str]:
    if breakdown in SUPPORTED_BREAKDOWNS:
        return breakdown
    return None


def _period_start(series: pd.Series, granularity: str) -> pd.Series:
    if granularity == "daily":
        return series.dt.floor("D")
    if granularity == "weekly":
        return series.dt.to_period("W-MON").apply(lambda period: period.start_time)
    if granularity == "monthly":
        return series.dt.to_period("M").dt.to_timestamp()
    if granularity == "yearly":
        return series.dt.to_period("Y").dt.to_timestamp()
    return series.dt.floor("D")


def _format_period_label(start: pd.Timestamp, granularity: str) -> str:
    config = GRANULARITY_CONFIG.get(granularity, {})
    fmt = config.get("format", "%Y-%m-%d")
    return start.strftime(fmt)


def _period_key(start: pd.Timestamp) -> str:
    return start.strftime("%Y-%m-%d")


def aggregate_timeseries(
    df: pd.DataFrame,
    granularity: str,
    breakdown: Optional[str] = None,
) -> pd.DataFrame:
    if df.empty:
        columns = ["period_start", "period_label", "period_key", "sales_amount", "gross_profit", "sales_qty", "gross_margin"]
        if breakdown:
            columns.insert(1, breakdown)
        return pd.DataFrame(columns=columns)

    dataset = df.copy()
    dataset["period_start"] = _period_start(dataset["date"], granularity)
    dataset["period_label"] = dataset["period_start"].apply(lambda ts: _format_period_label(ts, granularity))
    dataset["period_key"] = dataset["period_start"].apply(_period_key)

    group_cols = ["period_start", "period_label", "period_key"]
    if breakdown:
        group_cols.append(breakdown)

    aggregated = (
        dataset.groupby(group_cols)[["sales_amount", "gross_profit", "sales_qty"]]
        .sum()
        .reset_index()
        .sort_values(group_cols)
    )
    aggregated["gross_margin"] = (
        aggregated["gross_profit"] / aggregated["sales_amount"].replace(0, pd.NA)
    ).fillna(0.0)
    return aggregated


def timeseries_with_comparison(
    current: pd.DataFrame,
    comparison: Optional[pd.DataFrame],
    granularity: str,
    breakdown: Optional[str] = None,
) -> pd.DataFrame:
    segment = resolve_breakdown_column(breakdown)
    current_agg = aggregate_timeseries(current, granularity, segment)

    if comparison is None or comparison.empty:
        for column in [
            "comparison_sales",
            "comparison_gross_profit",
            "comparison_margin",
            "yoy_rate",
            "gross_profit_yoy",
            "margin_delta",
        ]:
            current_agg[column] = pd.NA
        return current_agg

    comparison_agg = aggregate_timeseries(comparison, granularity, segment)
    comparison_agg["period_start"] = comparison_agg["period_start"] + COMPARISON_OFFSET
    comparison_agg["period_label"] = comparison_agg["period_start"].apply(
        lambda ts: _format_period_label(ts, granularity)
    )
    comparison_agg["period_key"] = comparison_agg["period_start"].apply(_period_key)
    comparison_agg = comparison_agg.rename(
        columns={
            "sales_amount": "comparison_sales",
            "gross_profit": "comparison_gross_profit",
            "sales_qty": "comparison_qty",
            "gross_margin": "comparison_margin",
        }
    )

    join_cols = ["period_key"]
    if segment:
        join_cols.append(segment)

    merged = current_agg.merge(
        comparison_agg[join_cols + ["comparison_sales", "comparison_gross_profit", "comparison_margin"]],
        on=join_cols,
        how="left",
    )

    for column in ["comparison_sales", "comparison_gross_profit", "comparison_margin"]:
        merged[column] = merged[column].astype(float)

    sales_base = merged["comparison_sales"].replace(0, pd.NA)
    merged["yoy_rate"] = (merged["sales_amount"] - merged["comparison_sales"]) / sales_base
    merged.loc[sales_base.isna(), "yoy_rate"] = pd.NA

    profit_base = merged["comparison_gross_profit"].replace(0, pd.NA)
    merged["gross_profit_yoy"] = (
        merged["gross_profit"] - merged["comparison_gross_profit"]
    ) / profit_base
    merged.loc[profit_base.isna(), "gross_profit_yoy"] = pd.NA

    merged["margin_delta"] = merged["gross_margin"] - merged["comparison_margin"]
    merged.loc[merged["comparison_margin"].isna(), "margin_delta"] = pd.NA
    return merged


def period_bounds(period_key: str, granularity: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.to_datetime(period_key)
    if granularity == "daily":
        end = start
    elif granularity == "weekly":
        end = start + pd.Timedelta(days=6)
    elif granularity == "monthly":
        end = start + MonthEnd(0)
    elif granularity == "yearly":
        end = start + YearEnd(0)
    else:
        end = start
    return start, end


def drilldown_details(
    df: pd.DataFrame,
    selections: Sequence[Dict[str, object]],
    granularity: str,
    breakdown: Optional[str] = None,
) -> pd.DataFrame:
    if not selections:
        return pd.DataFrame()

    breakdown_column = resolve_breakdown_column(breakdown)
    detail_columns = [
        column
        for column in [
            "date",
            "store",
            "category",
            "region",
            "channel",
            "product",
            "sales_amount",
            "gross_profit",
            "sales_qty",
        ]
        if column in df.columns
    ]

    frames = []
    for selection in selections:
        custom_data = selection.get("customdata") if isinstance(selection, dict) else None
        if not custom_data:
            continue
        period_key = custom_data[0]
        start, end = period_bounds(period_key, granularity)
        mask = (df["date"] >= start) & (df["date"] <= end)
        segment_value = None
        if breakdown_column and len(custom_data) > 1:
            segment_value = custom_data[1]
            mask &= df[breakdown_column] == segment_value
        subset = df.loc[mask, detail_columns].copy()
        if subset.empty:
            continue
        subset["期間"] = _format_period_label(start, granularity)
        if segment_value is not None:
            subset["分析軸"] = segment_value
        if "sales_amount" in subset.columns and "gross_profit" in subset.columns:
            subset["粗利率"] = (
                subset["gross_profit"] / subset["sales_amount"].replace(0, pd.NA)
            ).fillna(0.0)
        frames.append(subset)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames).sort_values("date").reset_index(drop=True)
    rename_map = {
        "date": "日付",
        "store": "店舗",
        "category": "カテゴリ",
        "region": "地域",
        "channel": "チャネル",
        "product": "商品",
        "sales_amount": "売上",
        "gross_profit": "粗利",
        "sales_qty": "販売数量",
    }
    combined = combined.rename(columns=rename_map)
    return combined


def breakdown_summary(
    current: pd.DataFrame,
    comparison: Optional[pd.DataFrame],
    breakdown: Optional[str],
) -> pd.DataFrame:
    column = resolve_breakdown_column(breakdown)
    if column is None:
        return pd.DataFrame()
    if current.empty:
        columns = [
            column,
            "sales_amount",
            "gross_profit",
            "gross_margin",
            "comparison_sales",
            "comparison_gross_profit",
            "comparison_margin",
            "yoy_rate",
            "gross_profit_yoy",
            "margin_delta",
        ]
        return pd.DataFrame(columns=columns)

    aggregated = (
        current.groupby(column)[["sales_amount", "gross_profit", "sales_qty"]]
        .sum()
        .reset_index()
    )
    aggregated["gross_margin"] = (
        aggregated["gross_profit"] / aggregated["sales_amount"].replace(0, pd.NA)
    ).fillna(0.0)

    if comparison is None or comparison.empty:
        for column_name in [
            "comparison_sales",
            "comparison_gross_profit",
            "comparison_margin",
            "yoy_rate",
            "gross_profit_yoy",
            "margin_delta",
        ]:
            aggregated[column_name] = pd.NA
        return aggregated.sort_values("sales_amount", ascending=False)

    comparison_agg = (
        comparison.groupby(column)[["sales_amount", "gross_profit"]]
        .sum()
        .reset_index()
        .rename(
            columns={
                "sales_amount": "comparison_sales",
                "gross_profit": "comparison_gross_profit",
            }
        )
    )
    comparison_agg["comparison_margin"] = (
        comparison_agg["comparison_gross_profit"]
        / comparison_agg["comparison_sales"].replace(0, pd.NA)
    )

    merged = aggregated.merge(comparison_agg, on=column, how="left")
    for column_name in [
        "comparison_sales",
        "comparison_gross_profit",
        "comparison_margin",
    ]:
        merged[column_name] = merged[column_name].astype(float)

    sales_base = merged["comparison_sales"].replace(0, pd.NA)
    merged["yoy_rate"] = (merged["sales_amount"] - merged["comparison_sales"]) / sales_base
    merged.loc[sales_base.isna(), "yoy_rate"] = pd.NA

    profit_base = merged["comparison_gross_profit"].replace(0, pd.NA)
    merged["gross_profit_yoy"] = (
        merged["gross_profit"] - merged["comparison_gross_profit"]
    ) / profit_base
    merged.loc[profit_base.isna(), "gross_profit_yoy"] = pd.NA

    merged["margin_delta"] = merged["gross_margin"] - merged["comparison_margin"]
    merged.loc[merged["comparison_margin"].isna(), "margin_delta"] = pd.NA
    return merged.sort_values("sales_amount", ascending=False)
