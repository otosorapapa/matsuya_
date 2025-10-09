"""Rule-based decision support alerts for inventory and cash management."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class DecisionAlert:
    """Represents an actionable alert presented to the dashboard user."""

    title: str
    message: str
    severity: str
    recommendations: List[str]
    category: str


def _format_currency(value: float) -> str:
    return f"{value:,.0f}円"


def inventory_alerts(overview: pd.DataFrame) -> List[DecisionAlert]:
    if overview.empty:
        return []

    alerts: List[DecisionAlert] = []
    stockout_df = overview.loc[overview["stock_status"].isin(["在庫切れ", "在庫少"])]
    if not stockout_df.empty:
        items = stockout_df[["store", "product", "estimated_stock", "safety_lower"]]
        lines = []
        for _, row in items.head(10).iterrows():
            store = row.get("store") or "-"
            product = row.get("product") or "-"
            estimated = float(row.get("estimated_stock", 0))
            threshold = float(row.get("safety_lower", 0))
            lines.append(
                f"{store} - {product}: 在庫 {estimated:.0f} / 目標 {threshold:.0f}"
            )
        alerts.append(
            DecisionAlert(
                title="安全在庫を下回る品目があります",
                message=(
                    f"{len(stockout_df)}品目が安全在庫を下回っています。"
                    " 補充発注を優先し、需要予測の見直しを実施してください。"
                ),
                severity="warning",
                recommendations=lines,
                category="inventory",
            )
        )

    overstock_df = overview.loc[overview["stock_status"] == "在庫過多"]
    if not overstock_df.empty:
        lines = []
        for _, row in overstock_df.head(10).iterrows():
            store = row.get("store") or "-"
            product = row.get("product") or "-"
            coverage = row.get("coverage_days")
            if pd.isna(coverage):
                continue
            lines.append(f"{store} - {product}: 在庫日数 {coverage:.1f}日")
        alerts.append(
            DecisionAlert(
                title="過剰在庫が検出されました",
                message=(
                    f"{len(overstock_df)}品目が安全在庫上限を超過しています。"
                    " 仕入数量の調整や販促による消化を検討してください。"
                ),
                severity="info",
                recommendations=lines,
                category="inventory",
            )
        )
    return alerts


def cash_flow_alerts(
    forecast_df: pd.DataFrame,
    *,
    target_balance: Optional[float] = None,
) -> List[DecisionAlert]:
    if forecast_df.empty:
        return []

    alerts: List[DecisionAlert] = []
    negative = forecast_df.loc[forecast_df["balance"] < 0]
    if not negative.empty:
        first = negative.iloc[0]
        alerts.append(
            DecisionAlert(
                title="資金ショートのリスクがあります",
                message=(
                    f"{first['period_label']}に資金残高がマイナス({_format_currency(first['balance'])})"
                    " となる見込みです。入金前倒しや短期借入を検討してください。"
                ),
                severity="danger",
                recommendations=[
                    "販管費の支出を見直しキャッシュアウトを抑制",
                    "入金サイト短縮やファクタリングの検討",
                    "不足額に応じた短期借入や資金移動の準備",
                ],
                category="cash",
            )
        )

    if target_balance is not None:
        below_target = forecast_df.loc[forecast_df["balance"] < target_balance]
        if not below_target.empty:
            upcoming = below_target.iloc[0]
            alerts.append(
                DecisionAlert(
                    title="キャッシュ比率が目標を下回る見込みです",
                    message=(
                        f"{upcoming['period_label']}の残高は{_format_currency(upcoming['balance'])}で"
                        f" 目標 {_format_currency(target_balance)} を下回ります。"
                    ),
                    severity="warning",
                    recommendations=[
                        "利益率の高い商品の拡販で粗利を底上げ",
                        "在庫回転を改善し現金化を加速",
                        "不要資産の売却や返済スケジュールの再交渉",
                    ],
                    category="cash",
                )
            )
    return alerts


def cvp_alerts(
    sales_df: pd.DataFrame,
    fixed_cost_df: pd.DataFrame,
) -> List[DecisionAlert]:
    if sales_df.empty or fixed_cost_df.empty:
        return []

    total_sales = float(sales_df["sales_amount"].sum())
    total_gross_profit = float(sales_df["gross_profit"].sum())
    contribution_margin_rate = total_gross_profit / total_sales if total_sales else 0.0
    fixed_cost_columns = [
        column
        for column in ["rent", "payroll", "utilities", "marketing", "other_fixed"]
        if column in fixed_cost_df.columns
    ]
    fixed_cost = float(fixed_cost_df[fixed_cost_columns].sum().sum())
    if contribution_margin_rate <= 0:
        break_even = float("inf")
    else:
        break_even = fixed_cost / contribution_margin_rate

    alerts: List[DecisionAlert] = []
    if total_sales < break_even and break_even != float("inf"):
        gap = break_even - total_sales
        alerts.append(
            DecisionAlert(
                title="損益分岐点を下回っています",
                message=(
                    f"現在の売上 { _format_currency(total_sales) } は"
                    f" 損益分岐点 {_format_currency(break_even)} を { _format_currency(gap) } 下回ります。"
                ),
                severity="danger",
                recommendations=[
                    "高粗利商品の販売強化による貢献利益率の改善",
                    "固定費の見直し・削減案の検討",
                    "チャネル別の利益構造を分析し不採算を縮小",
                ],
                category="profitability",
            )
        )
    return alerts


def collect_alerts(
    inventory_df: Optional[pd.DataFrame] = None,
    cash_forecast_df: Optional[pd.DataFrame] = None,
    *,
    sales_df: Optional[pd.DataFrame] = None,
    fixed_cost_df: Optional[pd.DataFrame] = None,
    cash_target: Optional[float] = None,
) -> List[DecisionAlert]:
    alerts: List[DecisionAlert] = []
    if inventory_df is not None:
        alerts.extend(inventory_alerts(inventory_df))
    if cash_forecast_df is not None:
        alerts.extend(cash_flow_alerts(cash_forecast_df, target_balance=cash_target))
    if sales_df is not None and fixed_cost_df is not None:
        alerts.extend(cvp_alerts(sales_df, fixed_cost_df))
    return alerts


__all__ = [
    "DecisionAlert",
    "inventory_alerts",
    "cash_flow_alerts",
    "cvp_alerts",
    "collect_alerts",
]
