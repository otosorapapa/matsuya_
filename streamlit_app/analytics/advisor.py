"""Rule-based advisor that surfaces narrative insights for managers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd


@dataclass
class AdvisorContext:
    """Key metrics required to evaluate management insights."""

    total_sales: float
    operating_profit: float
    stockout_items: int
    cash_balance: float
    target_profit: float


@dataclass
class AdvisorInsight:
    """Structured insight returned by the advisor."""

    title: str
    description: str
    severity: str = "info"
    tags: Iterable[str] = ()


def _locate_benchmark(
    benchmark_df: Optional[pd.DataFrame], key_candidates: Iterable[str]
) -> Optional[pd.Series]:
    if benchmark_df is None or benchmark_df.empty:
        return None
    normalized = benchmark_df.copy()
    if "metric" not in normalized.columns:
        return None
    normalized["metric"] = normalized["metric"].astype(str).str.lower()
    for key in key_candidates:
        match = normalized.loc[normalized["metric"] == key.lower()]
        if not match.empty:
            return match.iloc[0]
    return None


def generate_advice(
    context: AdvisorContext,
    *,
    benchmark_df: Optional[pd.DataFrame] = None,
    monte_carlo_probability: Optional[float] = None,
    expected_profit: Optional[float] = None,
    sensitivity_df: Optional[pd.DataFrame] = None,
) -> List[AdvisorInsight]:
    """Generate qualitative advice based on the analytical outputs."""

    insights: List[AdvisorInsight] = []
    operating_margin = (
        context.operating_profit / context.total_sales if context.total_sales else 0.0
    )

    if monte_carlo_probability is not None:
        if monte_carlo_probability < 0.3:
            insights.append(
                AdvisorInsight(
                    title="目標利益の達成確率が低下しています",
                    description=(
                        f"モンテカルロ試算の結果、目標利益の達成確率は{monte_carlo_probability:.0%}です。"
                        " コスト構造の見直しや販促強化など、追加の対策を即時検討してください。"
                    ),
                    severity="danger",
                    tags=["リスク", "確率分析"],
                )
            )
        elif monte_carlo_probability < 0.55:
            insights.append(
                AdvisorInsight(
                    title="目標利益達成に向けた改善余地があります",
                    description=(
                        f"達成確率は{monte_carlo_probability:.0%}です。粗利率の底上げや固定費の圧縮を行えば"
                        " 達成確度を高められます。"
                    ),
                    severity="warning",
                    tags=["モンテカルロ", "利益"],
                )
            )
        elif monte_carlo_probability > 0.75:
            insights.append(
                AdvisorInsight(
                    title="利益目標の達成見込みは良好です",
                    description=(
                        f"達成確率が{monte_carlo_probability:.0%}まで向上しています。"
                        " 計画通りの在庫確保と売上維持に注力しましょう。"
                    ),
                    severity="success",
                    tags=["好調", "予測"],
                )
            )

    operating_margin_benchmark = _locate_benchmark(
        benchmark_df, ["operating_margin", "営業利益率"]
    )
    if operating_margin_benchmark is not None:
        industry_margin = float(operating_margin_benchmark.get("industry_avg", 0.0))
        if operating_margin + 1e-6 < industry_margin:
            diff = (industry_margin - operating_margin) * 100
            insights.append(
                AdvisorInsight(
                    title="営業利益率が業界平均を下回っています",
                    description=(
                        f"現在の営業利益率は{operating_margin:.1%}で、業界平均({industry_margin:.1%})より"
                        f" {diff:.1f}ポイント低い水準です。粗利率向上のための値付け見直しや、固定費の最適化を検討してください。"
                    ),
                    severity="warning",
                    tags=["ベンチマーク", "利益率"],
                )
            )
        elif operating_margin > industry_margin:
            insights.append(
                AdvisorInsight(
                    title="営業利益率は業界平均を上回っています",
                    description=(
                        f"営業利益率は{operating_margin:.1%}で、業界平均({industry_margin:.1%})を上回っています。"
                        " 優位性を維持するため、在庫効率化と販促施策の継続を推奨します。"
                    ),
                    severity="success",
                    tags=["ベンチマーク", "利益率"],
                )
            )

    if context.stockout_items > 0:
        insights.append(
            AdvisorInsight(
                title="欠品リスクが顕在化しています",
                description=(
                    f"安全在庫を下回る品目が{context.stockout_items}件あります。"
                    " 発注サイクルの短縮や需要予測の更新を実施してください。"
                ),
                severity="warning",
                tags=["在庫", "需要予測"],
            )
        )

    if context.cash_balance < context.target_profit:
        insights.append(
            AdvisorInsight(
                title="現預金残高が目標利益を下回っています",
                description=(
                    f"現預金残高は{context.cash_balance:,.0f}円で、目標利益{context.target_profit:,.0f}円を下回ります。"
                    " 回収サイトの短縮や支払条件の見直しで資金繰りを改善しましょう。"
                ),
                severity="warning",
                tags=["資金繰り", "キャッシュフロー"],
            )
        )

    if expected_profit is not None and expected_profit < context.target_profit:
        diff = context.target_profit - expected_profit
        insights.append(
            AdvisorInsight(
                title="予測営業利益が目標を下回っています",
                description=(
                    f"シミュレーション上の平均営業利益は{expected_profit:,.0f}円で、目標利益を{diff:,.0f}円下回ります。"
                    " 価格改定や高粗利商品の販促強化を検討してください。"
                ),
                severity="danger",
                tags=["利益", "シミュレーション"],
            )
        )

    if sensitivity_df is not None and not sensitivity_df.empty:
        margin_df = sensitivity_df[sensitivity_df["parameter"] == "gross_margin"]
        if not margin_df.empty:
            upside = margin_df[margin_df["change_pct"] > 0].tail(1)
            if not upside.empty:
                gap = float(upside.iloc[0]["gap_to_target"])
                margin_change = float(upside.iloc[0]["change_pct"]) * 100
                if gap >= 0:
                    insights.append(
                        AdvisorInsight(
                            title="粗利率改善で目標達成が可能です",
                            description=(
                                f"粗利率を{margin_change:.1f}%ポイント改善すると、目標利益を達成できる見込みです。"
                                " 高付加価値商品の販売強化や仕入れ条件の交渉を行いましょう。"
                            ),
                            severity="info",
                            tags=["感度分析", "粗利"],
                        )
                    )

        cost_df = sensitivity_df[sensitivity_df["parameter"] == "fixed_cost"]
        if not cost_df.empty:
            worst = cost_df.sort_values("change_pct").head(1)
            if not worst.empty and float(worst.iloc[0]["gap_to_target"]) < 0:
                insights.append(
                    AdvisorInsight(
                        title="固定費の上振れに注意が必要です",
                        description=(
                            "固定費が増加すると目標利益からの乖離が拡大します。"
                            " 契約更改前のコスト見直しや省エネ施策を検討してください。"
                        ),
                        severity="warning",
                        tags=["固定費", "感度分析"],
                    )
                )

    if not insights:
        insights.append(
            AdvisorInsight(
                title="主要指標は健全に推移しています",
                description=(
                    "現在、重大なリスクは検出されていません。日次モニタリングを継続してください。"
                ),
                severity="success",
                tags=["安定運用"],
            )
        )

    return insights


__all__ = ["AdvisorContext", "AdvisorInsight", "generate_advice"]
