"""Streamlit entry point for the Matsya management dashboard."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app import data_loader, transformers
from streamlit_app.analytics import inventory, products, profitability, sales, simulation
from streamlit_app.components import report, sidebar


st.set_page_config(page_title="松屋 計数管理ダッシュボード", layout="wide")


def _to_csv_source(upload) -> Optional[bytes]:
    if upload is None:
        return None
    upload.seek(0)
    return upload.read()


@st.cache_data(show_spinner=False)
def load_datasets(
    sales_source,
    inventory_source,
    fixed_cost_source,
) -> Dict[str, pd.DataFrame]:
    sales_df = data_loader.load_sales_data(sales_source)
    sales_df = transformers.prepare_sales_dataset(sales_df)
    inventory_df = data_loader.load_inventory_data(inventory_source)
    fixed_cost_df = data_loader.load_fixed_costs(fixed_cost_source)
    return {
        "sales": sales_df,
        "inventory": inventory_df,
        "fixed_costs": fixed_cost_df,
    }


def _default_period(df: pd.DataFrame) -> tuple[date, date]:
    if df.empty:
        today = date.today()
        return today.replace(day=1), today
    max_date = df["date"].max().date()
    min_date = df["date"].min().date()
    start = max(min_date, max_date - timedelta(days=30))
    return start, max_date


def _comparison_dataset(
    df: pd.DataFrame, filters: transformers.FilterState, mode: str
) -> pd.DataFrame:
    if df.empty:
        return df.head(0)
    if mode == "yoy":
        comparison_filters = transformers.compute_comparison_period(filters)
        return transformers.apply_filters(df, comparison_filters)
    if mode == "previous_period":
        period_days = (filters.end_date - filters.start_date).days + 1
        prev_end = filters.start_date - timedelta(days=1)
        prev_start = prev_end - timedelta(days=period_days - 1)
        comparison_filters = transformers.FilterState(
            store=filters.store,
            start_date=prev_start,
            end_date=prev_end,
            category=filters.category,
        )
        return transformers.apply_filters(df, comparison_filters)
    return df.head(0)


def render_sales_tab(
    filtered_sales: pd.DataFrame,
    comparison_sales: pd.DataFrame,
    filters: transformers.FilterState,
) -> None:
    if filtered_sales.empty:
        st.info("該当データがありません")
        return

    kpis = sales.kpi_summary(filtered_sales, comparison_sales)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "期間売上",
        f"{kpis['total_sales']:,.0f} 円",
        yoy_value if kpis["yoy_rate"] is not None else None,
    )
    yoy_value = (
        f"{kpis['yoy_rate']*100:.1f}%" if kpis["yoy_rate"] is not None else "-"
    )
    col2.metric("粗利", f"{kpis['total_gross_profit']:,.0f} 円")
    col3.metric("平均客単価", f"{kpis['avg_unit_price']:,.0f} 円")
    col4.metric("来客数(販売数量)", f"{kpis['total_customers']:,.0f}")
    col5.metric("粗利率", f"{kpis['gross_margin']*100:.1f}%")

    daily_df = sales.daily_timeseries(filtered_sales)
    st.subheader("日次売上推移")
    daily_chart = px.line(
        daily_df,
        x="date",
        y="sales_amount",
        markers=True,
        labels={"date": "日付", "sales_amount": "売上金額"},
    )
    daily_chart.update_layout(hovermode="x unified")
    st.plotly_chart(daily_chart, use_container_width=True)

    st.subheader("月次売上と前年同月比")
    monthly_df = sales.monthly_performance(filtered_sales, comparison_sales)
    fig = go.Figure()
    fig.add_bar(
        x=monthly_df["month_label"],
        y=monthly_df["sales_amount"],
        name="当期売上",
    )
    if monthly_df["yoy_sales"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=monthly_df["month_label"],
                y=monthly_df["yoy_sales"],
                name="前年同月売上",
                mode="lines+markers",
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis=dict(title="当期売上"),
            yaxis2=dict(
                title="前年同月売上",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("カテゴリ別売上構成")
    category_df = sales.sales_by_category(filtered_sales)
    pie_chart = px.pie(
        category_df,
        names="category",
        values="sales_amount",
        color_discrete_sequence=px.colors.sequential.Blues,
    )
    st.plotly_chart(pie_chart, use_container_width=True)

    st.dataframe(category_df, use_container_width=True)


def render_products_tab(
    filtered_sales: pd.DataFrame,
    comparison_sales: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    st.subheader("ABC分析")
    abc_df = products.abc_analysis(filtered_sales, comparison_sales)
    if abc_df.empty:
        st.info("該当データがありません")
        return abc_df, abc_df

    pareto_df = products.pareto_chart_data(abc_df)
    pareto_chart = go.Figure()
    pareto_chart.add_bar(
        x=pareto_df["product"],
        y=pareto_df["sales_amount"],
        name="売上",
    )
    pareto_chart.add_trace(
        go.Scatter(
            x=pareto_df["product"],
            y=pareto_df["cumulative_share"] * 100,
            mode="lines+markers",
            name="累積構成比 (%)",
            yaxis="y2",
        )
    )
    pareto_chart.update_layout(
        yaxis=dict(title="売上"),
        yaxis2=dict(
            title="累積構成比(%)",
            overlaying="y",
            side="right",
        ),
        hovermode="x unified",
    )
    st.plotly_chart(pareto_chart, use_container_width=True)

    st.dataframe(abc_df, use_container_width=True)

    st.subheader("Aランク商品の伸長率トップ3")
    top_growth = products.top_growth_products(abc_df)
    if top_growth.empty:
        st.info("前年データが不足しているため、伸長率を計算できません。")
    else:
        cols = st.columns(len(top_growth))
        for col, (_, row) in zip(cols, top_growth.iterrows()):
            col.metric(
                row["product"],
                f"{row['sales_amount']:,.0f} 円",
                f"{row['yoy_growth']*100:.1f}%",
            )
    return abc_df, top_growth


def render_profitability_tab(
    filtered_sales: pd.DataFrame,
    fixed_costs_df: pd.DataFrame,
) -> pd.DataFrame:
    st.subheader("固定費内訳調整")
    editable = fixed_costs_df.set_index("store")
    edited = st.data_editor(
        editable,
        num_rows="dynamic",
        use_container_width=True,
        key="fixed_cost_editor",
    )
    edited_df = edited.reset_index()
    cost_columns = ["rent", "payroll", "utilities", "marketing", "other_fixed"]
    for col in cost_columns:
        edited_df[col] = pd.to_numeric(edited_df[col], errors="coerce").fillna(0)

    st.subheader("店舗別損益表")
    pnl_df = profitability.store_profitability(filtered_sales, edited_df)
    styled = pnl_df.style.format(
        {
            "sales_amount": "{:,.0f}",
            "gross_profit": "{:,.0f}",
            "total_fixed_cost": "{:,.0f}",
            "operating_profit": "{:,.0f}",
            "gross_margin": "{:.1%}",
            "operating_margin": "{:.1%}",
        }
    ).apply(
        lambda s: ["background-color: #ffe5e5" if v < 0 else "" for v in s]
        if s.name == "operating_profit"
        else [""] * len(s),
        axis=0,
    )
    st.dataframe(styled, use_container_width=True)

    negative = pnl_df[pnl_df["operating_profit"] < 0]
    if not negative.empty:
        st.warning("赤字店舗があります。詳細を確認してください。")

    chart_df = profitability.profitability_chart_data(pnl_df)
    chart = px.bar(
        chart_df,
        x="store",
        y=["sales_amount", "gross_profit", "operating_profit"],
        barmode="group",
        labels={"value": "金額", "variable": "指標"},
    )
    st.plotly_chart(chart, use_container_width=True)
    return pnl_df


def render_inventory_tab(
    filtered_sales: pd.DataFrame,
    inventory_df: pd.DataFrame,
    abc_df: pd.DataFrame,
    filters: transformers.FilterState,
) -> None:
    st.subheader("在庫推定とアドバイス")
    if filters.store != transformers.ALL_STORES:
        inventory_df = inventory_df[inventory_df["store"] == filters.store]
    if filters.category != transformers.ALL_CATEGORIES:
        inventory_df = inventory_df[inventory_df["category"] == filters.category]
    overview_df = inventory.inventory_overview(filtered_sales, inventory_df)
    if overview_df.empty:
        st.info("在庫データが見つかりません。")
        return
    rank_lookup = dict(zip(abc_df["product"], abc_df.get("rank", [])))
    advice_df = inventory.inventory_advice(overview_df, rank_lookup)
    st.dataframe(advice_df, use_container_width=True)

    period_days = (filters.end_date - filters.start_date).days + 1
    turnover_df = inventory.turnover_by_category(overview_df, period_days=period_days)
    st.subheader("カテゴリ別在庫回転率")
    st.dataframe(turnover_df, use_container_width=True)
    heatmap = px.density_heatmap(
        advice_df,
        x="category",
        y="store",
        z="estimated_stock",
        color_continuous_scale="Blues",
        labels={"estimated_stock": "推定在庫"},
    )
    st.plotly_chart(heatmap, use_container_width=True)


def render_simulation_tab(
    pnl_df: pd.DataFrame,
    filters: transformers.FilterState,
) -> None:
    st.subheader("損益シミュレーション")
    total_sales = pnl_df["sales_amount"].sum()
    total_gross_profit = pnl_df["gross_profit"].sum()
    default_margin = total_gross_profit / total_sales if total_sales else 0.45
    default_margin = float(min(max(default_margin, 0.3), 0.8))
    default_fixed_cost = pnl_df["total_fixed_cost"].sum()

    col1, col2, col3 = st.columns(3)
    gross_margin = col1.slider("粗利率", min_value=0.3, max_value=0.8, value=float(round(default_margin, 2)), step=0.01)
    fixed_cost = col2.number_input("固定費合計", min_value=0.0, value=float(default_fixed_cost), step=10000.0)
    target_profit = col3.number_input("目標利益", min_value=0.0, value=500000.0, step=50000.0)

    inputs = simulation.SimulationInputs(
        gross_margin=gross_margin,
        fixed_cost=fixed_cost,
        target_profit=target_profit,
    )
    requirements = simulation.required_sales(inputs)

    st.metric("損益分岐点売上", f"{requirements['breakeven']:,.0f} 円")
    st.metric("目標利益達成に必要な売上", f"{requirements['target_sales']:,.0f} 円")

    curve = simulation.breakeven_sales_curve(simulation.DEFAULT_MARGIN_RANGE, fixed_cost)
    curve_chart = px.line(
        curve,
        x="gross_margin",
        y="breakeven_sales",
        labels={"gross_margin": "粗利率", "breakeven_sales": "損益分岐点売上"},
    )
    st.plotly_chart(curve_chart, use_container_width=True)

    st.subheader("シナリオ保存")
    scenario_name = st.text_input("シナリオ名", value=f"{filters.store}_{filters.start_date:%Y%m%d}")
    if st.button("シナリオを保存"):
        scenario = {
            "name": scenario_name,
            "gross_margin": gross_margin,
            "fixed_cost": fixed_cost,
            "target_profit": target_profit,
            "breakeven": requirements["breakeven"],
            "target_sales": requirements["target_sales"],
        }
        scenarios = st.session_state.setdefault("saved_scenarios", [])
        scenarios.append(scenario)
        st.success("シナリオを保存しました。")

    if st.session_state.get("saved_scenarios"):
        scenarios_df = pd.DataFrame(st.session_state["saved_scenarios"])
        st.dataframe(scenarios_df, use_container_width=True)
        csv_bytes = scenarios_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "シナリオ一覧をダウンロード",
            data=csv_bytes,
            file_name="matsuya_simulation_scenarios.csv",
        )


def main() -> None:
    st.title("松屋 計数管理ダッシュボード")

    sample_files = data_loader.available_sample_files()

    datasets = load_datasets(
        _to_csv_source(None),
        _to_csv_source(None),
        _to_csv_source(None),
    )

    default_period = _default_period(datasets["sales"])
    stores = transformers.extract_stores(datasets["sales"])
    categories = transformers.extract_categories(datasets["sales"])

    sidebar_state = sidebar.render_sidebar(
        stores,
        categories,
        default_period=default_period,
        sample_files={k: str(v) for k, v in sample_files.items()},
    )

    uploads = sidebar_state["uploads"]
    datasets = load_datasets(
        _to_csv_source(uploads["sales"]),
        _to_csv_source(uploads["inventory"]),
        _to_csv_source(uploads["fixed_costs"]),
    )

    filters = sidebar_state["filters"]
    filtered_sales = transformers.apply_filters(datasets["sales"], filters)
    comparison_mode = sidebar_state["comparison_mode"]
    comparison_sales = _comparison_dataset(datasets["sales"], filters, comparison_mode)

    tabs = st.tabs([
        "売上分析",
        "商品別分析",
        "損益管理",
        "在庫分析",
        "経営シミュレーション",
    ])

    with tabs[0]:
        render_sales_tab(filtered_sales, comparison_sales, filters)

    with tabs[1]:
        abc_df, _ = render_products_tab(filtered_sales, comparison_sales)

    with tabs[2]:
        pnl_df = render_profitability_tab(filtered_sales, datasets["fixed_costs"])

    with tabs[3]:
        render_inventory_tab(filtered_sales, datasets["inventory"], abc_df, filters)

    with tabs[4]:
        render_simulation_tab(pnl_df, filters)

    with st.sidebar:
        if sidebar_state["export_csv"]:
            report.csv_download(
                "売上データをCSV出力",
                filtered_sales,
                file_name=f"matsuya_sales_{filters.start_date:%Y%m%d}_{filters.end_date:%Y%m%d}.csv",
            )
        if sidebar_state["export_pdf"]:
            report.pdf_download(
                "売上データをPDF出力",
                "売上サマリー",
                filtered_sales[["date", "store", "category", "sales_amount", "gross_profit"]],
                file_name=f"matsuya_sales_{filters.start_date:%Y%m%d}.pdf",
            )


if __name__ == "__main__":
    main()
