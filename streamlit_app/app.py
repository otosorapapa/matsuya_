"""Streamlit entry point for the Matsya management dashboard."""
from __future__ import annotations

import hashlib
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from streamlit_app import data_loader, transformers
from streamlit_app.analytics import inventory, products, profitability, sales, simulation
from streamlit_app.components import import_dashboard, report, sidebar
from streamlit_app.integrations import IntegrationResult, available_providers, fetch_datasets


logger = logging.getLogger(__name__)


st.set_page_config(page_title="松屋 計数管理ダッシュボード", layout="wide")
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


def _default_period(df: pd.DataFrame) -> Tuple[date, date]:
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


_ERROR_COLUMNS = ["行番号", "列名", "内容"]

_DATASET_CONFIGS: Dict[str, Tuple[str, Callable[[Optional[object]], pd.DataFrame], str]] = {
    "sales": ("validate_sales_csv", data_loader.load_sales_data, "売上"),
    "inventory": ("validate_inventory_csv", data_loader.load_inventory_data, "仕入/在庫"),
    "fixed_costs": ("validate_fixed_costs_csv", data_loader.load_fixed_costs, "固定費"),
}


def _fallback_validator(
    loader: Callable[[Optional[object]], pd.DataFrame],
    dataset_label: str,
) -> Callable[[Optional[object]], data_loader.ValidationResult]:
    """Return a basic validator when dedicated helpers are unavailable."""

    def _validator(source: Optional[object]) -> data_loader.ValidationResult:
        try:
            dataframe = loader(source)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("%s CSV validation failed with fallback", dataset_label)
            try:
                empty_df = loader(None).head(0)
            except Exception:  # pragma: no cover - defensive path
                empty_df = pd.DataFrame()
            errors = pd.DataFrame(
                [
                    {
                        "行番号": "全体",
                        "列名": "-",
                        "内容": f"{dataset_label}のCSV読込でエラーが発生しました: {exc}",
                    }
                ],
                columns=_ERROR_COLUMNS,
            )
            return data_loader.ValidationResult(
                dataframe=empty_df,
                errors=errors,
                valid=False,
                total_rows=0,
                dropped_rows=0,
            )

        return data_loader.ValidationResult(
            dataframe=dataframe,
            errors=pd.DataFrame(columns=_ERROR_COLUMNS),
            valid=True,
            total_rows=len(dataframe),
            dropped_rows=0,
        )

    return _validator


def _build_csv_validators() -> Dict[str, Callable[[Optional[object]], data_loader.ValidationResult]]:
    validators: Dict[str, Callable[[Optional[object]], data_loader.ValidationResult]] = {}
    for dataset, (validator_name, loader, label) in _DATASET_CONFIGS.items():
        validator = getattr(data_loader, validator_name, None)
        if validator is None:
            logger.warning(
                "data_loader.%s is missing; using fallback validator for '%s' dataset",
                validator_name,
                dataset,
            )
            validator = _fallback_validator(loader, label)
        validators[dataset] = validator
    return validators


CSV_VALIDATORS = _build_csv_validators()


def _copy_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {name: df.copy() for name, df in datasets.items()}


def _hash_bytes(payload: bytes) -> str:
    return hashlib.md5(payload).hexdigest()


def _handle_csv_uploads(
    uploads: Dict[str, Optional[object]],
    baseline: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, data_loader.ValidationResult]]:
    datasets = _copy_datasets(baseline)
    validations: Dict[str, data_loader.ValidationResult] = {}

    for dataset, upload in uploads.items():
        if upload is None:
            continue
        validator = CSV_VALIDATORS.get(dataset)
        if validator is None:
            continue
        file_bytes = upload.getvalue()
        validation = validator(file_bytes)
        validations[dataset] = validation

        record_id = f"csv::{dataset}::{_hash_bytes(file_bytes)}"
        source_label = f"CSV: {getattr(upload, 'name', 'uploaded_file')}"
        import_dashboard.record_validation_import(
            dataset,
            source_label,
            validation,
            record_id=record_id,
        )

        if validation.valid:
            datasets[dataset] = validation.dataframe

    return datasets, validations


def _handle_api_mode(
    api_state: Dict[str, object],
    baseline: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], Optional[IntegrationResult]]:
    provider = api_state.get("provider")
    stored = st.session_state.get("api_datasets")
    datasets = _copy_datasets(stored or baseline)
    integration_result: Optional[IntegrationResult] = None

    if not provider:
        return datasets, st.session_state.get("latest_api_result")

    credentials = {
        key: value
        for key, value in {
            "api_key": api_state.get("api_key"),
            "api_secret": api_state.get("api_secret"),
        }.items()
        if value
    }

    if api_state.get("auto_daily"):
        auto_result = _maybe_auto_fetch(provider, credentials)
        if auto_result:
            _log_integration_result(auto_result)
            datasets = _copy_datasets(auto_result.datasets)
            st.session_state["api_datasets"] = datasets
            st.session_state["latest_api_result"] = auto_result
            integration_result = auto_result

    if api_state.get("fetch_triggered"):
        start_date = api_state.get("start_date")
        end_date = api_state.get("end_date")
        if isinstance(start_date, date) and isinstance(end_date, date):
            manual_result = fetch_datasets(provider, start_date, end_date, credentials)
            _log_integration_result(manual_result)
            datasets = _copy_datasets(manual_result.datasets)
            st.session_state["api_datasets"] = datasets
            st.session_state["latest_api_result"] = manual_result
            integration_result = manual_result

    if integration_result is None:
        stored_result = st.session_state.get("latest_api_result")
        if stored_result is not None:
            datasets = _copy_datasets(st.session_state.get("api_datasets", datasets))
        integration_result = stored_result

    return datasets, integration_result


def _maybe_auto_fetch(
    provider: str, credentials: Dict[str, str]
) -> Optional[IntegrationResult]:
    target_date = date.today() - timedelta(days=1)
    state_key = f"auto_fetch::{provider}"
    if st.session_state.get(state_key) == str(target_date):
        return None
    result = fetch_datasets(provider, target_date, target_date, credentials)
    st.session_state[state_key] = str(target_date)
    return result


def _log_integration_result(result: IntegrationResult) -> None:
    source_label = f"API: {result.provider}"
    for dataset, df in result.datasets.items():
        record_id = (
            f"api::{result.provider}::{dataset}::{result.start_date:%Y%m%d}_{result.end_date:%Y%m%d}"
            f"::{int(result.retrieved_at.timestamp())}"
        )
        import_dashboard.record_api_import(
            dataset,
            source_label,
            df,
            total_rows=result.row_counts.get(dataset, len(df)),
            notes=result.period_label(),
            record_id=record_id,
        )


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
    yoy_delta = (
        f"{kpis['yoy_rate']*100:.1f}%" if kpis["yoy_rate"] is not None else None
    )
    col1.metric(
        "期間売上",
        f"{kpis['total_sales']:,.0f} 円",
        yoy_delta,
    )
    col2.metric("粗利", f"{kpis['total_gross_profit']:,.0f} 円")
    col3.metric("平均客単価", f"{kpis['avg_unit_price']:,.0f} 円")
    col4.metric("来客数(販売数量)", f"{kpis['total_customers']:,.0f}")
    col5.metric("粗利率", f"{kpis['gross_margin']*100:.1f}%")

    granularity = filters.period_granularity
    breakdown_column = sales.resolve_breakdown_column(filters.breakdown_dimension)
    breakdown_label = sales.breakdown_label(filters.breakdown_dimension)
    if breakdown_column is None:
        breakdown_label = "全体"

    timeseries_df = sales.timeseries_with_comparison(
        filtered_sales, comparison_sales, granularity, breakdown_column
    )
    custom_data = ["period_key"]
    if breakdown_column:
        custom_data.append(breakdown_column)

    st.subheader(
        f"{sales.granularity_label(granularity)}売上推移（{breakdown_label}）"
    )
    timeseries_chart = px.line(
        timeseries_df,
        x="period_label",
        y="sales_amount",
        color=breakdown_column if breakdown_column else None,
        markers=True,
        labels={"period_label": "期間", "sales_amount": "売上"},
        custom_data=custom_data,
    )
    timeseries_chart.update_layout(
        hovermode="x unified",
        clickmode="event+select",
        yaxis_title="売上",
        xaxis_title="期間",
        legend_title=breakdown_label if breakdown_column else None,
    )
    timeseries_chart.update_traces(mode="lines+markers")
    selection = plotly_events(
        timeseries_chart,
        click_event=True,
        select_event=True,
        key=f"sales-timeseries-{granularity}-{breakdown_column or 'total'}",
    )
    st.caption("グラフをクリックすると該当期間の明細が表示されます。")

    def _highlight_growth(value: float) -> str:
        if pd.isna(value):
            return ""
        return "background-color: #ffe5e5" if value < 0 else "background-color: #e6f4ea"

    summary_columns = [
        "period_label",
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
    if breakdown_column:
        summary_columns.insert(1, breakdown_column)
    summary_columns = [col for col in summary_columns if col in timeseries_df.columns]
    summary_table = timeseries_df[summary_columns].copy()

    rename_map = {
        "period_label": "期間",
        "sales_amount": "売上",
        "gross_profit": "粗利",
        "gross_margin": "粗利率",
        "comparison_sales": "前年同期売上",
        "comparison_gross_profit": "前年同期粗利",
        "comparison_margin": "前年粗利率",
        "yoy_rate": "売上前年比",
        "gross_profit_yoy": "粗利前年比",
        "margin_delta": "粗利率差",
    }
    if breakdown_column:
        rename_map[breakdown_column] = breakdown_label.replace("別", "")
    rename_map = {k: v for k, v in rename_map.items() if k in summary_table.columns}
    summary_table = summary_table.rename(columns=rename_map)

    format_map = {
        "売上": "{:,.0f}",
        "粗利": "{:,.0f}",
        "粗利率": "{:.1%}",
        "前年同期売上": "{:,.0f}",
        "前年同期粗利": "{:,.0f}",
        "前年粗利率": "{:.1%}",
        "売上前年比": "{:+.1%}",
        "粗利前年比": "{:+.1%}",
        "粗利率差": "{:+.1%}",
    }
    summary_formatter = {
        column: fmt for column, fmt in format_map.items() if column in summary_table.columns
    }
    summary_styler = summary_table.style.format(summary_formatter)
    highlight_cols = [
        column for column in ["売上前年比", "粗利前年比", "粗利率差"] if column in summary_table.columns
    ]
    if highlight_cols:
        summary_styler = summary_styler.applymap(_highlight_growth, subset=highlight_cols)
    st.dataframe(summary_styler, use_container_width=True)

    drilldown_df = sales.drilldown_details(
        filtered_sales, selection, granularity, breakdown_column
    )
    if not drilldown_df.empty:
        st.subheader("ドリルダウン詳細")
        detail_format = {
            column: fmt
            for column, fmt in {
                "売上": "{:,.0f}",
                "粗利": "{:,.0f}",
                "粗利率": "{:.1%}",
                "販売数量": "{:,.1f}",
            }.items()
            if column in drilldown_df.columns
        }
        st.dataframe(
            drilldown_df.style.format(detail_format),
            use_container_width=True,
        )

    breakdown_target = breakdown_column or "store"
    summary_df = sales.breakdown_summary(
        filtered_sales, comparison_sales, breakdown_target
    )
    if not summary_df.empty:
        dimension_col = sales.resolve_breakdown_column(breakdown_target) or breakdown_target
        summary_label = sales.breakdown_label(breakdown_target)
        st.subheader(f"{summary_label}比較")
        comparison_chart = go.Figure()
        comparison_chart.add_bar(
            x=summary_df[dimension_col],
            y=summary_df["sales_amount"],
            name="売上",
        )
        if summary_df["comparison_sales"].notna().any():
            comparison_chart.add_bar(
                x=summary_df[dimension_col],
                y=summary_df["comparison_sales"],
                name="前年同期売上",
                opacity=0.7,
            )
        comparison_chart.add_trace(
            go.Scatter(
                x=summary_df[dimension_col],
                y=summary_df["gross_margin"] * 100,
                name="粗利率",
                mode="lines+markers",
                yaxis="y2",
            )
        )
        if summary_df["comparison_margin"].notna().any():
            comparison_chart.add_trace(
                go.Scatter(
                    x=summary_df[dimension_col],
                    y=summary_df["comparison_margin"] * 100,
                    name="前年粗利率",
                    mode="lines+markers",
                    line=dict(dash="dash"),
                    yaxis="y2",
                )
            )
        comparison_chart.update_layout(
            hovermode="x unified",
            barmode="group",
            legend_title=summary_label,
            yaxis=dict(title="売上"),
            yaxis2=dict(
                title="粗利率(%)",
                overlaying="y",
                side="right",
                ticksuffix="%",
            ),
        )
        st.plotly_chart(comparison_chart, use_container_width=True)

        dimension_label = summary_label.replace("別", "")
        table_columns = [
            dimension_col,
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
        table_columns = [col for col in table_columns if col in summary_df.columns]
        dimension_table = summary_df[table_columns].copy()
        rename_map = {
            dimension_col: dimension_label,
            "sales_amount": "売上",
            "gross_profit": "粗利",
            "gross_margin": "粗利率",
            "comparison_sales": "前年同期売上",
            "comparison_gross_profit": "前年同期粗利",
            "comparison_margin": "前年粗利率",
            "yoy_rate": "売上前年比",
            "gross_profit_yoy": "粗利前年比",
            "margin_delta": "粗利率差",
        }
        rename_map = {k: v for k, v in rename_map.items() if k in dimension_table.columns}
        dimension_table = dimension_table.rename(columns=rename_map)
        dimension_formatter = {
            column: fmt for column, fmt in format_map.items() if column in dimension_table.columns
        }
        dimension_styler = dimension_table.style.format(dimension_formatter)
        highlight_cols = [
            column for column in ["売上前年比", "粗利前年比", "粗利率差"] if column in dimension_table.columns
        ]
        if highlight_cols:
            dimension_styler = dimension_styler.applymap(
                _highlight_growth, subset=highlight_cols
            )
        st.dataframe(dimension_styler, use_container_width=True)
    else:
        st.info("選択した条件に該当する集計データがありません。")


def render_products_tab(
    filtered_sales: pd.DataFrame,
    comparison_sales: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    templates = data_loader.available_templates()
    baseline = load_datasets(None, None, None)

    if "current_datasets" not in st.session_state:
        st.session_state["current_datasets"] = _copy_datasets(baseline)
        st.session_state["current_source"] = "sample"

    active_datasets = _copy_datasets(st.session_state.get("current_datasets", baseline))

    default_period = _default_period(active_datasets["sales"])
    stores = transformers.extract_stores(active_datasets["sales"])
    categories = transformers.extract_categories(active_datasets["sales"])
    regions = transformers.extract_regions(active_datasets["sales"])
    channels = transformers.extract_channels(active_datasets["sales"])
    provider_options = available_providers()

    sidebar_state = sidebar.render_sidebar(
        stores,
        categories,
        default_period=default_period,
        regions=regions,
        channels=channels,
        sample_files={k: str(v) for k, v in sample_files.items()},
        templates=templates,
        providers=provider_options,
    )

    mode = sidebar_state["data_source_mode"]
    validation_results: Dict[str, data_loader.ValidationResult] = {}
    integration_result: Optional[IntegrationResult] = None

    if mode == "csv":
        datasets, validation_results = _handle_csv_uploads(
            sidebar_state["uploads"],
            baseline,
        )
    elif mode == "api":
        datasets, integration_result = _handle_api_mode(
            sidebar_state["api"],
            baseline,
        )
    else:
        datasets = active_datasets

    if not datasets:
        datasets = active_datasets

    for key, default_df in baseline.items():
        datasets.setdefault(key, default_df.copy())

    datasets["sales"] = transformers.prepare_sales_dataset(datasets["sales"])

    st.session_state["current_datasets"] = _copy_datasets(datasets)
    st.session_state["current_source"] = mode

    filters = sidebar_state["filters"]
    filtered_sales = transformers.apply_filters(datasets["sales"], filters)
    comparison_mode = sidebar_state["comparison_mode"]
    comparison_sales = _comparison_dataset(datasets["sales"], filters, comparison_mode)

    tabs = st.tabs(
        [
            "売上分析",
            "商品別分析",
            "損益管理",
            "在庫分析",
            "経営シミュレーション",
            "データ取込管理",
        ]
    )

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

    with tabs[5]:
        integration_display = integration_result or st.session_state.get("latest_api_result")
        import_dashboard.render_dashboard(validation_results, integration_display)

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
