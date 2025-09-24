"""Streamlit entry point for the Matsya management dashboard."""
from __future__ import annotations

import hashlib
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pandas.tseries.offsets import MonthEnd
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
            stores=list(filters.stores),
            start_date=prev_start,
            end_date=prev_end,
            categories=list(filters.categories),
            regions=list(filters.regions),
            channels=list(filters.channels),
            period_granularity=filters.period_granularity,
            breakdown_dimension=filters.breakdown_dimension,
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


GLOBAL_FILTER_KEY = "global_filters"
NAV_STATE_KEY = "active_main_tab"
DEFAULT_NAV_KEY = "dashboard"
GLOBAL_FILTER_PRESETS = ["今月", "先月", "直近30日", "カスタム"]
MAIN_TAB_CONFIG = [
    ("dashboard", "ダッシュボード"),
    ("sales", "売上分析"),
    ("profit", "利益"),
    ("product", "商品分析"),
    ("inventory", "在庫"),
    ("simulation", "経営シミュレーション"),
    ("data", "データ管理"),
]
TARGET_MARGIN_RATE = 0.12
TARGET_CASH_RATIO = 0.25
BASE_CASH_BUFFER = 3_000_000.0


def _inject_global_styles() -> None:
    """Inject shared CSS for the redesigned dashboard."""

    st.markdown(
        """
        <style>
        .kpi-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 2px 4px rgba(15, 23, 42, 0.08);
            margin-bottom: 0.8rem;
        }
        .kpi-card .label {
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 0.4rem;
        }
        .kpi-card .value {
            font-size: 1.6rem;
            font-weight: 600;
            color: #111827;
        }
        .kpi-card .delta {
            font-size: 0.9rem;
            font-weight: 500;
            margin-top: 0.2rem;
        }
        .kpi-card .delta.positive {
            color: #148a4a;
        }
        .kpi-card .delta.negative {
            color: #d93025;
        }
        .kpi-card .target {
            font-size: 0.85rem;
            margin-top: 0.3rem;
            color: #4b5563;
        }
        .kpi-card .target.positive {
            color: #148a4a;
        }
        .kpi-card .target.negative {
            color: #d93025;
        }
        .alert-box {
            background: #fef3f2;
            border: 1px solid #fda29b;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #b42318;
            gap: 1rem;
        }
        .alert-box.success {
            background: #ecfdf3;
            border-color: #a6f4c5;
            color: #027a48;
        }
        .filter-bar {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 1rem;
        }
        .quick-menu button {
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _dataset_bounds(df: pd.DataFrame) -> Tuple[date, date]:
    if df.empty:
        today = date.today()
        return today, today
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    return min_date, max_date


def _to_date(value: object) -> Optional[date]:
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    return None


def _normalize_range(value: object, fallback: Tuple[date, date]) -> Tuple[date, date]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start = _to_date(value[0]) or fallback[0]
        end = _to_date(value[1]) or fallback[1]
    else:
        start, end = fallback
    if start > end:
        start, end = end, start
    return start, end


def _preset_range(
    preset: str,
    custom_range: Tuple[date, date],
    bounds: Tuple[date, date],
    fallback: Tuple[date, date],
) -> Tuple[date, date]:
    min_date, max_date = bounds
    start, end = fallback
    if preset == "今月":
        reference = max_date
        start = reference.replace(day=1)
        end = max_date
    elif preset == "先月":
        reference = max_date.replace(day=1) - timedelta(days=1)
        start = reference.replace(day=1)
        end = reference
    elif preset == "直近30日":
        end = max_date
        start = end - timedelta(days=29)
    elif preset == "カスタム":
        start, end = _normalize_range(custom_range, fallback)

    start = max(start, min_date)
    end = min(end, max_date)
    if start > end:
        start, end = end, start
    return start, end


def _format_currency(value: float, unit: str = "円") -> str:
    return f"{value:,.0f} {unit}" if value == value else f"0 {unit}"


def _format_number(value: float, unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    return f"{value:,.0f}{suffix}"


def _format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value*100:.1f}%"


def _compute_growth(current: float, previous: Optional[float]) -> Optional[float]:
    if previous is None or previous == 0:
        return None
    return (current - previous) / previous


def _render_kpi_cards(cards: Sequence[Dict[str, object]]) -> None:
    if not cards:
        return
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        yoy = card.get("yoy")
        if yoy is None:
            yoy_text = "前年比: データ不足"
            delta_class = "neutral"
        else:
            arrow = "▲" if yoy >= 0 else "▼"
            yoy_text = f"前年比: {arrow} {yoy*100:.1f}%"
            delta_class = "positive" if yoy >= 0 else "negative"
        target_diff = card.get("target_diff", 0.0)
        target_class = "positive" if target_diff >= 0 else "negative"
        target_unit = card.get("unit", "")
        target_suffix = f" {target_unit}" if target_unit else ""
        column.markdown(
            f"""
            <div class="kpi-card">
                <div class="label">{card.get('label', '')}</div>
                <div class="value">{card.get('value_text', '')}</div>
                <div class="delta {delta_class}">{yoy_text}</div>
                <div class="target {target_class}">目標差: {target_diff:+,.0f}{target_suffix}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _cash_flow_summary(sales_df: pd.DataFrame, inventory_df: pd.DataFrame) -> Dict[str, float]:
    if sales_df.empty:
        return {
            "deposit": BASE_CASH_BUFFER,
            "receivable": 0.0,
            "payable": 0.0,
            "balance": BASE_CASH_BUFFER,
        }

    total_sales = float(sales_df["sales_amount"].sum())
    total_qty = float(sales_df["sales_qty"].sum())
    total_cogs = float(sales_df["cogs_amount"].sum())
    period_days = max(
        1,
        (sales_df["date"].max() - sales_df["date"].min()).days + 1,
    )

    deposit = BASE_CASH_BUFFER + (total_sales / max(1.0, period_days)) * 5
    receivable = total_sales * 0.2
    avg_cost = total_cogs / total_qty if total_qty else 0.0
    planned_units = (
        inventory_df.get("planned_purchase", pd.Series(dtype=float)).sum()
        if not inventory_df.empty
        else 0.0
    )
    payable = planned_units * avg_cost
    balance = deposit + receivable - payable
    return {
        "deposit": deposit,
        "receivable": receivable,
        "payable": payable,
        "balance": balance,
    }


def render_global_filter_bar(
    stores: Sequence[str],
    categories: Sequence[str],
    *,
    default_period: Tuple[date, date],
    bounds: Tuple[date, date],
) -> transformers.FilterState:
    state = st.session_state.setdefault(
        GLOBAL_FILTER_KEY,
        {
            "preset": "今月",
            "custom_range": default_period,
            "store": transformers.ALL_STORES,
        },
    )

    st.markdown("#### 基本フィルタ")
    with st.container():
        col1, col2 = st.columns([2, 1])
        preset_index = (
            GLOBAL_FILTER_PRESETS.index(state.get("preset", "今月"))
            if state.get("preset") in GLOBAL_FILTER_PRESETS
            else 0
        )
        preset = col1.selectbox(
            "期間", GLOBAL_FILTER_PRESETS, index=preset_index
        )

        store_options = [transformers.ALL_STORES, *stores] if stores else [transformers.ALL_STORES]
        selected_store = state.get("store", transformers.ALL_STORES)
        store_index = (
            store_options.index(selected_store)
            if selected_store in store_options
            else 0
        )
        store_choice = col2.selectbox(
            "店舗", store_options, index=store_index
        )

    custom_range = state.get("custom_range", default_period)
    if preset == "カスタム":
        custom_range = st.date_input(
            "対象期間", value=_normalize_range(custom_range, default_period)
        )
    start_date, end_date = _preset_range(
        preset,
        _normalize_range(custom_range, default_period),
        bounds,
        default_period,
    )

    state.update(
        {
            "preset": preset,
            "store": store_choice,
            "custom_range": (start_date, end_date),
        }
    )

    selected_stores = list(stores)
    if store_choice != transformers.ALL_STORES and store_choice:
        selected_stores = [store_choice]

    filters = transformers.FilterState(
        stores=selected_stores,
        start_date=start_date,
        end_date=end_date,
        categories=list(categories),
        regions=[],
        channels=[],
        period_granularity="monthly",
        breakdown_dimension="store",
    )

    st.caption(
        f"期間: {start_date:%Y-%m-%d} 〜 {end_date:%Y-%m-%d} ／ 店舗: {store_choice}"
    )
    return filters



def render_dashboard_tab(
    sales_df: pd.DataFrame,
    comparison_sales: pd.DataFrame,
    filters: transformers.FilterState,
    fixed_costs_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    *,
    navigate: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    st.markdown("### 経営ダッシュボード")

    filtered_costs = fixed_costs_df.copy()
    filtered_inventory = inventory_df.copy()
    if filters.store != transformers.ALL_STORES:
        filtered_costs = filtered_costs[filtered_costs["store"] == filters.store]
        filtered_inventory = filtered_inventory[
            filtered_inventory["store"] == filters.store
        ]

    pnl_df = profitability.store_profitability(sales_df, filtered_costs)
    comparison_pnl = (
        profitability.store_profitability(comparison_sales, filtered_costs)
        if not comparison_sales.empty
        else pd.DataFrame(columns=pnl_df.columns)
    )

    total_sales = float(sales_df["sales_amount"].sum())
    previous_sales = (
        float(comparison_sales["sales_amount"].sum())
        if not comparison_sales.empty
        else None
    )
    sales_target = previous_sales * 1.05 if previous_sales else total_sales * 1.05

    operating_profit = float(
        pnl_df.get("operating_profit", pd.Series(dtype=float)).sum()
    )
    previous_operating_profit = (
        float(
            comparison_pnl.get("operating_profit", pd.Series(dtype=float)).sum()
        )
        if not comparison_pnl.empty
        else None
    )
    profit_target = sales_target * TARGET_MARGIN_RATE

    cash_current = _cash_flow_summary(sales_df, filtered_inventory)
    cash_previous = _cash_flow_summary(comparison_sales, filtered_inventory)
    cash_target = sales_target * TARGET_CASH_RATIO

    _render_kpi_cards(
        [
            {
                "label": "期間売上",
                "value_text": _format_currency(total_sales),
                "unit": "円",
                "yoy": _compute_growth(total_sales, previous_sales),
                "target_diff": total_sales - sales_target,
            },
            {
                "label": "営業利益",
                "value_text": _format_currency(operating_profit),
                "unit": "円",
                "yoy": _compute_growth(operating_profit, previous_operating_profit),
                "target_diff": operating_profit - profit_target,
            },
            {
                "label": "資金残高",
                "value_text": _format_currency(cash_current["balance"]),
                "unit": "円",
                "yoy": _compute_growth(
                    cash_current["balance"], cash_previous.get("balance")
                ),
                "target_diff": cash_current["balance"] - cash_target,
            },
        ]
    )

    granularity = "weekly"
    sales_trend = sales.timeseries_with_comparison(
        sales_df, comparison_sales, granularity
    )
    sales_chart = px.line(
        sales_trend,
        x="period_label",
        y="sales_amount",
        labels={"period_label": "期間", "sales_amount": "売上"},
        markers=True,
    )
    if sales_trend["comparison_sales"].notna().any():
        sales_chart.add_trace(
            go.Scatter(
                x=sales_trend["period_label"],
                y=sales_trend["comparison_sales"],
                mode="lines+markers",
                name="前期比",
                line=dict(dash="dash"),
            )
        )

    profit_trend = sales.aggregate_timeseries(sales_df, granularity)
    cost_columns = ["rent", "payroll", "utilities", "marketing", "other_fixed"]
    total_fixed_cost = 0.0
    if not filtered_costs.empty:
        total_fixed_cost = float(
            filtered_costs[cost_columns].fillna(0).sum(axis=1).sum()
        )
    total_sales_for_alloc = float(profit_trend["sales_amount"].sum())
    if not profit_trend.empty:
        if total_sales_for_alloc:
            profit_trend["allocated_fixed"] = (
                profit_trend["sales_amount"] / total_sales_for_alloc
            ) * total_fixed_cost
        else:
            profit_trend["allocated_fixed"] = total_fixed_cost / max(
                len(profit_trend), 1
            )
        profit_trend["operating_profit"] = (
            profit_trend["gross_profit"] - profit_trend["allocated_fixed"]
        )

    profit_chart = px.line(
        profit_trend,
        x="period_label",
        y="gross_profit",
        labels={"period_label": "期間", "gross_profit": "粗利"},
        markers=True,
    )
    if "operating_profit" in profit_trend.columns:
        profit_chart.add_trace(
            go.Scatter(
                x=profit_trend["period_label"],
                y=profit_trend["operating_profit"],
                mode="lines+markers",
                name="営業利益",
                line=dict(color="#2563eb"),
            )
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("売上トレンド")
        st.plotly_chart(sales_chart, use_container_width=True)
    with col2:
        st.subheader("利益トレンド")
        st.plotly_chart(profit_chart, use_container_width=True)

    overview_df = inventory.inventory_overview(sales_df, filtered_inventory)
    stockouts = (
        int((overview_df["stock_status"] == "在庫切れ").sum())
        if not overview_df.empty
        else 0
    )
    negative_stores = (
        int((pnl_df["operating_profit"] < 0).sum())
        if not pnl_df.empty
        else 0
    )

    if stockouts or negative_stores:
        alert_messages = []
        if stockouts:
            alert_messages.append(f"在庫欠品{stockouts}件")
        if negative_stores:
            alert_messages.append(f"赤字店舗{negative_stores}店")
        alert_text = "／".join(alert_messages) + " ➜ 詳細へ"
        container = st.container()
        with container:
            col_alert, col_action = st.columns([4, 1])
            with col_alert:
                st.markdown(
                    f"<div class='alert-box'>⚠️ {alert_text}</div>",
                    unsafe_allow_html=True,
                )
            with col_action:
                if st.button("詳細へ", key="dashboard-alert-detail") and navigate:
                    if stockouts:
                        navigate("inventory")
                    else:
                        navigate("profit")
    else:
        st.markdown(
            "<div class='alert-box success'>現在アラートはありません。</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### 簡易メニュー")
    quick_items = [
        ("sales", "売上分析"),
        ("profit", "利益"),
        ("product", "商品分析"),
        ("inventory", "在庫"),
        ("simulation", "シミュレーション"),
        ("data", "データ管理"),
    ]
    cols = st.columns(len(quick_items))
    for col, (key, label) in zip(cols, quick_items):
        if col.button(label, key=f"quick-menu-{key}") and navigate:
            navigate(key)

    return pnl_df


def render_sales_tab(
    sales_df: pd.DataFrame,
    filters: transformers.FilterState,
    channels: Sequence[str],
    *,
    comparison_mode: str,
) -> None:
    st.markdown("### 売上分析")

    state = st.session_state.setdefault(
        "sales_tab_state",
        {
            "channel": transformers.ALL_CHANNELS,
            "granularity": "monthly",
            "breakdown": "store",
            "comparison": comparison_mode,
        },
    )

    channel_options = (
        [transformers.ALL_CHANNELS, *channels]
        if channels
        else [transformers.ALL_CHANNELS]
    )
    granularity_options = {"月次": "monthly", "週次": "weekly", "日次": "daily"}
    breakdown_options = {
        "店舗別": "store",
        "カテゴリ別": "category",
        "地域別": "region",
    }
    comparison_options = {"前年比": "yoy", "対前期比": "previous_period"}

    with st.container():
        cols = st.columns(4)
        channel_choice = cols[0].selectbox(
            "チャネル",
            channel_options,
            index=channel_options.index(
                state.get("channel", transformers.ALL_CHANNELS)
            ),
        )
        granularity_label = cols[1].selectbox(
            "粒度",
            list(granularity_options.keys()),
            index=list(granularity_options.values()).index(
                state.get("granularity", "monthly")
            ),
        )
        breakdown_label = cols[2].selectbox(
            "表示単位",
            list(breakdown_options.keys()),
            index=list(breakdown_options.values()).index(
                state.get("breakdown", "store")
            ),
        )
        comparison_label = cols[3].selectbox(
            "比較",
            list(comparison_options.keys()),
            index=list(comparison_options.values()).index(
                state.get("comparison", comparison_mode)
            ),
        )

    state.update(
        {
            "channel": channel_choice,
            "granularity": granularity_options[granularity_label],
            "breakdown": breakdown_options[breakdown_label],
            "comparison": comparison_options[comparison_label],
        }
    )

    view_filters = transformers.FilterState(
        stores=list(filters.stores),
        start_date=filters.start_date,
        end_date=filters.end_date,
        categories=list(filters.categories),
        regions=list(filters.regions),
        channels=(
            list(channels)
            if channel_choice == transformers.ALL_CHANNELS
            else [channel_choice]
        ),
        period_granularity=state["granularity"],
        breakdown_dimension=state["breakdown"],
    )

    filtered_sales = transformers.apply_filters(sales_df, view_filters)
    comparison_sales = _comparison_dataset(
        sales_df, view_filters, state["comparison"]
    )

    if filtered_sales.empty:
        st.info("該当データがありません")
        return

    kpis = sales.kpi_summary(filtered_sales, comparison_sales)
    total_customers = float(kpis["total_customers"])
    previous_customers = (
        float(comparison_sales["sales_qty"].sum())
        if not comparison_sales.empty
        else None
    )
    avg_unit_price = float(kpis["avg_unit_price"])
    previous_avg_price = None
    if previous_customers and previous_customers > 0:
        prev_sales_amount = float(comparison_sales["sales_amount"].sum())
        previous_avg_price = (
            prev_sales_amount / previous_customers if prev_sales_amount else 0.0
        )

    sales_target = (
        float(comparison_sales["sales_amount"].sum()) * 1.05
        if not comparison_sales.empty
        else float(kpis["total_sales"]) * 1.05
    )
    customer_target = (
        previous_customers * 1.03 if previous_customers else total_customers
    )
    unit_price_target = (
        previous_avg_price * 1.02 if previous_avg_price else avg_unit_price
    )

    _render_kpi_cards(
        [
            {
                "label": "月次売上",
                "value_text": _format_currency(kpis["total_sales"]),
                "unit": "円",
                "yoy": kpis.get("yoy_rate"),
                "target_diff": kpis["total_sales"] - sales_target,
            },
            {
                "label": "来客数",
                "value_text": _format_number(total_customers, "人"),
                "unit": "人",
                "yoy": _compute_growth(total_customers, previous_customers),
                "target_diff": total_customers - customer_target,
            },
            {
                "label": "客単価",
                "value_text": _format_currency(avg_unit_price),
                "unit": "円",
                "yoy": _compute_growth(avg_unit_price, previous_avg_price),
                "target_diff": avg_unit_price - unit_price_target,
            },
        ]
    )

    breakdown_column = sales.resolve_breakdown_column(
        view_filters.breakdown_dimension
    )
    breakdown_label = sales.breakdown_label(view_filters.breakdown_dimension)
    timeseries_df = sales.timeseries_with_comparison(
        filtered_sales,
        comparison_sales,
        view_filters.period_granularity,
        breakdown_column,
    )
    timeseries_chart = px.line(
        timeseries_df,
        x="period_label",
        y="sales_amount",
        color=breakdown_column if breakdown_column else None,
        markers=True,
        labels={"period_label": "期間", "sales_amount": "売上"},
    )
    if timeseries_df["comparison_sales"].notna().any():
        timeseries_chart.add_trace(
            go.Scatter(
                x=timeseries_df["period_label"],
                y=timeseries_df["comparison_sales"],
                name="比較期間",
                mode="lines+markers",
                line=dict(dash="dash"),
            )
        )
    st.subheader(f"売上推移（{breakdown_label}）")
    st.plotly_chart(timeseries_chart, use_container_width=True)

    st.subheader("チャネル別構成比")
    composition_df = (
        filtered_sales.groupby("channel")["sales_amount"].sum().reset_index()
    )
    if not composition_df.empty:
        composition_df["構成比"] = (
            composition_df["sales_amount"] / composition_df["sales_amount"].sum()
        )
        pie = px.pie(
            composition_df,
            names="channel",
            values="sales_amount",
            hole=0.35,
        )
        st.plotly_chart(pie, use_container_width=True)
        st.dataframe(
            composition_df.rename(
                columns={"channel": "チャネル", "sales_amount": "売上"}
            )
            .assign(構成比=lambda df: df["構成比"].map(lambda v: f"{v*100:.1f}%")),
            use_container_width=True,
        )

    st.subheader("売上明細")
    details = filtered_sales[
        [
            "date",
            "store",
            "channel",
            "category",
            "product",
            "sales_amount",
            "sales_qty",
            "gross_profit",
            "gross_margin",
        ]
    ].copy()
    details = details.sort_values("date", ascending=False)
    st.dataframe(
        details.style.format(
            {
                "sales_amount": "{:,.0f}",
                "sales_qty": "{:,.1f}",
                "gross_profit": "{:,.0f}",
                "gross_margin": "{:.1%}",
            }
        ),
        use_container_width=True,
    )

    report.csv_download(
        "売上データをCSV出力",
        details,
        file_name=f"matsuya_sales_{filters.start_date:%Y%m%d}_{filters.end_date:%Y%m%d}.csv",
    )
    report.pdf_download(
        "売上データをPDF出力",
        "売上サマリー",
        details,
        file_name=f"matsuya_sales_{filters.start_date:%Y%m%d}.pdf",
    )

def render_products_tab(
    sales_df: pd.DataFrame,
    comparison_sales: pd.DataFrame,
    filters: transformers.FilterState,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    st.markdown("### 商品分析")

    category_options = sorted(sales_df["category"].dropna().unique().tolist())
    category_choices = [transformers.ALL_CATEGORIES, *category_options]
    state = st.session_state.setdefault(
        "products_tab_state",
        {"category": transformers.ALL_CATEGORIES},
    )
    selected_category = st.selectbox(
        "カテゴリ",
        category_choices,
        index=category_choices.index(state.get("category", transformers.ALL_CATEGORIES))
        if state.get("category", transformers.ALL_CATEGORIES) in category_choices
        else 0,
    )
    state["category"] = selected_category

    working_sales = sales_df.copy()
    working_comparison = comparison_sales.copy()
    if selected_category != transformers.ALL_CATEGORIES:
        working_sales = working_sales[working_sales["category"] == selected_category]
        if not working_comparison.empty:
            working_comparison = working_comparison[
                working_comparison["category"] == selected_category
            ]

    if working_sales.empty:
        st.info("該当データがありません")
        return working_sales, working_comparison

    aggregated = (
        working_sales.groupby("product")["sales_amount"].sum().reset_index()
    ).sort_values("sales_amount", ascending=False)
    top5 = aggregated.head(5)
    total_sales = float(working_sales["sales_amount"].sum())
    top5_sales = float(top5["sales_amount"].sum())
    previous_top5 = None
    if not working_comparison.empty:
        previous_agg = (
            working_comparison.groupby("product")["sales_amount"].sum().reset_index()
        ).sort_values("sales_amount", ascending=False)
        previous_top5 = float(previous_agg.head(5)["sales_amount"].sum())

    share = top5_sales / total_sales if total_sales else 0.0
    target_share = 0.6
    target_value = total_sales * target_share

    _render_kpi_cards(
        [
            {
                "label": "トップ5売上/構成比",
                "value_text": f"{top5_sales:,.0f} 円 / {share*100:.1f}%",
                "unit": "円",
                "yoy": _compute_growth(top5_sales, previous_top5),
                "target_diff": top5_sales - target_value,
            }
        ]
    )

    abc_df = products.abc_analysis(working_sales, working_comparison)
    if abc_df.empty:
        st.info("ABC分析を実行できません")
        return abc_df, abc_df

    st.subheader("ABC分析チャート")
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

    st.subheader("商品明細")
    detail_df = working_sales[
        [
            "date",
            "store",
            "category",
            "product",
            "sales_amount",
            "sales_qty",
            "gross_profit",
        ]
    ].copy()
    detail_df = detail_df.sort_values("sales_amount", ascending=False)
    st.dataframe(
        detail_df.style.format(
            {
                "sales_amount": "{:,.0f}",
                "sales_qty": "{:,.1f}",
                "gross_profit": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )
    return abc_df, top_growth

def render_profitability_tab(
    sales_df: pd.DataFrame,
    comparison_sales: pd.DataFrame,
    fixed_costs_df: pd.DataFrame,
    filters: transformers.FilterState,
    *,
    navigate: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    st.markdown("### 利益管理")

    cost_columns = ["rent", "payroll", "utilities", "marketing", "other_fixed"]
    adjusted = st.session_state.get("adjusted_fixed_costs")
    base_costs = fixed_costs_df.copy()
    if adjusted is not None:
        base_costs = pd.DataFrame(adjusted)
    editable = base_costs.set_index("store") if "store" in base_costs.columns else base_costs

    with st.expander("固定費内訳調整", expanded=False):
        edited = st.data_editor(
            editable,
            num_rows="dynamic",
            use_container_width=True,
            key="fixed_cost_editor",
        )
    edited_df = edited.reset_index() if "store" in edited.index.names else edited
    for col in cost_columns:
        if col in edited_df:
            edited_df[col] = pd.to_numeric(edited_df[col], errors="coerce").fillna(0)
    st.session_state["adjusted_fixed_costs"] = edited_df.to_dict(orient="list")

    pnl_df = profitability.store_profitability(sales_df, edited_df)
    comparison_pnl = (
        profitability.store_profitability(comparison_sales, edited_df)
        if not comparison_sales.empty
        else pd.DataFrame(columns=pnl_df.columns)
    )

    total_sales = float(pnl_df["sales_amount"].sum())
    gross_profit = float(pnl_df["gross_profit"].sum())
    operating_profit = float(pnl_df["operating_profit"].sum())
    gross_margin = gross_profit / total_sales if total_sales else 0.0

    previous_operating = (
        float(comparison_pnl["operating_profit"].sum())
        if not comparison_pnl.empty
        else None
    )
    previous_gross = (
        float(comparison_pnl["gross_profit"].sum())
        if not comparison_pnl.empty
        else None
    )
    previous_sales_total = (
        float(comparison_pnl["sales_amount"].sum())
        if not comparison_pnl.empty
        else None
    )
    previous_margin = (
        previous_gross / previous_sales_total if previous_sales_total else None
    )

    _render_kpi_cards(
        [
            {
                "label": "粗利",
                "value_text": _format_currency(gross_profit),
                "unit": "円",
                "yoy": _compute_growth(gross_profit, previous_gross),
                "target_diff": gross_profit - total_sales * TARGET_MARGIN_RATE,
            },
                {
                    "label": "粗利率",
                    "value_text": f"{gross_margin*100:.1f}%",
                    "unit": "%",
                    "yoy": _compute_growth(gross_margin, previous_margin),
                    "target_diff": (gross_margin - TARGET_MARGIN_RATE) * 100,
                },
            {
                "label": "営業利益",
                "value_text": _format_currency(operating_profit),
                "unit": "円",
                "yoy": _compute_growth(operating_profit, previous_operating),
                "target_diff": operating_profit - total_sales * TARGET_MARGIN_RATE,
            },
        ]
    )

    st.subheader("店舗別損益表")
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

    if navigate is not None:
        st.info("シミュレーションで目標利益を検討できます。")
        if st.button("シミュレーションを開く", key="profit_to_sim"):
            navigate("simulation")

    return pnl_df

def render_inventory_tab(
    sales_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    abc_df: Optional[pd.DataFrame],
    filters: transformers.FilterState,
) -> None:
    st.markdown("### 在庫分析")

    store_options = sorted(inventory_df["store"].dropna().unique().tolist())
    category_options = sorted(inventory_df["category"].dropna().unique().tolist())
    store_choices = [transformers.ALL_STORES, *store_options]
    category_choices = [transformers.ALL_CATEGORIES, *category_options]

    state = st.session_state.setdefault(
        "inventory_tab_state",
        {
            "store": filters.store,
            "category": filters.category,
        },
    )
    with st.container():
        col1, col2 = st.columns(2)
        store_choice = col1.selectbox(
            "店舗",
            store_choices,
            index=store_choices.index(state.get("store", store_choices[0]))
            if state.get("store") in store_choices
            else 0,
        )
        category_choice = col2.selectbox(
            "カテゴリ",
            category_choices,
            index=category_choices.index(state.get("category", category_choices[0]))
            if state.get("category") in category_choices
            else 0,
        )
    state.update({"store": store_choice, "category": category_choice})

    working_sales = sales_df.copy()
    working_inventory = inventory_df.copy()
    if store_choice != transformers.ALL_STORES:
        working_sales = working_sales[working_sales["store"] == store_choice]
        working_inventory = working_inventory[working_inventory["store"] == store_choice]
    if category_choice != transformers.ALL_CATEGORIES:
        working_sales = working_sales[working_sales["category"] == category_choice]
        working_inventory = working_inventory[working_inventory["category"] == category_choice]

    overview_df = inventory.inventory_overview(working_sales, working_inventory)
    if overview_df.empty:
        st.info("在庫データが見つかりません。")
        return

    rank_lookup: Dict[str, str] = {}
    if abc_df is not None and not abc_df.empty:
        filtered_abc = abc_df.copy()
        if store_choice != transformers.ALL_STORES and "store" in filtered_abc.columns:
            filtered_abc = filtered_abc[filtered_abc.get("store") == store_choice]
        if category_choice != transformers.ALL_CATEGORIES:
            filtered_abc = filtered_abc[filtered_abc.get("category") == category_choice]
        if "product" in filtered_abc.columns:
            rank_lookup = dict(zip(filtered_abc["product"], filtered_abc.get("rank", [])))

    advice_df = inventory.inventory_advice(overview_df, rank_lookup)

    safety_excess = int((advice_df["stock_status"] == "在庫過多").sum())
    stockouts = int((advice_df["stock_status"] == "在庫切れ").sum())
    period_days = max((filters.end_date - filters.start_date).days + 1, 1)
    turnover_df = inventory.turnover_by_category(overview_df, period_days=period_days)
    avg_turnover = float(turnover_df["turnover"].mean()) if not turnover_df.empty else 0.0

    _render_kpi_cards(
        [
            {
                "label": "安全在庫超過数",
                "value_text": _format_number(safety_excess, "品目"),
                "unit": "品目",
                "yoy": None,
                "target_diff": -safety_excess,
            },
            {
                "label": "欠品数",
                "value_text": _format_number(stockouts, "品目"),
                "unit": "品目",
                "yoy": None,
                "target_diff": -stockouts,
            },
                {
                    "label": "平均在庫回転率",
                    "value_text": f"{avg_turnover:.1f} 回",
                    "unit": "回",
                    "yoy": None,
                    "target_diff": avg_turnover - 8,
                },
        ]
    )

    st.subheader("在庫推定とアドバイス")
    st.dataframe(advice_df, use_container_width=True)

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
    _inject_global_styles()

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
        show_filters=False,
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

    bounds = _dataset_bounds(datasets["sales"])
    global_filters = render_global_filter_bar(
        stores,
        categories,
        default_period=default_period,
        bounds=bounds,
    )

    filtered_sales = transformers.apply_filters(datasets["sales"], global_filters)
    dashboard_comparison = _comparison_dataset(
        datasets["sales"], global_filters, "previous_period"
    )

    def _navigate(tab_key: str) -> None:
        st.session_state[NAV_STATE_KEY] = tab_key
        st.experimental_rerun()

    tab_labels = [label for _, label in MAIN_TAB_CONFIG]
    current_tab = st.session_state.get(NAV_STATE_KEY, DEFAULT_NAV_KEY)
    current_index = next(
        (idx for idx, (key, _) in enumerate(MAIN_TAB_CONFIG) if key == current_tab),
        0,
    )
    selected_label = st.radio(
        "画面切替",
        tab_labels,
        index=current_index,
        horizontal=True,
        label_visibility="collapsed",
    )
    selected_key = next(
        key for key, label in MAIN_TAB_CONFIG if label == selected_label
    )
    st.session_state[NAV_STATE_KEY] = selected_key

    pnl_baseline = profitability.store_profitability(
        filtered_sales,
        datasets["fixed_costs"],
    )

    if selected_key == "dashboard":
        pnl_df = render_dashboard_tab(
            filtered_sales,
            dashboard_comparison,
            global_filters,
            datasets["fixed_costs"],
            datasets["inventory"],
            navigate=_navigate,
        )
        st.session_state["latest_pnl_df"] = pnl_df
    elif selected_key == "sales":
        render_sales_tab(
            datasets["sales"],
            global_filters,
            channels,
            comparison_mode="yoy",
        )
    elif selected_key == "product":
        abc_df, _ = render_products_tab(filtered_sales, dashboard_comparison, global_filters)
        st.session_state["latest_abc_df"] = abc_df
    elif selected_key == "profit":
        pnl_df = render_profitability_tab(
            filtered_sales,
            dashboard_comparison,
            datasets["fixed_costs"],
            global_filters,
            navigate=_navigate,
        )
        st.session_state["latest_pnl_df"] = pnl_df
    elif selected_key == "inventory":
        abc_df = st.session_state.get("latest_abc_df")
        if abc_df is None:
            abc_df = products.abc_analysis(filtered_sales, dashboard_comparison)
            st.session_state["latest_abc_df"] = abc_df
        render_inventory_tab(filtered_sales, datasets["inventory"], abc_df, global_filters)
    elif selected_key == "simulation":
        pnl_df = st.session_state.get("latest_pnl_df", pnl_baseline)
        render_simulation_tab(pnl_df, global_filters)
    else:
        integration_display = integration_result or st.session_state.get("latest_api_result")
        import_dashboard.render_dashboard(validation_results, integration_display)

if __name__ == "__main__":
    main()
