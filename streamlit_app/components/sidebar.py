"""Sidebar UI component for the Streamlit dashboard."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import streamlit as st

from .. import transformers


COMPARISON_OPTIONS = {
    "前年比": "yoy",
    "対前期比": "previous_period",
}


DATASET_LABELS = {
    "sales": "売上",
    "inventory": "仕入/在庫",
    "fixed_costs": "固定費",
}


PERIOD_OPTIONS = {
    "日次": "daily",
    "週次": "weekly",
    "月次": "monthly",
    "年次": "yearly",
}


BREAKDOWN_OPTIONS = {
    "店舗別": "store",
    "カテゴリ別": "category",
}


def _resolve_date_range(value: Tuple[date, date]) -> Tuple[date, date]:
    if isinstance(value, tuple) and len(value) == 2:
        return value
    if isinstance(value, list) and len(value) == 2:
        return value[0], value[1]
    today = date.today()
    return today.replace(month=max(1, today.month - 1), day=1), today


def _ensure_selection(options: Sequence[str], selected: Sequence[str]) -> List[str]:
    values = list(options)
    chosen = list(selected)
    if not values:
        return []
    return chosen or values


def render_sidebar(
    stores: List[str],
    categories: List[str],
    *,
    default_period: Tuple[date, date],
    regions: List[str],
    channels: List[str],
    sample_files: Dict[str, str],
    templates: Dict[str, bytes],
    providers: List[str],
    show_filters: bool = True,
) -> Dict[str, object]:
    st.sidebar.header("データソース")
    mode_label = st.sidebar.radio("連携方法", ["CSVアップロード", "API連携"])
    data_source_mode = "csv" if mode_label == "CSVアップロード" else "api"

    uploaded_sales = uploaded_inventory = uploaded_fixed_costs = None
    api_state = {
        "provider": providers[0] if providers else "",
        "api_key": "",
        "api_secret": "",
        "start_date": default_period[0],
        "end_date": default_period[1],
        "fetch_triggered": False,
        "auto_daily": False,
    }

    if data_source_mode == "csv":
        uploaded_sales = st.sidebar.file_uploader("売上CSVをアップロード", type="csv")
        uploaded_inventory = st.sidebar.file_uploader("仕入/在庫CSVをアップロード", type="csv")
        uploaded_fixed_costs = st.sidebar.file_uploader("固定費CSVをアップロード", type="csv")
    else:
        if providers:
            default_end = default_period[1]
            default_start = max(default_end - timedelta(days=6), default_period[0])
            with st.sidebar.form("api_connection_form"):
                provider = st.selectbox("連携先", providers)
                api_key = st.text_input("APIキー", type="password")
                api_secret = st.text_input("APIシークレット", type="password")
                api_range = st.date_input(
                    "取得期間",
                    value=(default_start, default_end),
                )
                fetch_triggered = st.form_submit_button("データ取得")
            start_date, end_date = _resolve_date_range(api_range)
            auto_daily = st.sidebar.checkbox("前日データを自動取得", key="auto_daily_toggle")
            api_state.update(
                {
                    "provider": provider,
                    "api_key": api_key,
                    "api_secret": api_secret,
                    "start_date": start_date,
                    "end_date": end_date,
                    "fetch_triggered": fetch_triggered,
                    "auto_daily": auto_daily,
                }
            )
        else:
            st.sidebar.warning("利用可能なAPI連携先が設定されていません。CSVをアップロードしてください。")

    with st.sidebar.expander("サンプルデータ・テンプレートをダウンロード"):
        for key, path in sample_files.items():
            file_path = Path(path)
            with file_path.open("rb") as fp:
                data = fp.read()
            label = DATASET_LABELS.get(key, key)
            st.download_button(
                label=f"{label}サンプルCSVをダウンロード",
                data=data,
                file_name=file_path.name,
                key=f"sample-{key}",
            )
        st.markdown("---")
        st.caption("アップロード用テンプレート")
        for key, content in templates.items():
            label = DATASET_LABELS.get(key, key)
            st.download_button(
                label=f"{label}テンプレートCSVをダウンロード",
                data=content,
                file_name=f"{key}_template.csv",
                key=f"template-{key}",
            )

    export_csv = export_pdf = False
    comparison_label = list(COMPARISON_OPTIONS.keys())[0]
    period_label = list(PERIOD_OPTIONS.keys())[2]
    breakdown_label = list(BREAKDOWN_OPTIONS.keys())[0]
    store_selection = list(stores)
    category_selection = list(categories)
    region_selection: List[str] = list(regions)
    channel_selection: List[str] = list(channels)
    start_date, end_date = default_period

    if show_filters:
        st.sidebar.header("フィルタ")
        store_selection = st.sidebar.multiselect(
            "店舗選択",
            stores,
            default=stores,
        )
        date_range_value = st.sidebar.date_input(
            "期間選択",
            value=default_period,
        )
        start_date, end_date = _resolve_date_range(date_range_value)
        category_selection = st.sidebar.multiselect(
            "商品カテゴリ",
            categories,
            default=categories,
        )
        region_selection = []
        if regions:
            region_selection = st.sidebar.multiselect(
                "地域",
                regions,
                default=regions,
            )
        channel_selection = []
        if channels:
            channel_selection = st.sidebar.multiselect(
                "販売チャネル",
                channels,
                default=channels,
            )
        comparison_label = st.sidebar.radio("比較モード", list(COMPARISON_OPTIONS.keys()))

        st.sidebar.header("集計設定")
        period_label = st.sidebar.radio(
            "期間粒度",
            list(PERIOD_OPTIONS.keys()),
            index=2,
            horizontal=True,
        )
        breakdown_label = st.sidebar.radio(
            "表示単位",
            list(BREAKDOWN_OPTIONS.keys()),
            horizontal=True,
        )

        st.sidebar.header("帳票出力")
        export_csv = st.sidebar.checkbox("CSV出力", value=True)
        export_pdf = st.sidebar.checkbox("PDF出力")
        st.sidebar.markdown("---")

    filters = transformers.FilterState(
        stores=_ensure_selection(stores, store_selection),
        start_date=start_date,
        end_date=end_date,
        categories=_ensure_selection(categories, category_selection),
        regions=_ensure_selection(regions, region_selection),
        channels=_ensure_selection(channels, channel_selection),
        period_granularity=PERIOD_OPTIONS[period_label],
        breakdown_dimension=BREAKDOWN_OPTIONS[breakdown_label],
    )
    return {
        "uploads": {
            "sales": uploaded_sales,
            "inventory": uploaded_inventory,
            "fixed_costs": uploaded_fixed_costs,
        },
        "data_source_mode": data_source_mode,
        "api": api_state,
        "filters": filters,
        "comparison_mode": COMPARISON_OPTIONS[comparison_label],
        "export_csv": export_csv,
        "export_pdf": export_pdf,
    }
