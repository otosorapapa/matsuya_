"""Sidebar UI component for the Streamlit dashboard."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

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


def _resolve_date_range(value: Tuple[date, date]) -> Tuple[date, date]:
    if isinstance(value, tuple) and len(value) == 2:
        return value
    if isinstance(value, list) and len(value) == 2:
        return value[0], value[1]
    today = date.today()
    return today.replace(month=max(1, today.month - 1), day=1), today


def render_sidebar(
    stores: List[str],
    categories: List[str],
    *,
    default_period: Tuple[date, date],
    sample_files: Dict[str, str],
    templates: Dict[str, bytes],
    providers: List[str],
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

    st.sidebar.header("フィルタ")
    store = st.sidebar.selectbox("店舗選択", stores)
    date_range_value = st.sidebar.date_input(
        "期間選択",
        value=default_period,
    )
    start_date, end_date = _resolve_date_range(date_range_value)
    category = st.sidebar.selectbox("商品カテゴリ", categories)
    comparison_label = st.sidebar.radio("比較モード", list(COMPARISON_OPTIONS.keys()))

    st.sidebar.header("帳票出力")
    export_csv = st.sidebar.checkbox("CSV出力", value=True)
    export_pdf = st.sidebar.checkbox("PDF出力")
    st.sidebar.markdown("---")

    filters = transformers.FilterState(
        store=store,
        start_date=start_date,
        end_date=end_date,
        category=category,
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
