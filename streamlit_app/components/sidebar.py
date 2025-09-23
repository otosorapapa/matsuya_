"""Sidebar UI component for the Streamlit dashboard."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Tuple

import streamlit as st

from .. import transformers


COMPARISON_OPTIONS = {
    "前年比": "yoy",
    "対前期比": "previous_period",
}


def _resolve_date_range(value: Tuple[date, date]) -> Tuple[date, date]:
    if isinstance(value, tuple) and len(value) == 2:
        return value
    if isinstance(value, list) and len(value) == 2:
        return value[0], value[1]
    today = date.today()
    return today.replace(month=max(1, today.month - 1), day=1), today


def render_sidebar(
    stores: list[str],
    categories: list[str],
    *,
    default_period: Tuple[date, date],
    sample_files: Dict[str, str],
) -> Dict[str, object]:
    st.sidebar.header("データソース")
    uploaded_sales = st.sidebar.file_uploader("売上CSVをアップロード", type="csv")
    uploaded_inventory = st.sidebar.file_uploader("仕入/在庫CSVをアップロード", type="csv")
    uploaded_fixed_costs = st.sidebar.file_uploader("固定費CSVをアップロード", type="csv")

    with st.sidebar.expander("サンプルデータをダウンロード"):
        for label, path in sample_files.items():
            file_path = Path(path)
            with file_path.open("rb") as fp:
                data = fp.read()
            st.download_button(
                label=f"{label}サンプルCSVをダウンロード",
                data=data,
                file_name=file_path.name,
                key=f"sample-{label}",
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
        "filters": filters,
        "comparison_mode": COMPARISON_OPTIONS[comparison_label],
        "export_csv": export_csv,
        "export_pdf": export_pdf,
    }
