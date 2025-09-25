"""Sidebar UI component for the Streamlit dashboard."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


DATASET_COLUMN_SPEC = {
    "sales": [
        ("date", "日付 (YYYY-MM-DD)"),
        ("store", "店舗名"),
        ("category", "商品カテゴリ"),
        ("product", "商品名"),
        ("sales_amount", "売上金額"),
        ("sales_qty", "販売数量"),
        ("cogs_amount", "売上原価"),
    ],
    "inventory": [
        ("store", "店舗名"),
        ("product", "商品名"),
        ("category", "カテゴリ"),
        ("opening_stock", "期首在庫数"),
        ("planned_purchase", "入荷予定数"),
        ("safety_stock", "安全在庫数"),
    ],
    "fixed_costs": [
        ("store", "店舗名 (任意)"),
        ("rent", "家賃"),
        ("utilities", "水道光熱費"),
        ("labor", "人件費"),
        ("other_costs", "その他固定費"),
    ],
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
    dataset_status: Optional[Dict[str, Dict[str, object]]] = None,
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

    status_map = dataset_status or {}
    st.sidebar.subheader("データ状態")
    status_container = st.sidebar.container()
    status_order = ["sales", "inventory", "fixed_costs"]
    for dataset in status_order:
        info = status_map.get(dataset, {})
        label = DATASET_LABELS.get(dataset, dataset)
        status = info.get("status", "missing")
        rows = int(info.get("rows", 0) or 0)
        source_label = info.get("source_label") or info.get("message") or "未アップロード"
        updated_at = info.get("updated_at")
        timestamp_text = None
        if isinstance(updated_at, datetime):
            timestamp_text = updated_at.strftime("%Y-%m-%d %H:%M")
        elif isinstance(updated_at, date):
            timestamp_text = updated_at.strftime("%Y-%m-%d")
        elif isinstance(updated_at, str):
            timestamp_text = updated_at
        if status == "ready":
            detail = source_label
            if rows:
                detail += f"（{rows:,}行）"
            if timestamp_text:
                detail += f"｜更新 {timestamp_text}"
            status_container.success(f"{label}: {detail}")
        elif status == "error":
            message = info.get("message") or info.get("error") or "取込エラーが発生しました。"
            if timestamp_text:
                message += f"（{timestamp_text}）"
            status_container.error(f"{label}: {message}")
        else:
            status_container.warning(f"{label}: 未アップロード")
    st.sidebar.caption("最新の取込状況を確認してから分析を進めてください。")

    if st.sidebar.button("データ仕様", key="open_data_spec_button"):
        st.session_state["show_data_spec_modal"] = True
    show_data_spec_modal = st.session_state.get("show_data_spec_modal", False)

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

    alert_state = st.session_state.setdefault(
        "alert_settings",
        {
            "stockout_threshold": 0,
            "excess_threshold": 5,
            "deficit_threshold": -500000,
            "notification_channel": "banner",
            "notification_email": "",
            "slack_webhook": "",
        },
    )
    st.sidebar.header("アラート設定")
    stockout_threshold = st.sidebar.number_input(
        "欠品アラート基準 (品目数)",
        min_value=0,
        value=int(alert_state.get("stockout_threshold", 0)),
        step=1,
    )
    excess_threshold = st.sidebar.number_input(
        "過剰在庫アラート基準 (品目数)",
        min_value=0,
        value=int(alert_state.get("excess_threshold", 5)),
        step=1,
    )
    deficit_threshold = st.sidebar.number_input(
        "赤字アラート基準 (円)",
        value=int(alert_state.get("deficit_threshold", -500000)),
        step=50000,
        format="%d",
    )
    notification_mode = st.sidebar.selectbox(
        "通知表示", ["ページ上部バナー", "モーダル"],
        index=0 if alert_state.get("notification_channel", "banner") == "banner" else 1,
    )
    notification_email = st.sidebar.text_input(
        "通知メールアドレス",
        value=alert_state.get("notification_email", ""),
        help="アラートをメール通知する場合に入力してください。",
    )
    slack_webhook = st.sidebar.text_input(
        "Slack Webhook URL",
        value=alert_state.get("slack_webhook", ""),
        help="Slack通知に利用するIncoming WebhookのURLを登録します。",
    )
    alert_state.update(
        {
            "stockout_threshold": int(stockout_threshold),
            "excess_threshold": int(excess_threshold),
            "deficit_threshold": float(deficit_threshold),
            "notification_channel": "banner"
            if notification_mode == "ページ上部バナー"
            else "modal",
            "notification_email": notification_email,
            "slack_webhook": slack_webhook,
        }
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
        "alert_settings": alert_state,
        "show_data_spec_modal": show_data_spec_modal,
        "data_spec_columns": DATASET_COLUMN_SPEC,
    }
