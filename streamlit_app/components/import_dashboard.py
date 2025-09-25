"""Components to display and manage data import history."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import streamlit as st

from streamlit_app import rerun as trigger_rerun

from streamlit_app.integrations import IntegrationResult


if TYPE_CHECKING:
    from streamlit_app.data_loader import ValidationResult


_HISTORY_KEY = "import_history"
_DATASET_LABELS = {
    "sales": "売上",
    "inventory": "仕入/在庫",
    "fixed_costs": "固定費",
}


def record_validation_import(
    dataset: str,
    source: str,
    result: ValidationResult,
    *,
    record_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    """Persist a validation outcome in the import history."""

    status = "エラー" if not result.valid else ("要確認" if result.has_errors else "取込済")
    record_import(
        dataset=dataset,
        source=source,
        dataframe=result.dataframe,
        status=status,
        total_rows=result.total_rows,
        dropped_rows=result.dropped_rows,
        errors_df=result.errors,
        notes=notes,
        record_id=record_id,
    )


def record_api_import(
    dataset: str,
    source: str,
    dataframe: pd.DataFrame,
    *,
    total_rows: Optional[int] = None,
    notes: Optional[str] = None,
    record_id: Optional[str] = None,
) -> None:
    """Persist an API integration outcome in the import history."""

    record_import(
        dataset=dataset,
        source=source,
        dataframe=dataframe,
        status="取込済",
        total_rows=total_rows if total_rows is not None else len(dataframe),
        dropped_rows=0,
        errors_df=pd.DataFrame(columns=["行番号", "列名", "内容"]),
        notes=notes,
        record_id=record_id,
    )


def record_import(
    *,
    dataset: str,
    source: str,
    dataframe: pd.DataFrame,
    status: str,
    total_rows: int,
    dropped_rows: int,
    errors_df: Optional[pd.DataFrame],
    notes: Optional[str],
    record_id: Optional[str] = None,
) -> None:
    history = st.session_state.setdefault(_HISTORY_KEY, [])
    if record_id and any(entry["id"] == record_id for entry in history):
        return

    record_id = record_id or str(uuid4())
    timestamp = datetime.utcnow().isoformat()

    errors = _errors_to_records(errors_df)
    entry = {
        "id": record_id,
        "dataset": dataset,
        "source": source,
        "timestamp": timestamp,
        "status": status,
        "rows": int(len(dataframe)),
        "total_rows": int(total_rows),
        "dropped_rows": int(dropped_rows),
        "notes": notes,
        "summary": _summarize_dataset(dataset, dataframe),
        "errors": errors,
    }
    history.append(entry)
    st.session_state[_HISTORY_KEY] = history


def delete_import(record_id: str) -> None:
    history = st.session_state.get(_HISTORY_KEY, [])
    st.session_state[_HISTORY_KEY] = [entry for entry in history if entry["id"] != record_id]


def render_dashboard(
    validation_results: Dict[str, ValidationResult],
    integration_result: Optional[IntegrationResult] = None,
) -> None:
    """Render import status, validation feedback and history."""

    st.subheader("フォーマットチェック結果")
    if not validation_results:
        st.info("アップロード済みファイルはありません。")
    else:
        for dataset, result in validation_results.items():
            if result is None:
                continue
            label = _DATASET_LABELS.get(dataset, dataset)
            if not result.valid:
                st.error(f"{label}のCSV構造がテンプレートと一致しません。")
            elif result.has_errors:
                st.warning(
                    f"{label}でフォーマットエラーが見つかりました（{result.dropped_rows}行を除外）。"
                )
            else:
                st.success(f"{label}のCSVを正常に取り込みました。")

            if result.errors is not None and not result.errors.empty:
                st.dataframe(result.errors, use_container_width=True)

    if integration_result is not None:
        st.subheader("API連携状況")
        st.info(
            f"{integration_result.provider} から {integration_result.period_label()} のデータを取得しました。"
        )
        counts = [
            {
                "データ種別": _DATASET_LABELS.get(name, name),
                "件数": len(df),
            }
            for name, df in integration_result.datasets.items()
        ]
        st.dataframe(pd.DataFrame(counts), use_container_width=True)
        st.caption(integration_result.message)

    st.subheader("取込履歴")
    history = st.session_state.get(_HISTORY_KEY, [])
    if not history:
        st.info("まだ取込履歴がありません。")
        return

    history_df = pd.DataFrame(history)
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
    history_df = history_df.sort_values("timestamp", ascending=False)

    display_df = history_df.assign(
        データ種別=history_df["dataset"].map(_DATASET_LABELS).fillna(history_df["dataset"]),
        取込元=history_df["source"],
        ステータス=history_df["status"],
        取込日時=history_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
        取込件数=history_df["rows"],
        除外件数=history_df["dropped_rows"],
        備考=history_df["notes"],
    )[
        ["取込日時", "データ種別", "取込元", "ステータス", "取込件数", "除外件数", "備考"]
    ]
    st.dataframe(display_df, use_container_width=True)

    selection_options = history_df["id"].tolist()
    if not selection_options:
        return

    selected_id = st.selectbox(
        "詳細を表示する履歴を選択", selection_options, format_func=lambda rid: _format_option(history_df, rid)
    )
    selected = history_df[history_df["id"] == selected_id].iloc[0]
    label = _DATASET_LABELS.get(selected["dataset"], selected["dataset"])
    st.markdown(f"**{label}の詳細**")

    previous = _previous_record(history_df, selected)
    summary = selected["summary"] or {}
    if summary:
        deltas = _summary_deltas(summary, previous["summary"] if previous is not None else None)
        columns = st.columns(len(summary))
        for column, (key, value) in zip(columns, summary.items()):
            delta_value = deltas.get(key)
            delta_label = None if delta_value is None else f"{delta_value:,.0f}"
            column.metric(key, f"{value:,.0f}", delta_label)

    if selected["errors"]:
        st.error(f"{label}の取込で{len(selected['errors'])}件のエラーがありました。")
        st.dataframe(pd.DataFrame(selected["errors"]), use_container_width=True)

    if st.button("選択した履歴を削除", key=f"delete-{selected_id}"):
        delete_import(selected_id)
        st.success("履歴を削除しました。画面を更新します。")
        trigger_rerun()


def _format_option(history_df: pd.DataFrame, record_id: str) -> str:
    record = history_df[history_df["id"] == record_id].iloc[0]
    label = _DATASET_LABELS.get(record["dataset"], record["dataset"])
    timestamp = record["timestamp"].strftime("%m/%d %H:%M")
    return f"{timestamp} - {label} ({record['source']})"


def _summary_deltas(current: Dict[str, float], previous: Optional[Dict[str, float]]) -> Dict[str, Optional[float]]:
    if previous is None:
        return {key: None for key in current}
    return {key: current.get(key, 0) - previous.get(key, 0) for key in current}


def _previous_record(history_df: pd.DataFrame, selected: pd.Series) -> Optional[pd.Series]:
    dataset = selected["dataset"]
    timestamp = selected["timestamp"]
    candidates = history_df[
        (history_df["dataset"] == dataset) & (history_df["timestamp"] < timestamp)
    ].sort_values("timestamp", ascending=False)
    if candidates.empty:
        return None
    return candidates.iloc[0]


def _summarize_dataset(dataset: str, dataframe: pd.DataFrame) -> Dict[str, float]:
    if dataframe.empty:
        return {}
    if dataset == "sales":
        return {
            "売上金額": float(dataframe["sales_amount"].sum()),
            "粗利": float(dataframe["gross_profit"].sum()),
        }
    if dataset == "inventory":
        return {
            "期首在庫": float(dataframe["opening_stock"].sum()),
            "仕入予定": float(dataframe["planned_purchase"].sum()),
        }
    if dataset == "fixed_costs":
        value = dataframe[["rent", "payroll", "utilities", "marketing", "other_fixed"]].sum().sum()
        return {"固定費合計": float(value)}
    return {}


def _errors_to_records(errors_df: Optional[pd.DataFrame]) -> List[Dict[str, object]]:
    if errors_df is None or errors_df.empty:
        return []
    return errors_df.to_dict(orient="records")
