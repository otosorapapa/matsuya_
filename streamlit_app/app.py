"""Streamlit entry point for the Matsya management dashboard."""
from __future__ import annotations

import hashlib
import logging
import sys
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import pandas as pd
from pandas.tseries.offsets import MonthEnd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from streamlit_app import data_loader, rerun as trigger_rerun, transformers
from streamlit_app.analytics import inventory, products, profitability, sales, simulation
from streamlit_app.components import import_dashboard, report, sidebar
from streamlit_app.integrations import IntegrationResult, available_providers, fetch_datasets
from streamlit_app.theme import inject_custom_css


logger = logging.getLogger(__name__)


def _set_state_and_rerun(key: str, value: object) -> None:
    """Update ``st.session_state`` and trigger a rerun immediately."""

    st.session_state[key] = value
    trigger_rerun()


st.set_page_config(
    page_title="松屋 計数管理ダッシュボード",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

MAIN_TAB_KEY = "main_active_tab"
MAIN_TAB_LABELS = ["売上", "粗利", "在庫", "資金"]
PAGE_OPTIONS = ["ダッシュボード", "データ管理", "ヘルプ／設定"]
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

DATASET_LABEL_MAP: Dict[str, str] = {
    name: label for name, (_, _, label) in _DATASET_CONFIGS.items()
}
DATASET_METADATA_KEY = "dataset_metadata"
DATA_SOURCE_LABELS = {
    "sample": "サンプルデータ",
    "csv": "CSVアップロード",
    "api": "API連携",
}

_NumT = TypeVar("_NumT")


def _coerce_number(
    value: object, converter: Callable[[object], _NumT], default: _NumT
) -> _NumT:
    """Safely convert user-provided values to numeric types."""

    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return default
    except TypeError:
        pass
    try:
        return converter(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    return _coerce_number(value, int, default)


def _coerce_float(value: object, default: float = 0.0) -> float:
    return _coerce_number(value, float, default)


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


def _coerce_datetime_column(
    df: pd.DataFrame, column: str = "date"
) -> Tuple[pd.DataFrame, int]:
    """Ensure ``column`` is a datetime and drop invalid rows."""

    if column not in df.columns:
        return df.copy(), 0
    dataset = df.copy()
    coerced = pd.to_datetime(dataset[column], errors="coerce")
    invalid = coerced.isna()
    dataset.loc[:, column] = coerced
    if invalid.any():
        dataset = dataset.loc[~invalid].copy()
    return dataset, int(invalid.sum())


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


def _ensure_dataset_metadata(
    datasets: Dict[str, pd.DataFrame],
    *,
    default_source: str = "sample",
) -> Dict[str, Dict[str, object]]:
    """Synchronise dataset metadata stored in ``st.session_state``."""

    if DATASET_METADATA_KEY not in st.session_state:
        st.session_state[DATASET_METADATA_KEY] = {}
    metadata: Dict[str, Dict[str, object]] = st.session_state[DATASET_METADATA_KEY]
    default_label = DATA_SOURCE_LABELS.get(default_source, default_source)
    for name, df in datasets.items():
        rows = int(len(df)) if df is not None else 0
        record = dict(metadata.get(name, {}))
        record.setdefault("label", DATASET_LABEL_MAP.get(name, name))
        record.setdefault("source", default_source)
        record.setdefault("source_label", default_label)
        record.setdefault("updated_at", st.session_state.get("last_data_update"))
        if record.get("status") != "error":
            if rows:
                record["status"] = "ready"
                source_label = record.get("source_label") or default_label
                record["message"] = f"{source_label} ({rows:,}行)"
            else:
                record["status"] = "missing"
                record["message"] = "未アップロード"
                record.pop("error", None)
        record["rows"] = rows
        metadata[name] = record
    st.session_state[DATASET_METADATA_KEY] = metadata
    return metadata


def _update_metadata_from_uploads(
    uploads: Dict[str, Optional[object]],
    validations: Dict[str, data_loader.ValidationResult],
) -> None:
    """Store upload results for the sidebar status indicator."""

    if not uploads:
        return
    if DATASET_METADATA_KEY not in st.session_state:
        st.session_state[DATASET_METADATA_KEY] = {}
    metadata: Dict[str, Dict[str, object]] = st.session_state[DATASET_METADATA_KEY]
    timestamp = datetime.now()
    any_success = False
    for dataset, upload in uploads.items():
        if upload is None:
            continue
        validation = validations.get(dataset)
        file_name = getattr(upload, "name", "uploaded_file")
        rows = int(validation.total_rows) if validation is not None else 0
        record = dict(metadata.get(dataset, {}))
        record.update(
            {
                "label": DATASET_LABEL_MAP.get(dataset, dataset),
                "source": "csv",
                "source_label": file_name,
                "rows": rows,
                "updated_at": timestamp,
            }
        )
        if validation is not None and validation.valid:
            record["status"] = "ready"
            record["message"] = f"{file_name} ({rows:,}行)"
            record.pop("error", None)
            any_success = True
        else:
            record["status"] = "error"
            error_message = None
            if validation is not None and validation.errors is not None and not validation.errors.empty:
                error_message = str(validation.errors.iloc[0].get("内容", ""))
            record["error"] = error_message or "CSVの形式を確認してください。"
            record["message"] = record["error"]
        metadata[dataset] = record
    st.session_state[DATASET_METADATA_KEY] = metadata
    if any_success:
        st.session_state["last_data_update"] = timestamp
        st.session_state["current_source"] = "csv"


def _update_metadata_from_integration(result: Optional[IntegrationResult]) -> None:
    """Update dataset metadata after API integrations."""

    if result is None:
        return
    if DATASET_METADATA_KEY not in st.session_state:
        st.session_state[DATASET_METADATA_KEY] = {}
    metadata: Dict[str, Dict[str, object]] = st.session_state[DATASET_METADATA_KEY]
    timestamp = result.retrieved_at
    for dataset, dataframe in result.datasets.items():
        rows = int(len(dataframe)) if dataframe is not None else 0
        record = dict(metadata.get(dataset, {}))
        record.update(
            {
                "label": DATASET_LABEL_MAP.get(dataset, dataset),
                "source": "api",
                "source_label": result.provider,
                "rows": rows,
                "updated_at": timestamp,
            }
        )
        if rows:
            record["status"] = "ready"
            record["message"] = f"{result.provider} ({rows:,}行)"
            record.pop("error", None)
        else:
            record["status"] = "missing"
            record["message"] = f"{result.provider} - データ未取得"
        metadata[dataset] = record
    st.session_state[DATASET_METADATA_KEY] = metadata
    st.session_state["last_data_update"] = timestamp
    st.session_state["current_source"] = "api"
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
GLOBAL_FILTER_PRESETS = ["今月", "先月", "直近30日", "カスタム"]
TARGET_MARGIN_RATE = 0.12
TARGET_SALES_GROWTH = 0.05
TARGET_CASH_RATIO = 0.25
BASE_CASH_BUFFER = 3_000_000.0
KPI_ALERT_THRESHOLD = -0.05
DEFAULT_PRIMARY_COLOR = "#2563EB"
DEFAULT_ACCENT_COLOR = "#1E88E5"
DEFAULT_SUCCESS_COLOR = "#16A34A"
DEFAULT_ERROR_COLOR = "#DC2626"

COST_COLUMN_LABELS = {
    "rent": "家賃",
    "payroll": "人件費",
    "utilities": "光熱費",
    "marketing": "販促費",
    "other_fixed": "その他固定費",
}
_COST_COLOR_FACTORS = [-0.25, -0.1, 0.0, 0.2, 0.4]


def _sanitize_hex_color(value: Optional[str], fallback: str = DEFAULT_ACCENT_COLOR) -> str:
    """Return a normalized 6-digit hex color string."""

    if not value or not isinstance(value, str):
        return fallback
    color = value.strip()
    if not color:
        return fallback
    if not color.startswith("#"):
        color = f"#{color}"
    hex_part = color.lstrip("#")
    if len(hex_part) == 3:
        hex_part = "".join(ch * 2 for ch in hex_part)
    if len(hex_part) != 6:
        return fallback
    try:
        int(hex_part, 16)
    except ValueError:
        return fallback
    return f"#{hex_part.upper()}"


def _adjust_hex_color(hex_color: str, factor: float) -> str:
    """Lighten or darken ``hex_color`` by ``factor`` (-1.0〜1.0)."""

    base = _sanitize_hex_color(hex_color)
    r = int(base[1:3], 16)
    g = int(base[3:5], 16)
    b = int(base[5:7], 16)
    if factor >= 0:
        ratio = min(factor, 1.0)
        target_rgb = (255, 255, 255)
    else:
        ratio = min(-factor, 1.0)
        target_rgb = (0, 0, 0)
    nr = int(round(r + (target_rgb[0] - r) * ratio))
    ng = int(round(g + (target_rgb[1] - g) * ratio))
    nb = int(round(b + (target_rgb[2] - b) * ratio))
    return f"#{nr:02X}{ng:02X}{nb:02X}"


def _get_theme_config() -> Dict[str, str]:
    """Return the current Streamlit theme configuration if available."""

    try:
        theme = st.get_option("theme")
    except RuntimeError:
        return {}
    if not isinstance(theme, dict):
        return {}
    return theme


def _cost_color_mapping(cost_columns: Sequence[str]) -> Tuple[Dict[str, str], str]:
    """Return a mapping of cost columns to theme accent-based colors."""

    theme = _get_theme_config()
    accent = _sanitize_hex_color(theme.get("accentColor"), DEFAULT_ACCENT_COLOR)
    palette = [_adjust_hex_color(accent, factor) for factor in _COST_COLOR_FACTORS]
    mapping = {col: palette[idx % len(palette)] for idx, col in enumerate(cost_columns)}
    return mapping, accent


def _resolve_theme_colors() -> Dict[str, str]:
    """Fetch commonly used theme colors with sensible fallbacks."""

    theme = _get_theme_config()
    primary = _sanitize_hex_color(theme.get("primaryColor"), DEFAULT_PRIMARY_COLOR)
    accent = _sanitize_hex_color(theme.get("accentColor"), DEFAULT_ACCENT_COLOR)
    success = _sanitize_hex_color(theme.get("successColor"), DEFAULT_SUCCESS_COLOR)
    error = _sanitize_hex_color(theme.get("errorColor"), DEFAULT_ERROR_COLOR)
    warning = _sanitize_hex_color(theme.get("warningColor"), "#F59E0B")
    neutral = _sanitize_hex_color(theme.get("secondaryBackgroundColor"), "#F1F5F9")
    text = _sanitize_hex_color(theme.get("textColor"), "#0F172A")
    return {
        "primary": primary,
        "accent": accent,
        "success": success,
        "error": error,
        "warning": warning,
        "neutral": neutral,
        "text": text,
    }


def _categorical_color_map(keys: Sequence[str], base_color: str) -> Dict[str, str]:
    """Generate a perceptually ordered color map for categorical values."""

    factors = [-0.35, -0.2, -0.05, 0.1, 0.25, 0.4, 0.55, 0.7]
    unique_keys = [key for key in keys if key is not None]
    if not unique_keys:
        return {}
    palette = [
        _adjust_hex_color(base_color, factors[idx % len(factors)])
        for idx in range(len(unique_keys))
    ]
    return {key: palette[idx] for idx, key in enumerate(unique_keys)}


def _resolve_target_fixed_cost() -> Optional[float]:
    """Return a user defined fixed cost target when available."""

    for key in ("target_fixed_cost", "simulation_fixed_cost"):
        raw_value = st.session_state.get(key)
        if raw_value is None:
            continue
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            continue
    return None


MESSAGE_DICTIONARY = {
    "empty_sales": {
        "level": "warning",
        "message": "指定期間の売上データがありません。",
        "guidance": "期間を変更すると過去データを確認できます。",
        "action_label": "期間を変更する",
    },
    "loading_data": {
        "level": "info",
        "message": "データを読み込み中です…しばらくお待ちください。",
        "guidance": "バックエンド処理に時間がかかる場合があります。",
        "show_progress": True,
    },
    "error_api": {
        "level": "error",
        "message": "データの取得に失敗しました。ネットワーク接続を確認の上、再試行してください。",
        "guidance": "接続が復旧したらボタンから再取得できます。",
        "action": {"label": "再試行する"},
    },
    "file_format_error": {
        "level": "error",
        "message": "アップロードされたCSVの形式が不正です。テンプレートに沿って修正してください。",
        "guidance": "列名・文字コードをテンプレートと比較してご確認ください。",
        "action": {"label": "テンプレートを確認する"},
    },
    "inventory_warning": {
        "level": "warning",
        "message": "在庫が安全在庫を下回っています：{summary}",
        "guidance": "欠品リスクが高い商品から優先的に補充を検討してください。",
    },
    "simulation_saved": {
        "level": "success",
        "message": "シミュレーション結果を保存しました。",
        "guidance": None,
    },
    "session_timeout": {
        "level": "warning",
        "message": "セッションが切れました。再ログインしてください。",
        "guidance": None,
        "action": {"label": "再ログインする"},
    },
    "complete": {
        "level": "success",
        "message": "データ取り込みが完了しました。更新日時：{timestamp}",
        "guidance": "最新データを反映した分析へ移動できます。",
        "action_label": "売上分析へ",
    },
    "deficit_alert": {
        "level": "error",
        "message": "{store}が営業赤字です（{amount}円）。対策を検討してください。",
        "guidance": "原因分析ページで粗利・固定費の内訳を確認し、改善策を検討しましょう。",
        "action_label": "損益詳細を開く",
    },
}

MESSAGE_DICTIONARY.update(
    {
        "empty": MESSAGE_DICTIONARY["empty_sales"],
        "loading": MESSAGE_DICTIONARY["loading_data"],
        "error": MESSAGE_DICTIONARY["file_format_error"],
        "stock_alert": MESSAGE_DICTIONARY["inventory_warning"],
    }
)


def render_guided_message(
    state_key: str,
    *,
    message_kwargs: Optional[Dict[str, object]] = None,
    action: Optional[Dict[str, object]] = None,
    progress: Optional[float] = None,
) -> None:
    """Render a contextual message with optional actions based on a dictionary key."""

    config = MESSAGE_DICTIONARY.get(state_key)
    if config is None:
        return

    message_kwargs = message_kwargs or {}
    text = config.get("message", "").format(**message_kwargs)
    level = config.get("level", "info")
    guidance = config.get("guidance")
    show_progress = config.get("show_progress", False)

    container = st.container()
    if level == "success":
        container.success(text)
    elif level == "warning":
        container.warning(text)
    elif level == "error":
        container.error(text)
    else:
        container.info(text)

    if guidance:
        container.caption(guidance)

    if show_progress:
        progress_value = progress if progress is not None else 0.0
        container.progress(min(max(progress_value, 0.0), 1.0))

    config_action = dict(config.get("action", {})) if config.get("action") else {}
    if action:
        config_action.update(action)
    label = config_action.get("label") or config.get("action_label")
    if label:
        action_type = config_action.get("type", "button")
        key = config_action.get("key") or f"message-action-{state_key}"
        if action_type == "download":
            data = config_action.get("data")
            file_name = config_action.get("file_name", "download.csv")
            mime = config_action.get("mime", "text/csv")
            if data is not None:
                container.download_button(
                    label,
                    data=data,
                    file_name=file_name,
                    mime=mime,
                    key=key,
                )
        elif action_type == "link":
            url = config_action.get("url")
            if url:
                container.link_button(label, url, key=key)
        else:
            container.button(
                label,
                key=key,
                on_click=config_action.get("on_click"),
                args=config_action.get("args", ()),
                kwargs=config_action.get("kwargs", {}),
            )


def _inject_global_styles() -> None:
    """Inject shared CSS for the redesigned dashboard."""

    colors = _resolve_theme_colors()
    st.markdown(
        f"""
        <style>
        .kpi-card {{
            background: #ffffff;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 2px 4px rgba(15, 23, 42, 0.08);
            margin-bottom: 0.8rem;
            }}
        .kpi-card.alert {{
            background: { _adjust_hex_color(colors['error'], 0.75) };
            border: 1px solid { _adjust_hex_color(colors['error'], 0.4) };
        }}
        .kpi-card.caution {{
            background: { _adjust_hex_color(colors['primary'], 0.85) };
            border: 1px solid { _adjust_hex_color(colors['primary'], 0.55) };
        }}
        .kpi-card .label {{
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 0.4rem;
        }}
        .kpi-card .value {{
            font-size: 1.6rem;
            font-weight: 600;
            color: {colors['text']};
        }}
        .kpi-card .sub-value {{
            font-size: 0.85rem;
            color: #4b5563;
            margin-top: 0.2rem;
        }}
        .kpi-card .sub-value.muted {{
            color: #9ca3af;
        }}
        .kpi-card .delta {{
            font-size: 0.9rem;
            font-weight: 500;
            margin-top: 0.2rem;
        }}
        .kpi-card .delta.positive {{
            color: {colors['success']};
        }}
        .kpi-card .delta.negative {{
            color: {colors['error']};
        }}
        .kpi-card .target {{
            font-size: 0.85rem;
            margin-top: 0.3rem;
            color: #4b5563;
        }}
        .kpi-card .target.positive {{
            color: {colors['success']};
        }}
        .kpi-card .target.negative {{
            color: {colors['error']};
        }}
        .kpi-card-wrapper {{
            position: relative;
            cursor: pointer;
        }}
        .kpi-card-wrapper > div[data-testid="stButton"] {{
            position: absolute;
            inset: 0;
            z-index: 2;
        }}
        .kpi-card-wrapper > div[data-testid="stButton"] button {{
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
            padding: 0;
        }}
        .kpi-card-wrapper .kpi-card {{
            pointer-events: none;
        }}
        .kpi-card-link {{
            text-decoration: none;
            display: block;
        }}
        .kpi-card-link .kpi-card {{
            pointer-events: none;
        }}
        .alert-box {{
            background: { _adjust_hex_color(colors['error'], 0.75) };
            border: 1px solid { _adjust_hex_color(colors['error'], 0.4) };
            border-radius: 12px;
            padding: 1rem 1.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #b42318;
            gap: 1rem;
        }}
        .alert-box.success {{
            background: { _adjust_hex_color(colors['success'], 0.75) };
            border-color: { _adjust_hex_color(colors['success'], 0.4) };
            color: {colors['success']};
        }}
        .alert-banner {{
            background: { _adjust_hex_color(colors['warning'], 0.7) };
            border: 1px solid { _adjust_hex_color(colors['warning'], 0.3) };
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            color: #8a5800;
        }}
        .alert-banner.success {{
            background: { _adjust_hex_color(colors['success'], 0.75) };
            border-color: { _adjust_hex_color(colors['success'], 0.4) };
            color: {colors['success']};
        }}
        .alert-banner__header {{
            font-weight: 600;
        }}
        .alert-card {{
            background: #ffffff;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(11, 31, 59, 0.08);
            border-left: 4px solid {colors['primary']};
            min-height: 150px;
        }}
        .alert-card.high {{
            border-left-color: {colors['error']};
        }}
        .alert-card.medium {{
            border-left-color: {colors['warning']};
        }}
        .alert-card__title {{
            font-weight: 600;
            color: {colors['primary']};
            margin-bottom: 0.3rem;
        }}
        .alert-card__count {{
            font-size: 1.4rem;
            font-weight: 600;
            color: {colors['text']};
        }}
        .alert-card__message {{
            font-size: 0.9rem;
            color: #475569;
            margin-top: 0.4rem;
        }}
        .filter-bar {{
            background: {colors['neutral']};
            border: 1px solid { _adjust_hex_color(colors['neutral'], -0.15) };
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 1rem;
        }}
        .quick-menu button {{
            width: 100%;
        }}
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


def _activate_main_tab(tab_name: str) -> None:
    st.session_state[MAIN_TAB_KEY] = tab_name
    trigger_rerun()


def _activate_inventory_focus(focus: str) -> None:
    if "inventory_tab_state" not in st.session_state:
        st.session_state["inventory_tab_state"] = {}
    tab_state = st.session_state["inventory_tab_state"]
    tab_state["focus"] = focus
    st.session_state[MAIN_TAB_KEY] = "在庫"
    trigger_rerun()


def _render_analysis_navigation(active_label: str) -> str:
    """Render the analysis tab selector without duplicate navigation widgets."""

    st.markdown("### 分析タブ")
    active_index = (
        MAIN_TAB_LABELS.index(active_label)
        if active_label in MAIN_TAB_LABELS
        else 0
    )
    selected_label = st.radio(
        "分析を選択",
        MAIN_TAB_LABELS,
        index=active_index,
        key="main_tab_selector",
        horizontal=True,
    )
    st.session_state[MAIN_TAB_KEY] = selected_label
    st.divider()
    return selected_label


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
        classes = ["kpi-card"]
        if card.get("alert"):
            classes.append("alert")
        elif yoy is not None and yoy <= KPI_ALERT_THRESHOLD:
            classes.append("alert")
        elif target_diff < 0:
            classes.append("caution")

        card_html = (
            f"""
            <div class="{' '.join(classes)}">
                <div class="label">{card.get('label', '')}</div>
                <div class="value">{card.get('value_text', '')}</div>
                <div class="delta {delta_class}">{yoy_text}</div>
                <div class="target {target_class}">目標差: {target_diff:+,.0f}{target_suffix}</div>
            </div>
            """
        )

        action_conf = card.get("action") or {}
        action_type = action_conf.get("type", "button") if action_conf else None
        card_container = column.container()

        if action_conf and action_type != "link":
            wrapper = card_container.container()
            wrapper.markdown("<div class=\"kpi-card-wrapper\">", unsafe_allow_html=True)
            interactive_area = wrapper.container()
            action_key = action_conf.get("key") or f"kpi-card-action-{card.get('label', '')}"
            interactive_area.button(
                action_conf.get("label", " "),
                key=action_key,
                help=action_conf.get("help"),
                on_click=action_conf.get("on_click"),
                args=action_conf.get("args", ()),
                kwargs=action_conf.get("kwargs", {}),
                type=action_conf.get("button_type", "secondary"),
                use_container_width=True,
            )
            interactive_area.markdown(card_html, unsafe_allow_html=True)
            wrapper.markdown("</div>", unsafe_allow_html=True)
        elif action_conf.get("type") == "link":
            url = action_conf.get("url")
            if url:
                card_container.markdown(
                    f"""
                    <a class=\"kpi-card-link\" href=\"{url}\" target=\"_blank\">
                        {card_html}
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                card_container.markdown(card_html, unsafe_allow_html=True)
        else:
            card_container.markdown(card_html, unsafe_allow_html=True)


def _render_kpi_highlights(
    highlights: Sequence[Dict[str, object]], colors: Dict[str, str]
) -> None:
    """Render visually emphasised KPI highlight cards."""

    if not highlights:
        return

    accent_light = _adjust_hex_color(colors["accent"], 0.65)
    border_color = _adjust_hex_color(colors["accent"], -0.45)
    card_html_list: List[str] = []
    for highlight in highlights:
        label = escape(str(highlight.get("label", "")))
        value_text = str(highlight.get("value", "-"))
        tooltip = highlight.get("tooltip")
        tooltip_attr = f' title="{escape(str(tooltip))}"' if tooltip else ""
        delta_ratio = highlight.get("delta")
        delta_text = highlight.get("delta_text")
        if delta_ratio is not None:
            arrow = "▲" if float(delta_ratio) >= 0 else "▼"
            formatted = f"{arrow} {abs(float(delta_ratio)) * 100:.1f}pt"
            delta_text = delta_text or formatted
            delta_color = colors["success"] if float(delta_ratio) >= 0 else colors["error"]
        else:
            delta_text = delta_text or "比較データなし"
            delta_color = colors.get("warning", colors["text"])
        target_label = highlight.get("target_label")
        target_html = (
            f"<div style=\"font-size:0.75rem;color:{colors['text']};opacity:0.65;margin-top:0.45rem;\">{escape(str(target_label))}</div>"
            if target_label
            else ""
        )
        card_html_list.append(
            f"""
            <div style="flex:1;min-width:220px;background-color:{accent_light};border-radius:0.9rem;padding:1.1rem;border:1px solid {border_color};box-shadow:0 8px 24px rgba(15, 23, 42, 0.08);"{tooltip_attr}>
                <div style="font-size:0.85rem;color:{colors['text']};opacity:0.75;font-weight:500;">{label}</div>
                <div style="font-size:2.1rem;font-weight:600;color:{colors['text']};margin:0.35rem 0 0.45rem;">{escape(value_text)}</div>
                <div style="font-size:0.9rem;font-weight:600;color:{delta_color};">{escape(delta_text)}</div>
                {target_html}
            </div>
            """
        )

    st.markdown(
        f"""
        <div style="border-radius:1.1rem;padding:1.35rem;background:linear-gradient(135deg, {accent_light} 0%, rgba(255,255,255,0.95) 65%);border:1px solid {border_color};margin-bottom:1.5rem;">
            <div style="font-size:1rem;font-weight:600;color:{colors['primary']};margin-bottom:0.9rem;">主要KPIハイライト</div>
            <div style="display:flex;flex-wrap:wrap;gap:1rem;">{''.join(card_html_list)}</div>
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


def _collect_alerts(
    datasets: Dict[str, pd.DataFrame],
    alert_settings: Optional[Dict[str, object]],
    default_period: Tuple[date, date],
) -> List[Dict[str, object]]:
    settings = alert_settings or {}
    stockout_threshold = _coerce_int(settings.get("stockout_threshold"), 0)
    excess_threshold = _coerce_int(settings.get("excess_threshold"), 0)
    deficit_threshold = settings.get("deficit_threshold")

    sales_df = datasets.get("sales", pd.DataFrame())
    inventory_df = datasets.get("inventory", pd.DataFrame())
    fixed_costs_df = datasets.get("fixed_costs", pd.DataFrame())

    start_date, end_date = default_period
    alerts: List[Dict[str, object]] = []

    overview_df = inventory.inventory_overview(
        sales_df,
        inventory_df,
        start_date=start_date,
        end_date=end_date,
    )
    if not overview_df.empty:
        stockouts = int((overview_df["stock_status"] == "在庫切れ").sum())
        excess = int((overview_df["stock_status"] == "在庫過多").sum())
        if stockouts > stockout_threshold:
            alerts.append(
                {
                    "title": "欠品アラート",
                    "count": stockouts,
                    "message": f"安全在庫を下回る商品が{stockouts}品目あります。",
                    "action": {
                        "label": "在庫タブで確認",
                        "callback": _activate_inventory_focus,
                        "args": ("stockout",),
                    },
                    "severity": "high",
                }
            )
        if excess > excess_threshold:
            alerts.append(
                {
                    "title": "過剰在庫アラート",
                    "count": excess,
                    "message": f"在庫過多の対象が{excess}品目あります。販促や発注調整を検討してください。",
                    "action": {
                        "label": "在庫タブへ",
                        "callback": _activate_inventory_focus,
                        "args": ("excess",),
                    },
                    "severity": "medium",
                }
            )

    pnl_df = profitability.store_profitability(sales_df, fixed_costs_df)
    if not pnl_df.empty:
        operating_profit_total = float(
            pnl_df.get("operating_profit", pd.Series(dtype=float)).sum()
        )
        loss_stores = int(
            (pnl_df.get("operating_profit", pd.Series(dtype=float)) < 0).sum()
        )
        if deficit_threshold is not None and operating_profit_total < float(deficit_threshold):
            alerts.append(
                {
                    "title": "損益警告",
                    "count": max(loss_stores, 1),
                    "message": (
                        "営業利益が目標値を下回っています。"
                        f" 現在の合計: {operating_profit_total:,.0f} 円"
                    ),
                    "action": {
                        "label": "粗利タブを開く",
                        "callback": _activate_main_tab,
                        "args": ("粗利",),
                    },
                    "severity": "high",
                }
            )
        elif loss_stores > 0:
            alerts.append(
                {
                    "title": "赤字店舗あり",
                    "count": loss_stores,
                    "message": f"営業赤字の店舗が{loss_stores}店あります。費目の見直しを検討してください。",
                    "action": {
                        "label": "粗利タブで確認",
                        "callback": _activate_main_tab,
                        "args": ("粗利",),
                    },
                    "severity": "medium",
                }
            )

    return alerts


def render_alert_center(
    alerts: Sequence[Dict[str, object]],
    alert_settings: Optional[Dict[str, object]],
) -> None:
    settings = alert_settings or {}
    channel = settings.get("notification_channel", "banner")
    total_alerts = int(sum(alert.get("count", 0) or 0 for alert in alerts))
    if "show_alert_modal" not in st.session_state:
        st.session_state["show_alert_modal"] = False
    if total_alerts == 0:
        st.session_state["show_alert_modal"] = False
    contact_info = []
    email = settings.get("notification_email")
    slack_webhook = settings.get("slack_webhook")
    if email:
        contact_info.append(f"メール通知先: {email}")
    if slack_webhook:
        contact_info.append("Slack連携: 登録済み")
    contact_text = "｜".join(contact_info)

    container = st.container()

    if channel == "modal":
        def _open_alert_modal() -> None:
            _set_state_and_rerun("show_alert_modal", True)

        def _close_alert_modal() -> None:
            _set_state_and_rerun("show_alert_modal", False)

        open_key = "alert_center_modal_open"
        if total_alerts > 0:
            container.button(
                f"🔔 アラートを確認 ({total_alerts})",
                key=open_key,
                type="primary",
                on_click=_open_alert_modal,
            )
        else:
            container.success("現在重大なアラートはありません。")

        if st.session_state.get("show_alert_modal") and alerts:
            with st.modal("アラートセンター", key="alert_center_modal"):
                if contact_text:
                    st.caption(contact_text)
                for alert in alerts:
                    st.markdown(
                        f"<div class='alert-card {alert.get('severity', 'medium')}'>"
                        f"<div class='alert-card__title'>{alert.get('title')}</div>"
                        f"<div class='alert-card__count'>{alert.get('count', 0)}件</div>"
                        f"<div class='alert-card__message'>{alert.get('message', '')}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    action = alert.get("action") or {}
                    callback = action.get("callback")
                    if callback:
                        st.button(
                            action.get("label", "詳細を開く"),
                            on_click=callback,
                            args=action.get("args", ()),
                            key=f"alert-action-modal-{alert.get('title')}",
                        )
                st.button(
                    "閉じる",
                    key="close-alert-modal",
                    on_click=_close_alert_modal,
                )
        return

    with container:
        banner_class = "alert-banner" if total_alerts else "alert-banner success"
        header_text = (
            f"🔔 アラートセンター（{total_alerts}件）"
            if total_alerts
            else "✅ アラートセンター"
        )
        container.markdown(
            f"<div class='{banner_class}'>"
            f"<div class='alert-banner__header'>{header_text}</div>"
            f"<div class='alert-banner__body'></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if contact_text:
            st.caption(contact_text)

        if not alerts:
            st.info("在庫・損益ともに基準値内です。分析を進めてください。")
            return

        columns = st.columns(len(alerts))
        for column, alert in zip(columns, alerts):
            column.markdown(
                f"<div class='alert-card {alert.get('severity', 'medium')}'>"
                f"<div class='alert-card__title'>{alert.get('title')}</div>"
                f"<div class='alert-card__count'>{alert.get('count', 0)}件</div>"
                f"<div class='alert-card__message'>{alert.get('message', '')}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            action = alert.get("action") or {}
            callback = action.get("callback")
            if callback:
                column.button(
                    action.get("label", "詳細を見る"),
                    on_click=callback,
                    args=action.get("args", ()),
                    key=f"alert-action-{alert.get('title')}",
                    use_container_width=True,
                )

def render_global_filter_bar(
    stores: Sequence[str],
    categories: Sequence[str],
    *,
    default_period: Tuple[date, date],
    bounds: Tuple[date, date],
) -> transformers.FilterState:
    if "selected_store" not in st.session_state:
        st.session_state["selected_store"] = transformers.ALL_STORES
    saved_store = st.session_state["selected_store"]
    if "date_range" not in st.session_state:
        st.session_state["date_range"] = default_period
    saved_range = st.session_state["date_range"]
    if "selected_period" not in st.session_state:
        st.session_state["selected_period"] = saved_range
    saved_period = st.session_state["selected_period"]
    if GLOBAL_FILTER_KEY not in st.session_state:
        st.session_state[GLOBAL_FILTER_KEY] = {
            "preset": "直近30日",
            "custom_range": saved_period,
            "store": saved_store,
        }
    state = st.session_state[GLOBAL_FILTER_KEY]

    st.markdown("### 共通コントロール")
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
        selected_store = state.get("store", saved_store)
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
    st.session_state["selected_store"] = store_choice
    st.session_state["date_range"] = (start_date, end_date)
    st.session_state["selected_period"] = (start_date, end_date)

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
    st.caption("※ 選択した条件はすべてのタブに適用されます。")
    return filters



def render_dashboard_tab(
    sales_df: pd.DataFrame,
    comparison_sales: pd.DataFrame,
    filters: transformers.FilterState,
    fixed_costs_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    st.markdown("### 経営ダッシュボード")
    colors = _resolve_theme_colors()

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
    total_gross_profit = float(
        pnl_df.get("gross_profit", pd.Series(dtype=float)).sum()
    )
    previous_gross_profit = (
        float(
            comparison_pnl.get("gross_profit", pd.Series(dtype=float)).sum()
        )
        if not comparison_pnl.empty
        else None
    )
    gross_margin_ratio = (
        total_gross_profit / total_sales if total_sales else None
    )
    previous_gross_margin_ratio = (
        previous_gross_profit / previous_sales
        if (previous_gross_profit is not None and previous_sales not in (None, 0))
        else None
    )
    operating_margin_ratio = (
        operating_profit / total_sales if total_sales else None
    )
    previous_operating_margin_ratio = (
        previous_operating_profit / previous_sales
        if (previous_operating_profit is not None and previous_sales not in (None, 0))
        else None
    )
    profit_target = sales_target * TARGET_MARGIN_RATE

    cash_current = _cash_flow_summary(sales_df, filtered_inventory)
    cash_previous = _cash_flow_summary(comparison_sales, filtered_inventory)
    cash_target = sales_target * TARGET_CASH_RATIO
    cash_ratio = (
        cash_current["balance"] / total_sales if total_sales else None
    )
    previous_cash_ratio = (
        cash_previous.get("balance", 0.0) / previous_sales
        if (previous_sales not in (None, 0))
        else None
    )

    overview_df = inventory.inventory_overview(
        sales_df,
        filtered_inventory,
        start_date=filters.start_date,
        end_date=filters.end_date,
    )
    stockouts = (
        int((overview_df["stock_status"] == "在庫切れ").sum())
        if not overview_df.empty
        else 0
    )
    alert_settings = st.session_state.get("alert_settings", {})
    stockout_threshold = _coerce_int(alert_settings.get("stockout_threshold"), 0)

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
                "label": "欠品品目数",
                "value_text": _format_number(stockouts, "品目"),
                "unit": "品目",
                "yoy": None,
                "target_diff": stockout_threshold - stockouts,
                "alert": stockouts > stockout_threshold,
            },
            {
                "label": "資金残高",
                "value_text": _format_currency(cash_current["balance"]),
                "unit": "円",
                "yoy": _compute_growth(
                    cash_current["balance"], cash_previous.get("balance")
                ),
                "target_diff": cash_current["balance"] - cash_target,
                "alert": cash_current["balance"] < cash_target,
            },
        ]
    )
    sales_growth_ratio = _compute_growth(total_sales, previous_sales)
    previous_sales_text = (
        _format_currency(previous_sales) if previous_sales is not None else "データ不足"
    )
    operating_margin_display = (
        _format_ratio(operating_margin_ratio)
        if operating_margin_ratio is not None
        else "データ不足"
    )
    previous_operating_margin_text = (
        _format_ratio(previous_operating_margin_ratio)
        if previous_operating_margin_ratio is not None
        else "データ不足"
    )
    cash_ratio_display = (
        _format_ratio(cash_ratio) if cash_ratio is not None else "データ不足"
    )
    previous_cash_ratio_text = (
        _format_ratio(previous_cash_ratio)
        if previous_cash_ratio is not None
        else "データ不足"
    )
    highlight_metrics = [
        {
            "label": "売上高成長率",
            "value": _format_ratio(sales_growth_ratio)
            if sales_growth_ratio is not None
            else "データ不足",
            "delta": (
                sales_growth_ratio - TARGET_SALES_GROWTH
                if sales_growth_ratio is not None
                else None
            ),
            "target_label": f"目標 {TARGET_SALES_GROWTH:.0%} ／ 前年売上: {previous_sales_text}",
            "tooltip": "前年同期間との比較です。社内基準は前年比+5%です。",
        },
        {
            "label": "営業利益率",
            "value": operating_margin_display,
            "delta": (
                operating_margin_ratio - TARGET_MARGIN_RATE
                if operating_margin_ratio is not None
                else None
            ),
            "target_label": f"目標 {TARGET_MARGIN_RATE:.0%} ／ 営業利益: {_format_currency(operating_profit)}",
            "tooltip": (
                "営業利益 ÷ 売上高。前年利益率: "
                + previous_operating_margin_text
            ),
        },
        {
            "label": "キャッシュ比率",
            "value": cash_ratio_display,
            "delta": (
                cash_ratio - TARGET_CASH_RATIO if cash_ratio is not None else None
            ),
            "target_label": f"目標 {TARGET_CASH_RATIO:.0%} ／ 資金残高: {_format_currency(cash_current['balance'])}",
            "tooltip": (
                "資金残高 ÷ 期間売上。前年比: " + previous_cash_ratio_text
            ),
        },
    ]
    _render_kpi_highlights(highlight_metrics, colors)
    negative_stores = (
        int((pnl_df["operating_profit"] < 0).sum())
        if not pnl_df.empty
        else 0
    )

    if stockouts or negative_stores:
        st.info("ページ上部のアラートセンターに重要な注意事項が表示されています。")

    st.caption(
        "指標カードと主要KPIハイライトは直近実績と前年比較・目標達成度を示します。下部のタブから詳細分析とレポート出力に進んでください。"
    )

    store_label = "、".join(filters.stores) if filters.stores else transformers.ALL_STORES
    category_label = (
        "、".join(filters.categories)
        if filters.categories
        else transformers.ALL_CATEGORIES
    )
    channel_label = (
        "、".join(filters.channels)
        if filters.channels
        else transformers.ALL_CHANNELS
    )
    filters_section = [
        f"- 期間: {filters.start_date:%Y-%m-%d} 〜 {filters.end_date:%Y-%m-%d}",
        f"- 店舗: {store_label}",
        f"- カテゴリ: {category_label}",
        f"- チャネル: {channel_label}",
        f"- 集計粒度: {filters.period_granularity}",
    ]
    highlight_section = [
        f"- 期間売上: {_format_currency(total_sales)}（前年: {previous_sales_text}）",
        f"- 売上高成長率: {_format_ratio(sales_growth_ratio) if sales_growth_ratio is not None else 'データ不足'}",
        f"- 営業利益: {_format_currency(operating_profit)}（利益率: {operating_margin_display}）",
        f"- 粗利率: {_format_ratio(gross_margin_ratio) if gross_margin_ratio is not None else 'データ不足'}",
        f"- 資金残高: {_format_currency(cash_current['balance'])}（キャッシュ比率: {cash_ratio_display}）",
        f"- 欠品品目数: {stockouts} 品目",
    ]
    action_items: List[str] = []
    if sales_growth_ratio is not None and sales_growth_ratio < TARGET_SALES_GROWTH:
        action_items.append(
            "- 売上高成長率が目標未達です。キャンペーン施策や顧客獲得の強化をご検討ください。"
        )
    if operating_margin_ratio is not None and operating_margin_ratio < TARGET_MARGIN_RATE:
        action_items.append(
            "- 営業利益率が目標を下回っています。粗利改善や固定費削減の余地を確認してください。"
        )
    if cash_ratio is not None and cash_ratio < TARGET_CASH_RATIO:
        action_items.append(
            "- キャッシュ比率が低下しています。入金サイクルや在庫回転の改善を検討してください。"
        )
    if stockouts > stockout_threshold:
        action_items.append(
            f"- 欠品が {stockouts} 品目発生しています。安全在庫を下回る商品の補充を優先してください。"
        )
    if not action_items:
        action_items.append("- 主要な懸念事項はありません。現状の運用を継続してください。")

    st.markdown("#### レポート出力")
    st.caption("現在のフィルター条件と主要指標をまとめたMarkdown／PDFレポートをダウンロードできます。")
    report.render_dashboard_report_downloads(
        "松屋 経営ダッシュボード レポート",
        [
            ("フィルター条件", filters_section),
            ("主要指標", highlight_section),
            ("アクションメモ", action_items),
        ],
        base_file_name=f"matsuya_dashboard_{filters.end_date:%Y%m%d}",
    )

    return pnl_df



def render_sales_tab(
    sales_df: pd.DataFrame,
    filters: transformers.FilterState,
    channels: Sequence[str],
    *,
    comparison_mode: str,
) -> None:
    colors = _resolve_theme_colors()
    st.markdown("### 売上分析")

    if "sales_tab_state" not in st.session_state:
        st.session_state["sales_tab_state"] = {
            "channel": transformers.ALL_CHANNELS,
            "granularity": "monthly",
            "breakdown": "store",
            "comparison": comparison_mode,
        }
    state = st.session_state["sales_tab_state"]

    channel_options = (
        [transformers.ALL_CHANNELS, *channels]
        if channels
        else [transformers.ALL_CHANNELS]
    )
    granularity_options = {"月次": "monthly", "週次": "weekly", "日次": "daily"}
    breakdown_options = {
        "店舗別": "store",
        "チャネル別": "channel",
        "カテゴリ別": "category",
        "地域別": "region",
    }
    comparison_options = {"前年比": "yoy", "対前期比": "previous_period"}

    with st.container():
        control_cols = st.columns([2, 3, 3])
        channel_index = channel_options.index(
            state.get("channel", transformers.ALL_CHANNELS)
        )
        if len(channel_options) <= 4:
            channel_choice = control_cols[0].radio(
                "チャネル",
                channel_options,
                index=channel_index,
                horizontal=True,
            )
        else:
            channel_choice = control_cols[0].selectbox(
                "チャネル",
                channel_options,
                index=channel_index,
            )

        breakdown_label = control_cols[1].radio(
            "表示単位",
            list(breakdown_options.keys()),
            index=list(breakdown_options.values()).index(
                state.get("breakdown", "store")
            ),
            horizontal=True,
        )
        comparison_label = control_cols[2].radio(
            "比較",
            list(comparison_options.keys()),
            index=list(comparison_options.values()).index(
                state.get("comparison", comparison_mode)
            ),
            horizontal=True,
        )

    granularity_label = st.radio(
        "粒度",
        list(granularity_options.keys()),
        index=list(granularity_options.values()).index(
            state.get("granularity", "monthly")
        ),
        horizontal=True,
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

    filtered_sales, invalid_current = _coerce_datetime_column(filtered_sales)
    comparison_sales, invalid_comparison = _coerce_datetime_column(
        comparison_sales
    )
    invalid_rows = invalid_current + invalid_comparison
    if invalid_rows:
        st.warning(f"日付形式が不正なデータ{invalid_rows}件を除外しました。")

    if filtered_sales.empty:
        render_guided_message("empty")
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

    st.markdown("#### 指標カード")
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
    try:
        timeseries_df = sales.timeseries_with_comparison(
            filtered_sales,
            comparison_sales,
            view_filters.period_granularity,
            breakdown_column,
        )
        custom_data_cols = ["period_key"]
        if breakdown_column:
            custom_data_cols.append(breakdown_column)
        color_map: Dict[str, str] = {}
        if breakdown_column:
            color_map = _categorical_color_map(
                timeseries_df[breakdown_column].dropna().unique().tolist(),
                colors["primary"],
            )
        timeseries_chart = px.line(
            timeseries_df,
            x="period_label",
            y="sales_amount",
            color=breakdown_column if breakdown_column else None,
            markers=True,
            labels={"period_label": "期間", "sales_amount": "売上金額（円）"},
            custom_data=custom_data_cols,
            color_discrete_map=color_map if breakdown_column else None,
        )
        if not breakdown_column:
            timeseries_chart.update_traces(line=dict(color=colors["primary"], width=3))
        timeseries_chart.update_layout(
            xaxis=dict(title="期間"),
            hovermode="x unified",
            yaxis=dict(title="売上金額（円）", tickformat=",.0f"),
            legend=dict(
                title=breakdown_label if breakdown_column else None,
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
        )
        timeseries_chart.update_traces(
            hovertemplate="期間: %{x}<br>売上: %{y:,.0f} 円<extra></extra>"
        )
        if timeseries_df["comparison_sales"].notna().any():
            timeseries_chart.add_trace(
                go.Scatter(
                    x=timeseries_df["period_label"],
                    y=timeseries_df["comparison_sales"],
                    name="前年同月",
                    mode="lines",
                    line=dict(color=colors["accent"], dash="dash"),
                    hovertemplate="<b>%{x}</b><br>前年同月: %{y:,.0f} 円<extra></extra>",
                )
            )

        composition_df = sales.aggregate_timeseries(
            filtered_sales,
            view_filters.period_granularity,
            "channel",
        )
        layout_cols = st.columns([3, 2], gap="large")
        selected_channels: List[str] = []
        trend_events: List[Dict[str, object]] = []

        with layout_cols[0]:
            st.subheader(f"売上推移（{breakdown_label}）")
            st.caption("ドラッグで期間選択すると明細が自動で絞り込まれます。")
            trend_events = plotly_events(
                timeseries_chart,
                select_event=True,
                click_event=True,
                override_width="100%",
                override_height=420,
                key="sales_trend_events",
            )

        with layout_cols[1]:
            st.subheader("チャネル別構成比")
            if not composition_df.empty:
                composition_df = composition_df.sort_values(
                    ["period_start", "sales_amount"], ascending=[True, False]
                ).copy()
                totals = composition_df.groupby("period_key")["sales_amount"].transform(
                    "sum"
                )
                composition_df["share_ratio"] = (
                    composition_df["sales_amount"] / totals.replace(0, pd.NA)
                ).fillna(0.0)
                composition_df["channel_display"] = (
                    composition_df["channel"].fillna("未分類")
                )
                channel_color_map = _categorical_color_map(
                    composition_df["channel_display"].unique().tolist(),
                    colors["primary"],
                )
                channel_composition_chart = go.Figure()
                channel_order = (
                    composition_df[["channel_display"]]
                    .drop_duplicates()["channel_display"]
                    .tolist()
                )
                for channel_name in channel_order:
                    channel_data = composition_df[
                        composition_df["channel_display"] == channel_name
                    ]
                    if channel_data.empty:
                        continue
                    custom_payload = [
                        (
                            channel_name,
                            float(amount),
                            float(ratio),
                        )
                        for amount, ratio in zip(
                            channel_data["sales_amount"],
                            channel_data["share_ratio"],
                        )
                    ]
                    channel_composition_chart.add_trace(
                        go.Bar(
                            x=channel_data["period_label"],
                            y=channel_data["share_ratio"],
                            name=channel_name,
                            marker_color=channel_color_map.get(
                                channel_name, colors["primary"]
                            ),
                            customdata=custom_payload,
                            hovertemplate=(
                                "期間: %{x}<br>チャネル: %{customdata[0]}"
                                "<br>構成比: %{customdata[2]:.1%}<br>売上: %{customdata[1]:,.0f} 円"
                                "<extra></extra>"
                            ),
                        )
                    )
                channel_composition_chart.update_layout(
                    barmode="stack",
                    yaxis=dict(title="構成比（%）", tickformat=".0%"),
                    xaxis=dict(title="期間"),
                    legend=dict(
                        title="チャネル",
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02,
                    ),
                    hovermode="x unified",
                )
                channel_events = plotly_events(
                    channel_composition_chart,
                    click_event=True,
                    select_event=True,
                    override_width="100%",
                    override_height=360,
                    key="sales_channel_events",
                )
                selected_set: set[str] = set()
                for event in channel_events:
                    if not isinstance(event, dict):
                        continue
                    customdata = event.get("customdata")
                    if not isinstance(customdata, (list, tuple)) or not customdata:
                        continue
                    channel_name = str(customdata[0] or "未分類")
                    selected_set.add(channel_name)
                selected_channels = sorted(selected_set)
                if selected_channels:
                    st.caption("選択中のチャネル: " + "、".join(selected_channels))
                else:
                    st.caption("棒をクリックするとチャネル別に明細を絞り込めます。")
                with st.expander("チャネル別明細", expanded=False):
                    composition_table = composition_df.copy()
                    composition_table["構成比"] = composition_table["share_ratio"].map(
                        lambda value: f"{value*100:.1f}%"
                    )
                    composition_table = composition_table.rename(
                        columns={
                            "period_label": "期間",
                            "channel_display": "チャネル",
                            "sales_amount": "売上金額",
                        }
                    )
                    st.dataframe(
                        composition_table[["期間", "チャネル", "売上金額", "構成比"]],
                        use_container_width=True,
                    )
            else:
                st.caption("チャネル別の集計データがありません。")
    except Exception as exc:  # pragma: no cover - UI guard
        logger.exception("Failed to render sales trend chart")
        st.error(f"売上チャートの描画に失敗しました: {exc}")
        return

    detail_expander = st.expander("売上明細と出力", expanded=False)
    with detail_expander:
        st.markdown("#### 売上明細")
        if trend_events:
            detail_df = sales.drilldown_details(
                filtered_sales,
                trend_events,
                view_filters.period_granularity,
                breakdown_column,
            )
            if selected_channels:
                if "チャネル" in detail_df.columns:
                    detail_df["チャネル"] = detail_df["チャネル"].fillna("未分類")
                    detail_df = detail_df[detail_df["チャネル"].isin(selected_channels)]
                elif "channel" in detail_df.columns:
                    detail_df["channel"] = detail_df["channel"].fillna("未分類")
                    detail_df = detail_df[detail_df["channel"].isin(selected_channels)]
        else:
            detail_columns = [
                "date",
                "store",
                "category",
                "region",
                "channel",
                "product",
                "sales_amount",
                "gross_profit",
                "sales_qty",
            ]
            detail_df = filtered_sales[detail_columns].copy()
            detail_df["channel"] = detail_df["channel"].fillna("未分類")
            if selected_channels:
                detail_df = detail_df[detail_df["channel"].isin(selected_channels)]
            detail_df = detail_df.sort_values("date", ascending=False)
            detail_df["gross_margin"] = (
                detail_df["gross_profit"]
                / detail_df["sales_amount"].replace(0, pd.NA)
            ).fillna(0.0)
            detail_df = detail_df.rename(
                columns={
                    "date": "日付",
                    "store": "店舗",
                    "category": "カテゴリ",
                    "region": "地域",
                    "channel": "チャネル",
                    "product": "商品",
                    "sales_amount": "売上",
                    "gross_profit": "粗利",
                    "sales_qty": "販売数量",
                    "gross_margin": "粗利率",
                }
            )

        export_df = detail_df
        if "チャネル" in export_df.columns:
            export_df["チャネル"] = export_df["チャネル"].fillna("未分類")
        download_cols = st.columns(2)
        with download_cols[0]:
            report.csv_download(
                "売上データをCSV出力",
                export_df,
                file_name=f"matsuya_sales_{filters.start_date:%Y%m%d}_{filters.end_date:%Y%m%d}.csv",
            )
        with download_cols[1]:
            report.pdf_download(
                "売上データをPDF出力",
                "売上サマリー",
                export_df,
                file_name=f"matsuya_sales_{filters.start_date:%Y%m%d}.pdf",
            )

        if export_df.empty:
            st.info("選択した条件に該当する売上明細がありません。")
        else:
            st.dataframe(
                export_df.style.format(
                    {
                        "売上": "{:,.0f}",
                        "粗利": "{:,.0f}",
                        "販売数量": "{:,.1f}",
                        "粗利率": "{:.1%}",
                    }
                ),
                use_container_width=True,
            )

def render_products_tab(
    sales_df: pd.DataFrame,
    comparison_sales: pd.DataFrame,
    filters: transformers.FilterState,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    st.markdown("### 商品分析")

    category_options = sorted(sales_df["category"].dropna().unique().tolist())
    category_choices = [transformers.ALL_CATEGORIES, *category_options]
    if "products_tab_state" not in st.session_state:
        st.session_state["products_tab_state"] = {
            "category": transformers.ALL_CATEGORIES
        }
    state = st.session_state["products_tab_state"]
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
                "label": "トップ5売上",
                "value_text": f"{top5_sales:,.0f} 円",
                "unit": "円",
                "yoy": _compute_growth(top5_sales, previous_top5),
                "target_diff": top5_sales - target_value,
            },
            {
                "label": "トップ5構成比",
                "value_text": f"{share*100:.1f}%",
                "unit": "%",
                "yoy": None,
                "target_diff": (share - target_share) * 100,
            },
        ]
    )

    abc_df = products.abc_analysis(working_sales, working_comparison)
    if abc_df.empty:
        render_guided_message("empty")
        return abc_df, abc_df

    st.subheader("ABC分析とAランク動向")
    pareto_df = products.pareto_chart_data(abc_df)
    pareto_df = pareto_df.copy()
    pareto_df["cumulative_pct"] = pareto_df["cumulative_share"] * 100

    rank_series = (
        pareto_df["rank"]
        if "rank" in pareto_df.columns
        else pd.Series([None] * len(pareto_df), index=pareto_df.index)
    )
    rank_series = rank_series.fillna("-")
    rank_list = rank_series.tolist()
    rank_colors = {"A": "#EF553B", "B": "#FFA15A", "C": "#636EFA"}
    marker_colors = [
        rank_colors.get(rank, "#B0BEC5") if isinstance(rank, str) else "#B0BEC5"
        for rank in rank_list
    ]
    bar_customdata = list(zip(pareto_df["cumulative_pct"], rank_list))

    pareto_chart = go.Figure()
    pareto_chart.add_bar(
        x=pareto_df["product"],
        y=pareto_df["sales_amount"],
        name="売上",
        marker_color=marker_colors,
        customdata=bar_customdata,
        hovertemplate=(
            "<b>%{x}</b><br>売上: %{y:,.0f} 円"
            "<br>累積構成比: %{customdata[0]:.1f}%"
            "<br>ランク: %{customdata[1]}<extra></extra>"
        ),
    )

    pareto_chart.add_trace(
        go.Scatter(
            x=pareto_df["product"],
            y=pareto_df["cumulative_pct"],
            mode="lines+markers",
            name="累積構成比（％）",
            yaxis="y2",
            customdata=rank_list,
            hovertemplate=(
                "<b>%{x}</b><br>累積構成比: %{y:.1f}%"
                "<br>ランク: %{customdata}<extra></extra>"
            ),
        )
    )

    boundary_color = "#9467BD"
    cumulative_pct = pareto_df["cumulative_pct"]
    boundary_product = (
        pareto_df.loc[cumulative_pct >= 80, "product"].iloc[0]
        if (cumulative_pct >= 80).any()
        else pareto_df["product"].iloc[-1]
    )
    pareto_chart.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=boundary_product,
        x1=boundary_product,
        y0=0,
        y1=1,
        line=dict(color=boundary_color, dash="dash"),
    )
    pareto_chart.add_annotation(
        x=boundary_product,
        y=1,
        yref="paper",
        text="Aランク境界",
        showarrow=False,
        yanchor="bottom",
        font=dict(color=boundary_color),
        bgcolor="rgba(255, 255, 255, 0.8)",
    )
    pareto_chart.add_trace(
        go.Scatter(
            x=[boundary_product, boundary_product],
            y=[0, 100],
            mode="lines",
            line=dict(color=boundary_color, dash="dash"),
            name="Aランク境界",
            hoverinfo="skip",
            showlegend=True,
            visible="legendonly",
            yaxis="y2",
        )
    )

    pareto_chart.update_layout(
        xaxis=dict(title="商品"),
        yaxis=dict(title="売上金額（円）", tickformat=",.0f"),
        yaxis2=dict(
            title="累積構成比（％）",
            overlaying="y",
            side="right",
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        hovermode="x unified",
    )

    chart_col, growth_col = st.columns([3, 2])
    with chart_col:
        st.plotly_chart(pareto_chart, use_container_width=True)

    top_growth = products.top_growth_products(abc_df)
    with growth_col:
        st.markdown("#### Aランク伸長率トップ3")
        if top_growth.empty:
            st.info("前年データが不足しているため、伸長率を計算できません。")
        else:
            for _, row in top_growth.iterrows():
                st.metric(
                    row["product"],
                    f"{row['sales_amount']:,.0f} 円",
                    f"{row['yoy_growth']*100:.1f}%",
                )

    with st.expander("ABC分析テーブル", expanded=False):
        st.dataframe(abc_df, use_container_width=True)

    st.markdown("#### トップ5商品一覧")
    top5_display = top5.copy()
    top5_display["構成比"] = (
        top5_display["sales_amount"] / total_sales if total_sales else 0.0
    )
    top5_display = top5_display.rename(columns={"product": "商品", "sales_amount": "売上"})
    st.dataframe(
        top5_display.assign(
            構成比=lambda df: df["構成比"].map(lambda value: f"{value*100:.1f}%")
        ),
        use_container_width=True,
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
    colors = _resolve_theme_colors()
    st.markdown("### 利益管理")

    if "profit_tab_state" not in st.session_state:
        st.session_state["profit_tab_state"] = {
            "selected_store": transformers.ALL_STORES
        }
    state = st.session_state["profit_tab_state"]
    alert_settings = st.session_state.get("alert_settings", {})
    deficit_threshold = _coerce_float(alert_settings.get("deficit_threshold"), 0.0)
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

    def _focus_profit_store(store_name: str) -> None:
        if "profit_tab_state" not in st.session_state:
            st.session_state["profit_tab_state"] = {}
        tab_state = st.session_state["profit_tab_state"]
        tab_state["selected_store"] = store_name

    negative = pnl_df[pnl_df["operating_profit"] <= deficit_threshold]
    if not negative.empty:
        worst = negative.sort_values("operating_profit").iloc[0]
        render_guided_message(
            "deficit_alert",
            message_kwargs={
                "store": worst["store"],
                "amount": f"{worst['operating_profit']:,.0f}",
            },
            action={
                "on_click": _focus_profit_store,
                "args": (worst["store"],),
                "key": "deficit-alert-action",
            },
        )

    store_options = [
        transformers.ALL_STORES,
        *sorted(pnl_df.get("store", pd.Series(dtype=str)).dropna().unique().tolist()),
    ]
    selected_store = state.get("selected_store", transformers.ALL_STORES)
    if selected_store not in store_options:
        selected_store = transformers.ALL_STORES

    store_choice = st.selectbox(
        "フォーカス店舗",
        store_options,
        index=store_options.index(selected_store),
    )
    state["selected_store"] = store_choice

    if store_choice != transformers.ALL_STORES:
        focused_df = pnl_df[pnl_df["store"] == store_choice]
        focused_comparison = comparison_pnl[comparison_pnl["store"] == store_choice]
    else:
        focused_df = pnl_df
        focused_comparison = comparison_pnl

    total_sales = float(focused_df["sales_amount"].sum())
    gross_profit = float(focused_df["gross_profit"].sum())
    operating_profit = float(focused_df["operating_profit"].sum())
    gross_margin = gross_profit / total_sales if total_sales else 0.0

    previous_operating = (
        float(focused_comparison["operating_profit"].sum())
        if not focused_comparison.empty
        else None
    )
    previous_gross = (
        float(focused_comparison["gross_profit"].sum())
        if not focused_comparison.empty
        else None
    )
    previous_sales_total = (
        float(focused_comparison["sales_amount"].sum())
        if not focused_comparison.empty
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

    trend_sales_df = sales_df.copy()
    trend_comparison_df = comparison_sales.copy()
    if store_choice != transformers.ALL_STORES:
        trend_sales_df = trend_sales_df[trend_sales_df["store"] == store_choice]
        trend_comparison_df = trend_comparison_df[
            trend_comparison_df["store"] == store_choice
        ]

    trend_granularity = filters.period_granularity or "monthly"
    profit_trend_df = sales.aggregate_timeseries(trend_sales_df, trend_granularity)
    comparison_trend_df = (
        sales.aggregate_timeseries(trend_comparison_df, trend_granularity)
        if not trend_comparison_df.empty
        else pd.DataFrame(columns=profit_trend_df.columns)
    )
    profit_trend_fig: Optional[go.Figure] = None
    if "store" in edited_df.columns and store_choice != transformers.ALL_STORES:
        fixed_scope = edited_df[edited_df["store"] == store_choice]
    else:
        fixed_scope = edited_df
    total_fixed_cost_scope = float(fixed_scope[cost_columns].fillna(0).sum().sum())
    if not profit_trend_df.empty:
        total_sales_for_alloc = float(profit_trend_df["sales_amount"].sum())
        if total_sales_for_alloc:
            profit_trend_df["allocated_fixed"] = (
                profit_trend_df["sales_amount"] / total_sales_for_alloc
            ) * total_fixed_cost_scope
        else:
            profit_trend_df["allocated_fixed"] = total_fixed_cost_scope / max(
                len(profit_trend_df), 1
            )
        profit_trend_df["operating_profit"] = (
            profit_trend_df["gross_profit"] - profit_trend_df["allocated_fixed"]
        )

    if not comparison_trend_df.empty:
        total_sales_comparison = float(comparison_trend_df["sales_amount"].sum())
        if total_sales_comparison:
            comparison_trend_df["allocated_fixed"] = (
                comparison_trend_df["sales_amount"] / total_sales_comparison
            ) * total_fixed_cost_scope
        else:
            comparison_trend_df["allocated_fixed"] = total_fixed_cost_scope / max(
                len(comparison_trend_df), 1
            )
        comparison_trend_df["operating_profit"] = (
            comparison_trend_df["gross_profit"]
            - comparison_trend_df["allocated_fixed"]
        )

    ranking_df = pnl_df.sort_values("sales_amount", ascending=False)
    comparison_fig = go.Figure()
    metric_config = [
        ("sales_amount", "売上", colors["primary"]),
        ("gross_profit", "粗利", colors["accent"]),
        ("operating_profit", "営業利益", _adjust_hex_color(colors["accent"], -0.2)),
    ]
    stores_display = ranking_df.get("store", pd.Series(dtype=str)).fillna("-")
    for column, label, color in metric_config:
        if column not in ranking_df:
            continue
        comparison_fig.add_trace(
            go.Bar(
                y=stores_display.tolist(),
                x=ranking_df[column].astype(float).tolist(),
                name=label,
                orientation="h",
                marker_color=color,
                hovertemplate=f"店舗: %{{y}}<br>{label}: %{{x:,.0f}} 円<extra></extra>",
            )
        )
    comparison_fig.update_layout(
        barmode="group",
        yaxis=dict(autorange="reversed", title="店舗"),
        xaxis=dict(title="金額（円）", tickformat=",.0f"),
        legend=dict(
            title="指標",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )

    if not profit_trend_df.empty:
        profit_trend_fig = go.Figure()
        profit_trend_fig.add_trace(
            go.Scatter(
                x=profit_trend_df["period_label"],
                y=profit_trend_df["gross_profit"],
                mode="lines+markers",
                name="粗利",
                line=dict(color=colors["accent"], width=2),
                hovertemplate="期間: %{x}<br>粗利: %{y:,.0f} 円<extra></extra>",
            )
        )
        if "operating_profit" in profit_trend_df:
            profit_trend_fig.add_trace(
                go.Scatter(
                    x=profit_trend_df["period_label"],
                    y=profit_trend_df["operating_profit"],
                    mode="lines+markers",
                    name="営業利益",
                    line=dict(color=_adjust_hex_color(colors["accent"], -0.2), width=2),
                    hovertemplate="期間: %{x}<br>営業利益: %{y:,.0f} 円<extra></extra>",
                )
            )
        if not comparison_trend_df.empty:
            profit_trend_fig.add_trace(
                go.Scatter(
                    x=comparison_trend_df["period_label"],
                    y=comparison_trend_df["gross_profit"],
                    mode="lines",
                    name="前年粗利",
                    line=dict(color=_adjust_hex_color(colors["accent"], 0.35), dash="dash"),
                    hovertemplate="期間: %{x}<br>前年粗利: %{y:,.0f} 円<extra></extra>",
                )
            )
            if "operating_profit" in comparison_trend_df:
                profit_trend_fig.add_trace(
                    go.Scatter(
                        x=comparison_trend_df["period_label"],
                        y=comparison_trend_df["operating_profit"],
                        mode="lines",
                        name="前年営業利益",
                        line=dict(
                            color=_adjust_hex_color(colors["accent"], 0.15),
                            dash="dot",
                        ),
                        hovertemplate="期間: %{x}<br>前年営業利益: %{y:,.0f} 円<extra></extra>",
                    )
                )
        profit_trend_fig.update_layout(
            xaxis=dict(title="期間"),
            yaxis=dict(title="金額（円）", tickformat=",.0f"),
            hovermode="x unified",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
        )

    trend_cols = st.columns([3, 2], gap="large")
    with trend_cols[0]:
        st.subheader("粗利・営業利益トレンド")
        if profit_trend_fig is not None:
            st.plotly_chart(profit_trend_fig, use_container_width=True)
        else:
            st.info("利益トレンドを表示できるデータがありません。")

    with trend_cols[1]:
        st.subheader("店舗別売上・利益比較")
        st.caption("棒の長さで店舗ごとの売上・利益規模を比較できます。")
        st.plotly_chart(comparison_fig, use_container_width=True)

    has_store_column = "store" in edited_df.columns
    if has_store_column:
        if store_choice == transformers.ALL_STORES:
            breakdown_series = edited_df[cost_columns].sum()
        else:
            breakdown_series = (
                edited_df[edited_df["store"] == store_choice][cost_columns]
            ).sum()
    else:
        breakdown_series = edited_df[cost_columns].sum()

    breakdown_series = breakdown_series.reindex(cost_columns).fillna(0)
    breakdown_df = (
        breakdown_series.reset_index(name="金額").rename(columns={"index": "項目"})
    )
    breakdown_df["項目"] = breakdown_df["項目"].map(
        lambda key: COST_COLUMN_LABELS.get(key, key)
    )

    color_map, accent_color = _cost_color_mapping(cost_columns)
    cost_chart = go.Figure()
    if has_store_column and store_choice == transformers.ALL_STORES:
        chart_source = (
            edited_df.groupby("store")[cost_columns]
            .sum()
            .reindex(columns=cost_columns, fill_value=0)
            .reset_index()
        )
        x_values = chart_source["store"].astype(str).tolist()
        for column in cost_columns:
            if column not in chart_source:
                continue
            cost_chart.add_trace(
                go.Bar(
                    x=x_values,
                    y=chart_source[column].astype(float).tolist(),
                    name=COST_COLUMN_LABELS.get(column, column),
                    marker_color=color_map[column],
                    hovertemplate="%{x}<br>%{fullData.name}: %{y:,.0f}円<extra></extra>",
                )
            )
        total_series = chart_source[cost_columns].sum(axis=1)
        cost_chart.add_trace(
            go.Scatter(
                x=x_values,
                y=total_series.astype(float).tolist(),
                mode="lines+markers",
                name="固定費合計",
                marker_color=_adjust_hex_color(accent_color, -0.25),
                hovertemplate="%{x}<br>固定費合計: %{y:,.0f}円<extra></extra>",
            )
        )
    else:
        if has_store_column and store_choice != transformers.ALL_STORES:
            store_df = edited_df[edited_df["store"] == store_choice]
        else:
            store_df = edited_df
        aggregated_costs = (
            store_df[cost_columns].fillna(0).sum().reindex(cost_columns, fill_value=0)
        )
        stack_label = (
            store_choice if store_choice != transformers.ALL_STORES else "全店舗"
        )
        if not has_store_column and store_choice == transformers.ALL_STORES:
            stack_label = "全体"
        stack_label = str(stack_label)
        for column in cost_columns:
            cost_chart.add_trace(
                go.Bar(
                    x=[stack_label],
                    y=[float(aggregated_costs.get(column, 0.0))],
                    name=COST_COLUMN_LABELS.get(column, column),
                    marker_color=color_map[column],
                    hovertemplate="%{x}<br>%{fullData.name}: %{y:,.0f}円<extra></extra>",
                )
            )
        total_value = float(aggregated_costs.sum())
        cost_chart.add_trace(
            go.Scatter(
                x=[stack_label],
                y=[total_value],
                mode="lines+markers",
                name="固定費合計",
                marker_color=_adjust_hex_color(accent_color, -0.25),
                hovertemplate="%{x}<br>固定費合計: %{y:,.0f}円<extra></extra>",
            )
        )

    xaxis_title = (
        "店舗" if has_store_column and store_choice == transformers.ALL_STORES else "固定費内訳"
    )
    cost_chart.update_layout(
        barmode="stack",
        xaxis=dict(title=xaxis_title),
        yaxis=dict(title="金額（円）", tickformat=",.0f"),
        legend=dict(
            title=dict(text="費目"),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        hovermode="x unified",
    )

    target_fixed_cost = _resolve_target_fixed_cost()
    if target_fixed_cost is not None:
        line_color = _adjust_hex_color(accent_color, -0.35)
        cost_chart.add_hline(
            y=target_fixed_cost,
            line_dash="dash",
            line_color=line_color,
            annotation_text=f"目標固定費：{target_fixed_cost:,.0f} 円",
            annotation_position="top left",
            annotation=dict(
                font=dict(color=line_color),
                bgcolor="rgba(255, 255, 255, 0.8)",
            ),
        )

    st.subheader("固定費内訳")
    cost_cols = st.columns([3, 2], gap="large")
    with cost_cols[0]:
        st.plotly_chart(cost_chart, use_container_width=True)
    with cost_cols[1]:
        with st.expander("費目別金額", expanded=False):
            st.dataframe(breakdown_df, use_container_width=True)

    styled = focused_df.style.format(
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

    with st.expander("損益明細を表示", expanded=False):
        st.dataframe(styled, use_container_width=True)

    if navigate is not None:
        st.info("シミュレーションで目標利益を検討できます。")
        st.button(
            "シミュレーションを開く",
            key="profit_to_sim",
            on_click=navigate,
            args=("simulation",),
        )

    return pnl_df
def render_inventory_tab(
    sales_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    abc_df: Optional[pd.DataFrame],
    filters: transformers.FilterState,
) -> None:
    colors = _resolve_theme_colors()
    st.markdown("### 在庫分析")

    store_options = sorted(inventory_df["store"].dropna().unique().tolist())
    category_options = sorted(inventory_df["category"].dropna().unique().tolist())
    store_choices = [transformers.ALL_STORES, *store_options]
    category_choices = [transformers.ALL_CATEGORIES, *category_options]

    if "inventory_tab_state" not in st.session_state:
        st.session_state["inventory_tab_state"] = {
            "store": filters.store,
            "category": filters.category,
            "focus": "all",
        }
    state = st.session_state["inventory_tab_state"]
    alert_settings = st.session_state.get("alert_settings", {})
    stockout_threshold = int(alert_settings.get("stockout_threshold", 0))
    excess_threshold = int(alert_settings.get("excess_threshold", 5))

    def _build_inventory_timeseries(
        sales_subset: pd.DataFrame,
        inventory_subset: pd.DataFrame,
        *,
        rolling_window: int,
        safety_buffer: int,
    ) -> pd.DataFrame:
        if inventory_subset.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "estimated_stock",
                    "safety_stock",
                    "moving_stock",
                    "safety_lower",
                    "safety_upper",
                ]
            )

        date_range = pd.date_range(filters.start_date, filters.end_date, freq="D")
        records = []
        window = max(int(rolling_window), 1)
        buffer_days = max(int(safety_buffer), 0)

        for _, row in inventory_subset.iterrows():
            store_name = row.get("store")
            product_name = row.get("product")
            available = float(row.get("opening_stock", 0)) + float(
                row.get("planned_purchase", 0)
            )
            safety = float(row.get("safety_stock", 0))
            product_sales = sales_subset.copy()
            if "store" in product_sales.columns:
                product_sales = product_sales[product_sales["store"] == store_name]
            if "product" in product_sales.columns:
                product_sales = product_sales[product_sales["product"] == product_name]
            daily_sales = (
                product_sales.groupby(product_sales["date"].dt.floor("D"))["sales_qty"]
                .sum()
                .reindex(date_range, fill_value=0.0)
            )
            remaining = (available - daily_sales.cumsum()).clip(lower=0.0)
            moving_stock = remaining.rolling(window=window, min_periods=1).mean()
            avg_daily_sales = daily_sales.rolling(window=window, min_periods=1).mean()

            records.append(
                pd.DataFrame(
                    {
                        "date": date_range,
                        "store": store_name,
                        "estimated_stock": remaining,
                        "safety_stock": safety,
                        "moving_stock": moving_stock,
                        "daily_sales": daily_sales,
                        "avg_daily_sales": avg_daily_sales,
                    }
                )
            )

        if not records:
            return pd.DataFrame(
                columns=[
                    "date",
                    "estimated_stock",
                    "safety_stock",
                    "moving_stock",
                    "safety_lower",
                    "safety_upper",
                ]
            )

        combined = pd.concat(records)
        aggregated = (
            combined.groupby("date")
            .agg(
                estimated_stock=("estimated_stock", "sum"),
                safety_stock=("safety_stock", "sum"),
                moving_stock=("moving_stock", "sum"),
                daily_sales=("daily_sales", "sum"),
                avg_daily_sales=("avg_daily_sales", "sum"),
            )
            .reset_index()
        )
        aggregated["moving_stock"] = aggregated["moving_stock"].rolling(
            window=window, min_periods=1
        ).mean()
        aggregated["avg_daily_sales"] = aggregated["avg_daily_sales"].rolling(
            window=window, min_periods=1
        ).mean()
        aggregated["safety_lower"] = (
            (aggregated["safety_stock"] - aggregated["avg_daily_sales"] * buffer_days)
            .clip(lower=0.0)
            .fillna(0.0)
        )
        aggregated["safety_upper"] = (
            aggregated["safety_stock"] + aggregated["avg_daily_sales"] * buffer_days
        )
        return aggregated

    def _focus_stockouts() -> None:
        if "inventory_tab_state" not in st.session_state:
            st.session_state["inventory_tab_state"] = {}
        tab_state = st.session_state["inventory_tab_state"]
        tab_state["focus"] = "stockout"

    with st.container():
        col1, col2 = st.columns(2)
        store_choice = col1.selectbox(
            "店舗",
            store_choices,
            index=store_choices.index(state.get("store", store_choices[0]))
            if state.get("store") in store_choices
            else 0,
            key="inventory_store_select",
        )
        category_choice = col2.selectbox(
            "カテゴリ",
            category_choices,
            index=category_choices.index(state.get("category", category_choices[0]))
            if state.get("category") in category_choices
            else 0,
            key="inventory_category_select",
        )
    state.update({"store": store_choice, "category": category_choice})

    focus_map = {"全件": "all", "欠品のみ": "stockout", "過剰のみ": "excess"}
    focus_values = list(focus_map.values())
    focus_labels = list(focus_map.keys())
    current_focus = state.get("focus", "all")
    focus_index = focus_values.index(current_focus) if current_focus in focus_values else 0
    synced_label = focus_labels[focus_index]
    if st.session_state.get("inventory_focus_overview") != synced_label:
        st.session_state["inventory_focus_overview"] = synced_label
    focus_label = st.radio(
        "表示対象",
        focus_labels,
        index=focus_index,
        horizontal=True,
        key="inventory_focus_overview",
    )
    state["focus"] = focus_map[focus_label]

    working_sales = sales_df.copy()
    working_inventory = inventory_df.copy()
    if store_choice != transformers.ALL_STORES:
        working_sales = working_sales[working_sales["store"] == store_choice]
        working_inventory = working_inventory[working_inventory["store"] == store_choice]
    if category_choice != transformers.ALL_CATEGORIES:
        working_sales = working_sales[working_sales["category"] == category_choice]
        working_inventory = working_inventory[working_inventory["category"] == category_choice]

    overview_df = inventory.inventory_overview(
        working_sales,
        working_inventory,
        start_date=filters.start_date,
        end_date=filters.end_date,
    )
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
    coverage_series = overview_df.get("coverage_days")
    avg_coverage = float(coverage_series.dropna().mean()) if coverage_series is not None else None
    if coverage_series is not None and coverage_series.dropna().empty:
        avg_coverage = None

    shortage_products = advice_df[advice_df["stock_status"] == "在庫切れ"]["product"].dropna().tolist()
    if stockouts > 0 and shortage_products:
        highlight = shortage_products[0]
        if len(shortage_products) > 1:
            highlight += f" 他{len(shortage_products) - 1}件"
        warning_cols = st.columns([3, 1], gap="medium")
        with warning_cols[0]:
            render_guided_message(
                "inventory_warning",
                message_kwargs={"summary": highlight},
            )
        with warning_cols[1]:
            warning_cols[1].button(
                "発注リストを表示",
                key="inventory-order-button",
                on_click=_focus_stockouts,
                use_container_width=True,
            )
            warning_cols[1].metric("欠品数", f"{stockouts}件")

    _render_kpi_cards(
        [
            {
                "label": "過剰在庫数",
                "value_text": _format_number(safety_excess, "品目"),
                "unit": "品目",
                "yoy": None,
                "target_diff": excess_threshold - safety_excess,
                "alert": safety_excess > excess_threshold,
            },
            {
                "label": "安全在庫欠品数",
                "value_text": _format_number(stockouts, "品目"),
                "unit": "品目",
                "yoy": None,
                "target_diff": stockout_threshold - stockouts,
                "alert": stockouts > stockout_threshold,
            },
            {
                "label": "平均在庫回転率",
                "value_text": (
                    f"{avg_turnover:.1f} 回"
                    + (
                        f"<div class='sub-value'>残日数 {avg_coverage:.1f} 日</div>"
                        if avg_coverage is not None
                        else "<div class='sub-value muted'>残日数 データ不足</div>"
                    )
                ),
                "unit": "回",
                "yoy": None,
                "target_diff": avg_turnover - 8,
            },
        ]
    )

    rolling_window = int(
        overview_df.get("analysis_window", pd.Series([inventory.DEFAULT_ROLLING_WINDOW]))
        .iloc[0]
    )
    buffer_days = int(
        overview_df.get("safety_buffer_days", pd.Series([inventory.DEFAULT_SAFETY_BUFFER_DAYS]))
        .iloc[0]
    )
    timeseries_df = _build_inventory_timeseries(
        working_sales,
        working_inventory,
        rolling_window=rolling_window,
        safety_buffer=buffer_days,
    )
    heatmap_source = advice_df.pivot_table(
        index="store", columns="category", values="estimated_stock", aggfunc="sum"
    ).fillna(0)
    heatmap = None
    if not heatmap_source.empty:
        heatmap = px.imshow(
            heatmap_source,
            color_continuous_scale="Blues",
            labels={"color": "推定在庫（個）"},
            aspect="auto",
        )
        heatmap.update_xaxes(title="カテゴリ")
        heatmap.update_yaxes(title="店舗")
        heatmap.update_layout(coloraxis_colorbar=dict(title="推定在庫（個）"))

    analysis_tabs = st.tabs([
        "在庫推移",
        "カテゴリ別回転率",
        "在庫ヒートマップ",
        "発注リスト",
    ])

    with analysis_tabs[0]:
        st.subheader("在庫推移")
        if timeseries_df.empty:
            st.info("在庫推移を表示できるデータが不足しています。")
        else:
            inventory_trend = go.Figure()
            inventory_trend.add_trace(
                go.Scatter(
                    x=timeseries_df["date"],
                    y=timeseries_df["estimated_stock"],
                    mode="lines",
                    name="推定在庫",
                    line=dict(color=colors["primary"], width=2),
                    fill="tozeroy",
                    hovertemplate="日付: %{x|%Y-%m-%d}<br>推定在庫: %{y:,.0f} 個<extra></extra>",
                )
            )
            if "moving_stock" in timeseries_df.columns:
                inventory_trend.add_trace(
                    go.Scatter(
                        x=timeseries_df["date"],
                        y=timeseries_df["moving_stock"],
                        mode="lines",
                        name=f"在庫移動平均（{rolling_window}日）",
                        line=dict(color=_adjust_hex_color(colors["primary"], -0.15), dash="dot"),
                        hovertemplate="日付: %{x|%Y-%m-%d}<br>移動平均: %{y:,.0f} 個<extra></extra>",
                    )
                )
            inventory_trend.add_trace(
                go.Scatter(
                    x=timeseries_df["date"],
                    y=timeseries_df["safety_stock"],
                    mode="lines",
                    name="安全在庫ライン",
                    line=dict(color=colors["success"], dash="dash"),
                    hovertemplate="日付: %{x|%Y-%m-%d}<br>安全在庫: %{y:,.0f} 個<extra></extra>",
                )
            )
            if "safety_upper" in timeseries_df.columns and "safety_lower" in timeseries_df.columns:
                inventory_trend.add_trace(
                    go.Scatter(
                        x=timeseries_df["date"],
                        y=timeseries_df["safety_upper"],
                        mode="lines",
                        name=f"安全在庫+{buffer_days}日分",
                        line=dict(color=_adjust_hex_color(colors["success"], -0.1), dash="dash"),
                        hovertemplate="日付: %{x|%Y-%m-%d}<br>安全上限: %{y:,.0f} 個<extra></extra>",
                        showlegend=True,
                    )
                )
                inventory_trend.add_trace(
                    go.Scatter(
                        x=timeseries_df["date"],
                        y=timeseries_df["safety_lower"],
                        mode="lines",
                        name=f"安全在庫-{buffer_days}日分",
                        line=dict(color=_adjust_hex_color(colors["success"], 0.2), dash="dash"),
                        hovertemplate="日付: %{x|%Y-%m-%d}<br>安全下限: %{y:,.0f} 個<extra></extra>",
                        fill="tonexty",
                        fillcolor="rgba(63, 178, 126, 0.08)",
                        showlegend=True,
                    )
                )
                safety_floor = timeseries_df["safety_lower"].fillna(timeseries_df["safety_stock"])
            else:
                safety_floor = timeseries_df["safety_stock"]
            shortage_mask = timeseries_df["estimated_stock"] < safety_floor
            if shortage_mask.any():
                inventory_trend.add_trace(
                    go.Scatter(
                        x=timeseries_df.loc[shortage_mask, "date"],
                        y=timeseries_df.loc[shortage_mask, "estimated_stock"],
                        mode="markers",
                        name="欠品リスク",
                        marker=dict(color=colors["error"], size=8),
                        hovertemplate="日付: %{x|%Y-%m-%d}<br>在庫: %{y:,.0f} 個<extra></extra>",
                    )
                )
            inventory_trend.update_layout(
                xaxis=dict(title="日付"),
                yaxis=dict(title="数量（個）", tickformat=",.0f"),
                hovermode="x unified",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                ),
            )
            st.caption(
                f"安全在庫と{buffer_days}日分の上下限を表示しています。赤色マーカーは安全下限を下回った日を示します。"
            )
            st.plotly_chart(inventory_trend, use_container_width=True)

    with analysis_tabs[1]:
        st.subheader("カテゴリ別在庫回転率")
        if turnover_df.empty:
            st.info("在庫回転率を算出できるデータがありません。")
        else:
            turnover_plot_df = turnover_df.sort_values(
                "turnover", ascending=False
            ).reset_index(drop=True)
            turnover_values = (
                turnover_plot_df["turnover"].astype(float).clip(lower=0.0).tolist()
            )
            bar_colors = [colors["primary"] for _ in turnover_values]
            turnover_chart = go.Figure(
                go.Bar(
                    x=turnover_plot_df["category"].astype(str).tolist(),
                    y=turnover_values,
                    marker_color=bar_colors,
                    hovertemplate="カテゴリ: %{x}<br>在庫回転率: %{y:.1f} 回<extra></extra>",
                )
            )
            turnover_chart.update_layout(
                xaxis=dict(title="カテゴリ"),
                yaxis=dict(title="在庫回転率（回）", tickformat=".1f"),
                showlegend=False,
            )
            st.plotly_chart(turnover_chart, use_container_width=True)
            with st.expander("カテゴリ別在庫回転率表", expanded=False):
                st.dataframe(turnover_plot_df, use_container_width=True)

    with analysis_tabs[2]:
        st.subheader("在庫ヒートマップ")
        if heatmap is None:
            st.info("在庫ヒートマップを描画するデータが不足しています。")
        else:
            st.plotly_chart(heatmap, use_container_width=True)

    with analysis_tabs[3]:
        st.subheader("発注リスト・在庫推定")
        focus_map = {"全件": "all", "欠品のみ": "stockout", "過剰のみ": "excess"}
        focus_labels = list(focus_map.keys())
        focus_values = list(focus_map.values())
        current_focus = state.get("focus", "all")
        focus_index = (
            focus_values.index(current_focus)
            if current_focus in focus_values
            else 0
        )
        current_label = focus_labels[focus_index]
        if st.session_state.get("inventory_focus_orders") != current_label:
            st.session_state["inventory_focus_orders"] = current_label
        focus_label = st.radio(
            "表示対象",
            focus_labels,
            index=focus_index,
            horizontal=True,
            key="inventory_focus_orders",
        )
        focus_value = focus_map[focus_label]
        state["focus"] = focus_value
        focused_advice = advice_df.copy()
        if focus_value == "stockout":
            focused_advice = focused_advice[
                focused_advice["stock_status"] == "在庫切れ"
            ]
        elif focus_value == "excess":
            focused_advice = focused_advice[
                focused_advice["stock_status"] == "在庫過多"
            ]
        if focused_advice.empty:
            st.info("該当する在庫データがありません。")
        else:
            st.dataframe(focused_advice, use_container_width=True)


def render_data_management_tab(
    validation_results: Dict[str, data_loader.ValidationResult],
    integration_result: Optional[IntegrationResult],
    baseline: Dict[str, pd.DataFrame],
    sample_files: Dict[str, str],
    templates: Dict[str, bytes],
) -> None:
    st.markdown("### データ管理")

    dataset_labels = {"sales": "売上", "inventory": "仕入/在庫", "fixed_costs": "固定費"}
    current_datasets = st.session_state.get("current_datasets", baseline)
    status_rows = []
    for key, label in dataset_labels.items():
        df = current_datasets.get(key, baseline.get(key, pd.DataFrame()))
        row_count = len(df) if df is not None else 0
        status = "取込済" if row_count else "未取込"
        status_rows.append({"データ種別": label, "件数": row_count, "ステータス": status})
    status_df = pd.DataFrame(status_rows)
    st.subheader("取込状況")
    st.dataframe(status_df, use_container_width=True)

    def _detect_sales_anomalies(df: pd.DataFrame) -> pd.DataFrame:
        columns = ["検出項目", "件数", "備考"]
        if df.empty:
            return pd.DataFrame(columns=columns)
        issues: List[Dict[str, object]] = []
        negative_amount = df[df["sales_amount"] < 0]
        if not negative_amount.empty:
            first = negative_amount.iloc[0]
            issues.append(
                {
                    "検出項目": "売上金額がマイナス",
                    "件数": len(negative_amount),
                    "備考": f"例: {first['product']} ({first['date']:%Y-%m-%d})",
                }
            )
        negative_qty = df[df["sales_qty"] < 0]
        if not negative_qty.empty:
            sample = negative_qty.iloc[0]
            issues.append(
                {
                    "検出項目": "販売数量がマイナス",
                    "件数": len(negative_qty),
                    "備考": f"例: {sample['product']} ({sample['date']:%Y-%m-%d})",
                }
            )
        margin_outliers = df[(df["gross_margin"] < 0) | (df["gross_margin"] > 1)]
        if not margin_outliers.empty:
            issues.append(
                {
                    "検出項目": "粗利率が0〜100%の範囲外",
                    "件数": len(margin_outliers),
                    "備考": "粗利率の算出ロジックを確認してください。",
                }
            )
        return pd.DataFrame(issues, columns=columns)

    def _detect_inventory_anomalies(df: pd.DataFrame) -> pd.DataFrame:
        columns = ["検出項目", "件数", "備考"]
        if df.empty:
            return pd.DataFrame(columns=columns)
        issues: List[Dict[str, object]] = []
        for column in ["opening_stock", "planned_purchase", "safety_stock"]:
            negative = df[df[column] < 0]
            if not negative.empty:
                issues.append(
                    {
                        "検出項目": f"{column} がマイナス",
                        "件数": len(negative),
                        "備考": f"例: {negative.iloc[0]['product']}",
                    }
                )
        zero_safety = df[df["safety_stock"] == 0]
        if not zero_safety.empty:
            issues.append(
                {
                    "検出項目": "安全在庫が0",
                    "件数": len(zero_safety),
                    "備考": "安全在庫基準を設定してください。",
                }
            )
        return pd.DataFrame(issues, columns=columns)

    def _detect_fixed_anomalies(df: pd.DataFrame) -> pd.DataFrame:
        columns = ["検出項目", "件数", "備考"]
        if df.empty:
            return pd.DataFrame(columns=columns)
        issues: List[Dict[str, object]] = []
        numeric_cols = [
            col for col in df.columns if col not in {"store", "month", "date"}
        ]
        for column in numeric_cols:
            negative = df[df[column] < 0]
            if not negative.empty:
                issues.append(
                    {
                        "検出項目": f"{column} がマイナス",
                        "件数": len(negative),
                        "備考": "固定費の入力値をご確認ください。",
                    }
                )
        return pd.DataFrame(issues, columns=columns)

    if status_df["件数"].sum() == 0:
        sample_path = Path(sample_files.get("sales", "")) if sample_files else None
        action: Optional[Dict[str, object]] = None
        if sample_path and sample_path.exists():
            action = {
                "type": "download",
                "data": sample_path.read_bytes(),
                "file_name": sample_path.name,
            }
        render_guided_message("empty", action=action)

    invalid_items = [
        (name, result)
        for name, result in validation_results.items()
        if result is not None and not result.valid
    ]
    if invalid_items:
        dataset_key, first_error = invalid_items[0]
        reason = "フォーマットエラー"
        if first_error.errors is not None and not first_error.errors.empty:
            reason = str(first_error.errors.iloc[0]["内容"])
        template_bytes = templates.get(dataset_key)
        action: Optional[Dict[str, object]] = None
        if template_bytes:
            action = {
                "type": "download",
                "data": template_bytes,
                "file_name": f"{dataset_key}_template.csv",
            }
        render_guided_message(
            "file_format_error",
            message_kwargs={"reason": reason},
            action=action,
        )
    elif validation_results:
        updated_at = st.session_state.get("last_data_update", datetime.now())
        render_guided_message(
            "complete",
            message_kwargs={"timestamp": updated_at.strftime("%Y-%m-%d %H:%M")},
        )
        st.caption("売上タブで最新データをご確認ください。")

    sales_preview = current_datasets.get("sales", baseline.get("sales", pd.DataFrame()))
    inventory_preview = current_datasets.get(
        "inventory", baseline.get("inventory", pd.DataFrame())
    )
    fixed_preview = current_datasets.get(
        "fixed_costs", baseline.get("fixed_costs", pd.DataFrame())
    )

    st.subheader("データクレンジング")
    cleansing_tabs = st.tabs(["売上データ", "仕入／在庫データ", "固定費データ"])

    with cleansing_tabs[0]:
        st.markdown("##### プレビュー")
        st.dataframe(sales_preview.head(20), use_container_width=True)
        sales_issues = _detect_sales_anomalies(sales_preview)
        if sales_issues.empty:
            st.success("売上データに異常値は見つかりませんでした。")
        else:
            st.error("売上データに異常値を検出しました。")
            st.dataframe(sales_issues, use_container_width=True)

    with cleansing_tabs[1]:
        st.markdown("##### プレビュー")
        st.dataframe(inventory_preview.head(20), use_container_width=True)
        inventory_issues = _detect_inventory_anomalies(inventory_preview)
        if inventory_issues.empty:
            st.success("在庫データは正常です。")
        else:
            st.warning("在庫データで確認が必要な項目があります。")
            st.dataframe(inventory_issues, use_container_width=True)

        raw_categories = sorted(
            {
                *(
                    sales_preview.get("category", pd.Series(dtype=str))
                    .dropna()
                    .unique()
                    .tolist()
                ),
                *(
                    inventory_preview.get("category", pd.Series(dtype=str))
                    .dropna()
                    .unique()
                    .tolist()
                ),
            }
        )
        baseline_categories = (
            baseline.get("sales", pd.DataFrame())
            .get("category", pd.Series(dtype=str))
            .dropna()
            .unique()
            .tolist()
        )
        if "category_mapping" not in st.session_state:
            st.session_state["category_mapping"] = {}
        mapping_state = st.session_state["category_mapping"]
        if raw_categories:
            st.markdown("##### カテゴリマッピング")
            options = sorted({*raw_categories, *baseline_categories})
            with st.form("category_mapping_form"):
                st.caption("自社データのカテゴリを分析用の呼称に揃えます。")
                selections: Dict[str, str] = {}
                for category in raw_categories:
                    default_value = mapping_state.get(category, category)
                    default_index = options.index(default_value) if default_value in options else 0
                    selections[category] = st.selectbox(
                        f"{category} を次のカテゴリに統合", options, index=default_index
                    )
                apply_mapping = st.form_submit_button("マッピングを適用", type="primary")
            if apply_mapping:
                mapping_state.update(selections)
                updated_datasets = _copy_datasets(st.session_state["current_datasets"])
                if not updated_datasets["sales"].empty:
                    updated_datasets["sales"]["category"] = (
                        updated_datasets["sales"]["category"].map(mapping_state).fillna(
                            updated_datasets["sales"]["category"]
                        )
                    )
                if not updated_datasets["inventory"].empty:
                    updated_datasets["inventory"]["category"] = (
                        updated_datasets["inventory"]["category"].map(mapping_state).fillna(
                            updated_datasets["inventory"]["category"]
                        )
                    )
                st.session_state["current_datasets"] = updated_datasets
                st.session_state["category_mapping"] = mapping_state
                st.success("カテゴリマッピングを適用しました。分析全体に即時反映されます。")
                trigger_rerun()
            if mapping_state:
                mapping_df = pd.DataFrame(
                    [
                        {"元カテゴリ": src, "適用カテゴリ": dst}
                        for src, dst in sorted(mapping_state.items())
                    ]
                )
                st.dataframe(mapping_df, use_container_width=True)
        else:
            st.caption("カテゴリ列が見つかりません。CSVテンプレートに従って項目を追加してください。")

    with cleansing_tabs[2]:
        st.markdown("##### プレビュー")
        st.dataframe(fixed_preview.head(20), use_container_width=True)
        fixed_issues = _detect_fixed_anomalies(fixed_preview)
        if fixed_issues.empty:
            st.success("固定費データに異常はありません。")
        else:
            st.warning("固定費データの見直しが必要です。")
            st.dataframe(fixed_issues, use_container_width=True)

    st.subheader("CSVアップロード")
    uploaded_files = st.file_uploader(
        "3種類のCSVをまとめてアップロード",
        type="csv",
        accept_multiple_files=True,
        key="data_tab_uploader",
    )
    mapping: Dict[str, object] = {}
    if uploaded_files:
        st.caption("各ファイルのデータ種別を選択してください")
        label_to_key = {v: k for k, v in dataset_labels.items()}
        for idx, file in enumerate(uploaded_files):
            choice = st.selectbox(
                f"{file.name}",
                list(dataset_labels.values()),
                key=f"data_tab_choice_{file.name}_{idx}",
            )
            mapping[label_to_key[choice]] = file
        def _apply_uploaded_files() -> None:
            new_datasets, validations = _handle_csv_uploads(mapping, baseline)
            st.session_state["current_datasets"] = new_datasets
            st.session_state["data_tab_validations"] = validations
            st.session_state["current_source"] = "csv"
            st.session_state["last_data_update"] = datetime.now()
            _update_metadata_from_uploads(mapping, validations)
            _ensure_dataset_metadata(new_datasets, default_source="csv")
            st.success("アップロードを反映しました。")
            trigger_rerun()

        st.button(
            "アップロードを取り込む",
            key="data_tab_apply_upload",
            on_click=_apply_uploaded_files,
        )

    with st.expander("テンプレートをダウンロード", expanded=False):
        for key, content in templates.items():
            label = dataset_labels.get(key, key)
            st.download_button(
                f"{label}テンプレートをダウンロード",
                data=content,
                file_name=f"{key}_template.csv",
                key=f"data-tab-template-{key}",
    )

    import_dashboard.render_dashboard(validation_results, integration_result)


def render_help_settings_page() -> None:
    """Render the help and settings guidance page."""

    st.markdown("### ヘルプ／設定")
    st.caption(
        "ダッシュボードの使い方と設定手順をまとめています。必要に応じてメンバーと共有してください。"
    )

    st.markdown("#### 基本操作")
    st.markdown(
        """
        - 左サイドバーの「ページ切替」からダッシュボード／データ管理／ヘルプページを移動できます。
        - 上部の共通フィルターで期間・店舗・カテゴリを変更すると、売上／粗利／在庫／資金タブすべてに反映されます。
        - 画面上部の「アラートセンター」では、在庫欠品や赤字など重要な通知と対象タブへのショートカットを確認できます。
        - データ管理ページでは、取込状況の確認・異常値チェック・カテゴリマッピングまで一括で実施できます。
        """
    )

    st.markdown("#### データ準備とテンプレート")
    st.markdown(
        """
        - 「データクレンジング」タブでアップロード済みデータのプレビューと異常値検出が行えます。
        - カテゴリマッピングを使うと、店舗ごとに異なる商品カテゴリを分析用の共通カテゴリへ統一できます。
        - CSVテンプレートはサイドバーとデータ管理ページの双方からダウンロード可能です。
        """
    )
    template_info = pd.DataFrame(
        [
            {"データ種別": "売上", "必須列": "date, store, category, product, sales_amount, sales_qty, cogs_amount"},
            {"データ種別": "仕入/在庫", "必須列": "store, product, category, opening_stock, planned_purchase, safety_stock"},
            {"データ種別": "固定費", "必須列": "store, rent, payroll, utilities, marketing, other_fixed"},
        ]
    )
    st.dataframe(template_info, use_container_width=True)

    st.markdown("#### アラート通知の設定")
    st.markdown(
        """
        - サイドバーでアラート表示方法（ページ上部バナー／モーダル）と通知先メール・Slack Webhookを登録できます。
        - しきい値を超えた在庫欠品・過剰在庫・赤字が発生すると、アラートセンターに件数とショートカットが表示されます。
        - メールアドレス／Slack Webhookを設定すると、将来的な外部通知連携の準備が整います。
        """
    )

    st.markdown("#### 推奨環境")
    st.markdown(
        """
        - 推奨ブラウザ: Google Chrome 最新版、Microsoft Edge 最新版。
        - 解像度: 1440×900 以上を推奨（フルHD環境での表示最適化済み）。
        - セキュリティソフトや広告ブロッカーをご利用の場合は、必要に応じて *.streamlit.app ドメインを許可してください。
        """
    )

    st.markdown("#### 操作ガイド動画")
    st.markdown(
        """
        - [ダッシュボードの基本操作デモを見る](https://example.com/matsuya-dashboard-demo)
        - 社内向けトレーニング資料はナレッジベースの「松屋ダッシュボード運用ガイド」を参照してください。
        """
    )


def _render_cash_flow_wizard(
    inputs_state: Dict[str, float],
    *,
    margin_default: float,
    fixed_default: float,
    preset_options: Dict[str, Optional[float]],
    total_sales: float,
    colors: Dict[str, str],
) -> Dict[str, object]:
    """Guide users through cash flow assumptions with a step wizard."""

    steps = [
        {"title": "粗利率の確認", "description": "直近実績から推奨範囲を提案します。"},
        {"title": "固定費の確認", "description": "固定費を期間合計（円）で入力します。"},
        {"title": "目標利益の設定", "description": "テンプレートまたはカスタム値を選択します。"},
    ]
    wizard_state = st.session_state.setdefault(
        "cash_flow_wizard", {"step": 0, "completed": False}
    )
    step_index = int(wizard_state.get("step", 0))
    step_index = max(0, min(step_index, len(steps) - 1))
    wizard_state["step"] = step_index

    st.markdown("#### 入力ウィザード")
    st.caption("粗利率 → 固定費 → 目標利益の順に入力し、各ステップでヒントを参照してください。")
    progress_value = int(((step_index + 1) / len(steps)) * 100)
    st.progress(progress_value)

    pill_items: List[str] = []
    active_color = colors["primary"]
    inactive_color = _adjust_hex_color(colors["neutral"], -0.1)
    for idx, meta in enumerate(steps):
        is_active = idx == step_index
        background = active_color if is_active else inactive_color
        text_color = "#FFFFFF" if is_active else colors["text"]
        pill_items.append(
            f"""
            <span title="{escape(meta['description'])}" style="display:inline-flex;align-items:center;gap:0.35rem;padding:0.35rem 0.9rem;border-radius:999px;font-size:0.85rem;font-weight:600;background:{background};color:{text_color};">
                {idx + 1}. {escape(meta['title'])}
            </span>
            """
        )
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.8rem;'>{''.join(pill_items)}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"ステップ {step_index + 1} / {len(steps)}：{steps[step_index]['title']}")

    alerts: List[Tuple[str, str]] = []
    blocking_error = False

    if step_index == 0:
        margin_choice = st.slider(
            "粗利率を設定",  # UI label
            min_value=0.1,
            max_value=0.8,
            value=float(inputs_state.get("gross_margin", margin_default)),
            step=0.01,
            format="%0.2f",
            help="粗利率の推奨レンジ（20%〜70%）を目安に設定してください。",
        )
        inputs_state["gross_margin"] = float(margin_choice)
        st.caption(f"参考値：直近平均 {margin_default:.0%}")
        if margin_choice < 0.2:
            alerts.append(
                (
                    "warning",
                    "粗利率が20%未満です。原価や値引き施策を再確認してください。",
                )
            )
        if margin_choice > 0.7:
            alerts.append(
                (
                    "warning",
                    "粗利率が70%を超えています。入力ミスがないか確認してください。",
                )
            )
    elif step_index == 1:
        fixed_choice = st.number_input(
            "固定費（円）",
            min_value=0.0,
            value=float(inputs_state.get("fixed_cost", fixed_default)),
            step=100000.0,
            format="%0.0f",
            help="家賃・人件費などの固定費合計を円単位で入力してください。",
        )
        inputs_state["fixed_cost"] = float(fixed_choice)
        st.caption(
            "※ 金額は円単位・期間合計で入力します。例：5,000,000円"
        )
        if fixed_choice <= 0:
            alerts.append(("error", "固定費は0円より大きい値を入力してください。"))
        if total_sales and fixed_choice > total_sales * 1.2:
            alerts.append(
                (
                    "warning",
                    "固定費が期間売上を大幅に上回っています。想定期間を再確認してください。",
                )
            )
    else:
        preset_keys = list(preset_options.keys())
        preset_choice = st.selectbox(
            "目標利益テンプレート",
            preset_keys,
            index=(
                preset_keys.index(inputs_state.get("preset", preset_keys[0]))
                if inputs_state.get("preset") in preset_keys
                else 0
            ),
            key="cash_flow_wizard_preset",
            help="テンプレートを選ぶと推奨目標利益が自動で入力されます。",
        )
        inputs_state["preset"] = preset_choice
        preset_target_value = preset_options[preset_choice]
        target_default = (
            float(preset_target_value)
            if preset_target_value is not None
            else float(inputs_state.get("target_profit", 5_000_000.0))
        )
        target_choice = st.number_input(
            "目標利益（円）",
            min_value=0.0,
            value=target_default,
            step=50000.0,
            format="%0.0f",
            disabled=preset_target_value is not None,
            help="審査資料などに利用する年間（または対象期間）目標利益を入力してください。",
            key="cash_flow_wizard_target",
        )
        if preset_target_value is not None:
            inputs_state["target_profit"] = float(preset_target_value)
        else:
            inputs_state["target_profit"] = float(target_choice)
        st.caption("※ 金額は円単位です。カスタムを選択すると直接入力できます。")
        target_profit = float(inputs_state.get("target_profit", 0.0))
        fixed_cost = float(inputs_state.get("fixed_cost", fixed_default))
        if fixed_cost and target_profit < fixed_cost * 0.05:
            alerts.append(
                (
                    "warning",
                    "目標利益が固定費に対して低めです。達成基準を再確認してください。",
                )
            )
        if total_sales and target_profit > total_sales * 0.5:
            alerts.append(
                (
                    "warning",
                    "目標利益が期間売上の50%を超えています。前提値を確認してください。",
                )
            )

    for level, message in alerts:
        if level == "error":
            st.error(message)
        else:
            st.warning(message)
    blocking_error = any(level == "error" for level, _ in alerts)

    current_signature = (
        float(inputs_state.get("gross_margin", margin_default)),
        float(inputs_state.get("fixed_cost", fixed_default)),
        float(inputs_state.get("target_profit", 0.0)),
        str(inputs_state.get("preset", "")),
    )
    if wizard_state.get("completed") and wizard_state.get("last_signature") != current_signature:
        wizard_state["completed"] = False

    nav_cols = st.columns([1, 1, 2])
    prev_clicked = nav_cols[0].button(
        "前へ",
        disabled=step_index == 0,
        key=f"cash_flow_wizard_prev_{step_index}",
    )
    if prev_clicked:
        wizard_state["step"] = max(0, step_index - 1)
        wizard_state["completed"] = False
        trigger_rerun()

    next_label = "入力を確定" if step_index == len(steps) - 1 else "次へ"
    next_clicked = nav_cols[1].button(
        next_label,
        key=f"cash_flow_wizard_next_{step_index}",
        type="primary" if step_index == len(steps) - 1 else "secondary",
        disabled=blocking_error,
    )
    if next_clicked and not blocking_error:
        if step_index < len(steps) - 1:
            wizard_state["step"] = step_index + 1
        else:
            wizard_state["completed"] = True
            wizard_state["last_signature"] = current_signature
        trigger_rerun()

    helper_message = (
        "入力を進めると設定値が下部のシミュレーションカードに反映されます。"
    )
    nav_cols[2].markdown(
        f"<div style='font-size:0.8rem;color:{colors['text']};opacity:0.7;'>{escape(helper_message)}</div>",
        unsafe_allow_html=True,
    )

    if wizard_state.get("completed") and not blocking_error:
        st.success("ウィザードの入力内容を保存しました。シミュレーションが最新の値で更新されています。")

    return {"completed": bool(wizard_state.get("completed")), "alerts": alerts}


def render_cash_tab(
    sales_df: pd.DataFrame,
    comparison_sales: pd.DataFrame,
    inventory_df: pd.DataFrame,
    pnl_df: pd.DataFrame,
    filters: transformers.FilterState,
) -> None:
    colors = _resolve_theme_colors()
    st.subheader("資金繰りダッシュボード")

    cash_current = _cash_flow_summary(sales_df, inventory_df)
    cash_previous = _cash_flow_summary(comparison_sales, inventory_df)
    total_sales = float(sales_df["sales_amount"].sum())
    cash_target = total_sales * TARGET_CASH_RATIO if total_sales else cash_current["balance"]

    _render_kpi_cards(
        [
            {
                "label": "資金残高",
                "value_text": _format_currency(cash_current["balance"]),
                "unit": "円",
                "yoy": _compute_growth(cash_current["balance"], cash_previous.get("balance")),
                "target_diff": cash_current["balance"] - cash_target,
            },
            {
                "label": "入金予定",
                "value_text": _format_currency(cash_current["receivable"]),
                "unit": "円",
                "yoy": _compute_growth(cash_current["receivable"], cash_previous.get("receivable")),
                "target_diff": cash_current["receivable"],
            },
            {
                "label": "支払予定",
                "value_text": _format_currency(cash_current["payable"]),
                "unit": "円",
                "yoy": _compute_growth(cash_current["payable"], cash_previous.get("payable")),
                "target_diff": -cash_current["payable"],
            },
        ]
    )

    granularity = filters.period_granularity or "monthly"

    def _cash_timeseries(dataset: pd.DataFrame) -> pd.DataFrame:
        if dataset.empty:
            return pd.DataFrame(
                columns=[
                    "period_start",
                    "period_label",
                    "balance",
                    "deposit",
                    "receivable",
                    "payable",
                    "scope",
                ]
            )
        aggregated = sales.aggregate_timeseries(dataset, granularity)
        records = []
        for _, row in aggregated.iterrows():
            start, end = sales.period_bounds(row["period_key"], granularity)
            mask = (dataset["date"] >= start) & (dataset["date"] <= end)
            summary = _cash_flow_summary(dataset.loc[mask], inventory_df)
            records.append(
                {
                    "period_start": row["period_start"],
                    "period_label": row["period_label"],
                    "balance": summary["balance"],
                    "deposit": summary["deposit"],
                    "receivable": summary["receivable"],
                    "payable": summary["payable"],
                    "scope": "actual",
                }
            )
        return pd.DataFrame(records)

    def _forecast_cash_flow(history: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
        if history.empty or periods <= 0:
            return pd.DataFrame(
                columns=[
                    "period_start",
                    "period_label",
                    "balance",
                    "deposit",
                    "receivable",
                    "payable",
                    "scope",
                ]
            )

        actual_history = history.sort_values("period_start")
        value_columns = ["balance", "deposit", "receivable", "payable"]
        delta_map: Dict[str, float] = {}
        for column in value_columns:
            if column not in actual_history:
                delta_map[column] = 0.0
                continue
            deltas = actual_history[column].diff().dropna()
            delta = float(deltas.mean()) if not deltas.empty else 0.0
            if pd.isna(delta):
                delta = 0.0
            delta_map[column] = delta

        offsets = {
            "daily": pd.DateOffset(days=1),
            "weekly": pd.DateOffset(weeks=1),
            "monthly": pd.DateOffset(months=1),
            "yearly": pd.DateOffset(years=1),
        }
        offset = offsets.get(granularity, pd.DateOffset(months=1))

        last_row = actual_history.iloc[-1]
        next_start = last_row["period_start"] + offset
        label_format = sales.GRANULARITY_CONFIG.get(granularity, {}).get(
            "format", "%Y-%m-%d"
        )
        values = {column: float(last_row[column]) for column in value_columns}

        records = []
        for _ in range(periods):
            for column, delta in delta_map.items():
                values[column] = float(values[column] + delta)
            records.append(
                {
                    "period_start": next_start,
                    "period_label": next_start.strftime(label_format),
                    "balance": values["balance"],
                    "deposit": values["deposit"],
                    "receivable": values["receivable"],
                    "payable": values["payable"],
                    "scope": "forecast",
                }
            )
            next_start = next_start + offset

        return pd.DataFrame(records)

    cash_trend_df = _cash_timeseries(sales_df)
    comparison_trend_df = _cash_timeseries(comparison_sales)

    tick_format_map = {
        "daily": "%Y-%m-%d",
        "weekly": "%Y-%m-%d",
        "monthly": "%Y-%m",
        "yearly": "%Y",
    }
    tick_format = tick_format_map.get(granularity, "%Y-%m")

    st.subheader("現預金推移")
    if not cash_trend_df.empty:
        cash_chart = go.Figure()
        cash_chart.add_trace(
            go.Scatter(
                x=cash_trend_df["period_start"],
                y=cash_trend_df["balance"],
                mode="lines+markers",
                name="資金残高",
                line=dict(color=colors["primary"], width=2),
                text=cash_trend_df["period_label"],
                hovertemplate="期間: %{text}<br>資金残高: %{y:,.0f} 円<extra></extra>",
            )
        )
        if not comparison_trend_df.empty:
            cash_chart.add_trace(
                go.Scatter(
                    x=comparison_trend_df["period_start"],
                    y=comparison_trend_df["balance"],
                    mode="lines",
                    name="前年資金残高",
                    line=dict(color=_adjust_hex_color(colors["primary"], 0.3), dash="dash"),
                    text=comparison_trend_df["period_label"],
                    hovertemplate="期間: %{text}<br>前年資金残高: %{y:,.0f} 円<extra></extra>",
                )
            )
        cash_chart.update_layout(
            xaxis=dict(title="期間", type="date", tickformat=tick_format),
            yaxis=dict(title="金額（円）", tickformat=",.0f"),
            hovermode="x unified",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
        )
        st.plotly_chart(cash_chart, use_container_width=True)
    else:
        st.info("現預金推移を表示できるデータがありません。")

    if not cash_trend_df.empty:
        forecast_df = _forecast_cash_flow(cash_trend_df)
        projection_df = pd.concat(
            [cash_trend_df, forecast_df], ignore_index=True, sort=False
        )
        projection_df = projection_df.sort_values("period_start")
        st.subheader("資金繰り予測（入出金構成）")
        area_source = projection_df.melt(
            id_vars=["period_start", "period_label", "scope"],
            value_vars=["deposit", "receivable", "payable"],
            var_name="component",
            value_name="amount",
        )
        if area_source.empty:
            st.info("資金繰り予測を算出するためのデータが不足しています。")
        else:
            area_source = area_source.copy()
            area_source["display_amount"] = area_source["amount"]
            payable_mask = area_source["component"] == "payable"
            area_source.loc[payable_mask, "display_amount"] *= -1
            component_labels = {
                "deposit": "預金",
                "receivable": "売掛金",
                "payable": "買掛金",
            }
            component_colors = {
                "deposit": colors["primary"],
                "receivable": _adjust_hex_color(colors["primary"], 0.25),
                "payable": colors["error"],
            }
            scope_labels = {"actual": "実績", "forecast": "予測"}
            area_fig = go.Figure()
            for component, label in component_labels.items():
                comp_data = area_source[area_source["component"] == component].sort_values(
                    "period_start"
                )
                if comp_data.empty:
                    continue
                area_fig.add_trace(
                    go.Scatter(
                        x=comp_data["period_start"],
                        y=comp_data["display_amount"],
                        name=label,
                        stackgroup="cash_projection",
                        mode="lines",
                        line=dict(color=component_colors[component], width=2),
                        text=comp_data["period_label"],
                        customdata=comp_data["scope"].map(scope_labels).fillna("実績"),
                        hovertemplate=f"期間: %{{text}}<br>{label}: %{{y:,.0f}} 円<br>%{{customdata}}<extra></extra>",
                    )
                )
            area_fig.update_layout(
                xaxis=dict(title="期間", type="date", tickformat=tick_format),
                yaxis=dict(title="金額（円）", tickformat=",.0f"),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                ),
                hovermode="x unified",
            )
            if not forecast_df.empty:
                forecast_start = pd.to_datetime(forecast_df["period_start"].min())
                forecast_end = pd.to_datetime(forecast_df["period_start"].max())
                area_fig.add_vrect(
                    x0=forecast_start,
                    x1=forecast_end,
                    fillcolor=_adjust_hex_color(colors["neutral"], 0.2),
                    opacity=0.2,
                    line_width=0,
                    layer="below",
                )
                midpoint = (
                    forecast_start + (forecast_end - forecast_start) / 2
                    if forecast_end > forecast_start
                    else forecast_start
                )
                area_fig.add_annotation(
                    x=midpoint,
                    y=0,
                    yref="paper",
                    text="予測期間",
                    showarrow=False,
                    font=dict(color=colors["text"]),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                )
            st.plotly_chart(area_fig, use_container_width=True)
            st.caption("正の領域は資金の流入（預金・売掛金）、負の領域は支払予定を表します。網掛け部分は推計期間です。")

    st.subheader("資金繰りシミュレーション")
    total_gross_profit = float(pnl_df["gross_profit"].sum())
    default_margin = total_gross_profit / total_sales if total_sales else 0.45
    default_margin = float(min(max(default_margin, 0.1), 0.8))
    default_fixed_cost = float(pnl_df["total_fixed_cost"].sum())

    if "cash_flow_inputs" not in st.session_state:
        st.session_state["cash_flow_inputs"] = {}
    inputs_state = st.session_state["cash_flow_inputs"]
    previous_margin_default = inputs_state.get("gross_margin_default")
    margin_default = float(round(default_margin, 2))
    if "gross_margin" not in inputs_state:
        inputs_state["gross_margin"] = margin_default
    elif (
        previous_margin_default is not None
        and abs(inputs_state.get("gross_margin", margin_default) - previous_margin_default) < 1e-6
    ):
        inputs_state["gross_margin"] = margin_default
    inputs_state["gross_margin_default"] = margin_default

    previous_fixed_default = inputs_state.get("fixed_cost_default")
    fixed_default = float(default_fixed_cost)
    if "fixed_cost" not in inputs_state:
        inputs_state["fixed_cost"] = fixed_default
    elif (
        previous_fixed_default is not None
        and abs(inputs_state.get("fixed_cost", fixed_default) - previous_fixed_default) < 1.0
    ):
        inputs_state["fixed_cost"] = fixed_default
    inputs_state["fixed_cost_default"] = fixed_default

    if "target_profit" not in inputs_state:
        inputs_state["target_profit"] = 5_000_000.0
    if "preset" not in inputs_state:
        inputs_state["preset"] = "500万円"

    preset_options = {"500万円": 5_000_000.0, "1,000万円": 10_000_000.0, "カスタム": None}

    wizard_result = _render_cash_flow_wizard(
        inputs_state,
        margin_default=margin_default,
        fixed_default=fixed_default,
        preset_options=preset_options,
        total_sales=total_sales,
        colors=colors,
    )

    wizard_completed = bool(wizard_result.get("completed"))
    if not wizard_completed and wizard_result.get("alerts"):
        st.caption("※ ウィザード内の警告を解消すると次のステップに進めます。")
    elif not wizard_completed:
        st.caption("※ ウィザードを最後まで完了すると入力内容が確定します。")

    if inputs_state.get("preset") != "カスタム":
        preset_target = preset_options.get(inputs_state.get("preset"))
        if preset_target is not None:
            inputs_state["target_profit"] = float(preset_target)

    gross_margin = float(inputs_state.get("gross_margin", margin_default))
    fixed_cost = float(inputs_state.get("fixed_cost", fixed_default))
    target_profit = float(inputs_state.get("target_profit", 5_000_000.0))

    summary_html = f"""
    <div style="border:1px solid rgba(148,163,184,0.35);border-radius:0.9rem;padding:0.9rem 1.2rem;background-color:rgba(255,255,255,0.85);display:flex;flex-wrap:wrap;gap:1.5rem;margin-top:0.8rem;margin-bottom:0.6rem;">
        <div>
            <div style="font-size:0.75rem;color:#64748b;">粗利率</div>
            <div style="font-size:1.6rem;font-weight:600;color:{colors['text']};">{gross_margin*100:.1f}%</div>
        </div>
        <div>
            <div style="font-size:0.75rem;color:#64748b;">固定費</div>
            <div style="font-size:1.6rem;font-weight:600;color:{colors['text']};">{fixed_cost:,.0f}<span style="font-size:0.9rem;"> 円</span></div>
        </div>
        <div>
            <div style="font-size:0.75rem;color:#64748b;">目標利益</div>
            <div style="font-size:1.6rem;font-weight:600;color:{colors['text']};">{target_profit:,.0f}<span style="font-size:0.9rem;"> 円</span></div>
        </div>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)
    st.caption("粗利率は割合（%）、固定費・目標利益は円単位の入力値です。")

    validation_messages: List[Tuple[str, str]] = []
    if total_sales and fixed_cost > total_sales:
        validation_messages.append(
            (
                "warning",
                f"固定費（{fixed_cost:,.0f}円）が期間売上（{total_sales:,.0f}円）を上回っています。期間設定や金額を再確認してください。",
            )
        )
    if gross_margin < 0.2:
        validation_messages.append(
            (
                "warning",
                "粗利率が20%未満です。商品構成や原価率の調整を検討してください。",
            )
        )
    if target_profit > 0 and target_profit < fixed_cost * 0.05:
        validation_messages.append(
            (
                "warning",
                "目標利益が固定費に対して小さいため、達成しても収益改善効果が限定的です。",
            )
        )
    if total_sales and target_profit > total_sales * 0.6:
        validation_messages.append(
            (
                "warning",
                "目標利益が期間売上の60%を超えています。現実的な目標か確認してください。",
            )
        )

    for level, message in validation_messages:
        if level == "error":
            st.error(message)
        else:
            st.warning(message)

    inputs = simulation.SimulationInputs(
        gross_margin=gross_margin,
        fixed_cost=fixed_cost,
        target_profit=target_profit,
    )
    requirements = simulation.required_sales(inputs)
    breakeven_sales_value = float(requirements["breakeven"])
    target_sales_value = float(requirements["target_sales"])
    current_sales_value = total_sales
    progress_ratio = (
        current_sales_value / target_sales_value if target_sales_value > 0 else 0.0
    )
    reached_target = target_sales_value > 0 and current_sales_value >= target_sales_value
    gauge_base = max(target_sales_value, current_sales_value)
    gauge_max = gauge_base * 1.1 if gauge_base else 1.0

    curve = simulation.breakeven_sales_curve(simulation.DEFAULT_MARGIN_RANGE, fixed_cost)
    curve_chart = px.line(
        curve,
        x="gross_margin",
        y="breakeven_sales",
        labels={"gross_margin": "粗利率", "breakeven_sales": "損益分岐点売上高（円）"},
    )
    curve_chart.update_traces(
        hovertemplate="粗利率: %{x:.0%}<br>損益分岐点売上高: %{y:,.0f} 円<extra></extra>",
    )
    curve_chart.update_layout(
        xaxis=dict(title="粗利率", tickformat=".0%"),
        yaxis=dict(title="損益分岐点売上高（円）", tickformat=",.0f"),
        showlegend=False,
    )
    if gross_margin is not None:
        curve_chart.add_vline(
            x=gross_margin,
            line_dash="dash",
            line_color=colors["error"],
            annotation_text="目標粗利率",
            annotation_position="top right",
            annotation=dict(font=dict(color=colors["error"])),
        )
    st.plotly_chart(curve_chart, use_container_width=True)

    results_col, saved_col = st.columns([3, 2])
    with results_col:
        progress_display = f"{progress_ratio:.1%}" if target_sales_value > 0 else "ー"
        summary_card_html = f"""
        <div style="border:1px solid #dee2e6;border-radius:0.75rem;padding:1.5rem;background-color:#ffffff;margin-bottom:1rem;">
            <div style="font-size:0.9rem;color:#6c757d;">損益分岐点売上</div>
            <div style="font-size:2.6rem;font-weight:600;line-height:1.1;">{breakeven_sales_value:,.0f}<span style="font-size:1.2rem;"> 円</span></div>
            <div style="margin-top:1.2rem;display:flex;flex-wrap:wrap;gap:1.5rem;">
                <div>
                    <div style="font-size:0.85rem;color:#6c757d;">目標利益達成に必要な売上</div>
                    <div style="font-size:1.2rem;font-weight:500;">{target_sales_value:,.0f} 円</div>
                </div>
                <div>
                    <div style="font-size:0.85rem;color:#6c757d;">現状売上</div>
                    <div style="font-size:1.2rem;font-weight:500;">{current_sales_value:,.0f} 円</div>
                </div>
                <div>
                    <div style="font-size:0.85rem;color:#6c757d;">目標達成率</div>
                    <div style="font-size:1.2rem;font-weight:500;">{progress_display}</div>
                </div>
            </div>
        </div>
        """
        st.markdown(summary_card_html, unsafe_allow_html=True)

        success_color = colors["success"]
        error_color = colors["error"]
        success_bg = _adjust_hex_color(success_color, 0.75)
        error_bg = _adjust_hex_color(error_color, 0.75)
        neutral_bg = _adjust_hex_color(colors["neutral"], 0.6)

        gauge_config: Dict[str, object] = {
            "axis": {"range": [0, gauge_max], "tickformat": ",.0f"},
            "bar": {"color": success_color if reached_target else error_color},
            "bgcolor": "#FFFFFF",
        }
        gauge_steps = []
        if target_sales_value > 0:
            gauge_steps.append(
                {
                    "range": [0, min(target_sales_value, gauge_max)],
                    "color": success_bg if reached_target else error_bg,
                }
            )
            if target_sales_value < gauge_max:
                gauge_steps.append(
                    {"range": [target_sales_value, gauge_max], "color": neutral_bg}
                )
            gauge_config["steps"] = gauge_steps
            gauge_config["threshold"] = {
                "line": {"color": success_color if reached_target else error_color, "width": 4},
                "value": target_sales_value,
            }

        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=current_sales_value,
                number={
                    "valueformat": ",.0f",
                    "suffix": " 円",
                    "font": {"size": 28},
                },
                title={"text": "目標売上に対する現状売上（円）", "font": {"size": 16}},
                gauge=gauge_config,
            )
        )
        gauge_fig.update_layout(margin=dict(t=40, b=10, l=30, r=30), height=320)
        st.plotly_chart(gauge_fig, use_container_width=True)

        if target_sales_value > 0:
            status_text = "達成" if reached_target else "未達"
            st.caption(
                f"バーは現状売上、ラインは目標売上を示します（単位: 円）。背景色は目標{status_text}の状態を表します。"
            )
        else:
            st.caption(
                "バーは現状売上を示します（単位: 円）。目標売上が未設定の場合はラインは表示されません。"
            )

        st.markdown("#### シナリオ詳細")

        if "simulation_scenario_name" not in st.session_state:
            st.session_state["simulation_scenario_name"] = (
                f"{filters.store}_{datetime.now():%Y%m%d_%H%M}"
            )
        scenario_name_default = st.session_state["simulation_scenario_name"]
        scenario_name = st.text_input(
            "シナリオ名",
            key="_simulation_scenario_name_widget",
            value=scenario_name_default,
        )
        st.session_state["simulation_scenario_name"] = scenario_name

        if "saved_scenarios" not in st.session_state:
            st.session_state["saved_scenarios"] = []

        def _save_scenario() -> None:
            scenario_title = st.session_state.get(
                "simulation_scenario_name", scenario_name
            )
            scenario = {
                "name": scenario_title,
                "gross_margin": gross_margin,
                "fixed_cost": fixed_cost,
                "target_profit": target_profit,
                "breakeven": requirements["breakeven"],
                "target_sales": requirements["target_sales"],
                "preset": inputs_state.get("preset", "カスタム"),
                "saved_at": datetime.now().isoformat(),
            }
            st.session_state["saved_scenarios"].append(scenario)
            new_default = f"{filters.store}_{datetime.now():%Y%m%d_%H%M}"
            st.session_state["simulation_scenario_name"] = new_default
            st.session_state["_simulation_scenario_name_widget"] = new_default
            message_config = MESSAGE_DICTIONARY.get("simulation_saved", {})
            toast_message = message_config.get(
                "message", "シミュレーション結果を保存しました。"
            )
            st.toast(toast_message)
            trigger_rerun()

        st.button(
            "シナリオ保存",
            key="save_simulation_scenario",
            on_click=_save_scenario,
        )

    if "saved_scenarios" not in st.session_state:
        st.session_state["saved_scenarios"] = []
    saved_scenarios = st.session_state["saved_scenarios"]
    with saved_col:
        if saved_scenarios:
            selected_index = st.selectbox(
                "シナリオを読み込む",
                options=list(range(len(saved_scenarios))),
                format_func=lambda idx: saved_scenarios[idx]["name"],
                key="simulation_selected_scenario",
            )
            selected = saved_scenarios[selected_index]
            def _apply_scenario() -> None:
                if "cash_flow_inputs" not in st.session_state:
                    st.session_state["cash_flow_inputs"] = {}
                scenario_inputs = st.session_state["cash_flow_inputs"]
                scenario_inputs.update(
                    {
                        "gross_margin": float(selected["gross_margin"]),
                        "fixed_cost": float(selected["fixed_cost"]),
                        "target_profit": float(selected["target_profit"]),
                        "preset": "カスタム",
                    }
                )
                trigger_rerun()

            st.button(
                "選択したシナリオを適用",
                key="apply_simulation_scenario",
                on_click=_apply_scenario,
            )

            scenarios_df = pd.DataFrame(saved_scenarios)
            st.dataframe(scenarios_df, use_container_width=True)
            csv_bytes = scenarios_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "シナリオ一覧をダウンロード",
                data=csv_bytes,
                file_name="matsuya_simulation_scenarios.csv",
            )
        else:
            st.info("保存されたシナリオはまだありません。")

def main() -> None:
    st.title("松屋 計数管理ダッシュボード")
    _inject_global_styles()
    if MAIN_TAB_KEY not in st.session_state:
        st.session_state[MAIN_TAB_KEY] = MAIN_TAB_LABELS[0]
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = PAGE_OPTIONS[0]

    sample_files = data_loader.available_sample_files()
    templates = data_loader.available_templates()
    sample_files_for_ui = {k: str(v) for k, v in sample_files.items()}
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
    current_source = st.session_state.get("current_source", "sample")
    dataset_status = _ensure_dataset_metadata(
        active_datasets, default_source=current_source
    )

    sidebar_state = sidebar.render_sidebar(
        stores,
        categories,
        default_period=default_period,
        regions=regions,
        channels=channels,
        sample_files=sample_files_for_ui,
        templates=templates,
        providers=provider_options,
        show_filters=False,
        dataset_status=dataset_status,
    )

    if sidebar_state.get("show_data_spec_modal"):
        with st.modal("データ仕様", key="data_spec_modal"):
            st.markdown("### データ取り込みの前提")
            st.write("API連携: 現在開発中です。CSVアップロードをご利用ください。")
            st.markdown("#### サンプルCSV")
            for dataset, path in sample_files.items():
                file_path = Path(path)
                if not file_path.exists():
                    continue
                with file_path.open("rb") as handle:
                    data = handle.read()
                label = sidebar.DATASET_LABELS.get(dataset, dataset)
                st.download_button(
                    f"{label}サンプルCSVをダウンロード",
                    data=data,
                    file_name=file_path.name,
                    key=f"data-spec-sample-{dataset}",
                )
            st.markdown("#### 必須カラム")
            spec_columns = sidebar_state.get("data_spec_columns", {})
            for dataset, columns in spec_columns.items():
                label = sidebar.DATASET_LABELS.get(dataset, dataset)
                st.markdown(f"**{label}**")
                for column, description in columns:
                    st.markdown(f"- `{column}`: {description}")
            st.markdown("#### CSVテンプレート")
            for dataset, content in templates.items():
                label = sidebar.DATASET_LABELS.get(dataset, dataset)
                st.download_button(
                    f"{label}テンプレートをダウンロード",
                    data=content,
                    file_name=f"{dataset}_template.csv",
                    key=f"data-spec-template-{dataset}",
                )
            def _close_data_spec_modal() -> None:
                _set_state_and_rerun("show_data_spec_modal", False)

            st.button(
                "閉じる",
                key="close_data_spec_modal",
                on_click=_close_data_spec_modal,
            )

    mode = sidebar_state["data_source_mode"]
    validation_results: Dict[str, data_loader.ValidationResult] = {}
    integration_result: Optional[IntegrationResult] = None

    if mode == "csv":
        datasets, validation_results = _handle_csv_uploads(
            sidebar_state["uploads"],
            baseline,
        )
        _update_metadata_from_uploads(sidebar_state["uploads"], validation_results)
    elif mode == "api":
        datasets, integration_result = _handle_api_mode(
            sidebar_state["api"],
            baseline,
        )
        _update_metadata_from_integration(integration_result)
    else:
        datasets = active_datasets

    extra_validations = st.session_state.get("data_tab_validations")
    if extra_validations:
        validation_results.update(extra_validations)

    if not datasets:
        datasets = active_datasets

    for key, default_df in baseline.items():
        datasets.setdefault(key, default_df.copy())

    datasets["sales"] = transformers.prepare_sales_dataset(datasets["sales"])
    
    st.session_state["current_datasets"] = _copy_datasets(datasets)

    if mode == "csv":
        uploaded_keys = {
            key
            for key, upload in sidebar_state["uploads"].items()
            if upload is not None
        }
        if uploaded_keys and any(
            validation_results.get(key) and validation_results[key].valid
            for key in uploaded_keys
        ):
            st.session_state["current_source"] = "csv"
    elif mode == "api" and integration_result is not None:
        st.session_state["current_source"] = "api"
    else:
        if "current_source" not in st.session_state:
            st.session_state["current_source"] = current_source

    _ensure_dataset_metadata(
        st.session_state["current_datasets"],
        default_source=st.session_state.get("current_source", "sample"),
    )

    stores = transformers.extract_stores(datasets["sales"])
    categories = transformers.extract_categories(datasets["sales"])
    regions = transformers.extract_regions(datasets["sales"])
    channels = transformers.extract_channels(datasets["sales"])
    default_period = _default_period(datasets["sales"])
    bounds = _dataset_bounds(datasets["sales"])

    alerts = _collect_alerts(
        datasets,
        sidebar_state.get("alert_settings"),
        default_period,
    )
    st.session_state["latest_alerts"] = alerts
    render_alert_center(alerts, sidebar_state.get("alert_settings"))

    st.sidebar.header("ページ切替")
    current_page = st.session_state.get("active_page", PAGE_OPTIONS[0])
    page_choice = st.sidebar.radio(
        "表示ページ",
        PAGE_OPTIONS,
        index=PAGE_OPTIONS.index(current_page)
        if current_page in PAGE_OPTIONS
        else 0,
        key="page_selector",
    )
    st.session_state["active_page"] = page_choice

    if page_choice == "ダッシュボード":
        active_tab = st.session_state.get(MAIN_TAB_KEY, MAIN_TAB_LABELS[0])
        if active_tab not in MAIN_TAB_LABELS:
            active_tab = MAIN_TAB_LABELS[0]

        header_container = st.container()
        with header_container:
            login_col, control_col = st.columns([1, 3])
            login_name = st.session_state.get("login_user", "経営者A")
            login_col.markdown(f"#### 👤 {login_name}")
            with control_col:
                global_filters = render_global_filter_bar(
                    stores,
                    categories,
                    default_period=default_period,
                    bounds=bounds,
                )

        filtered_sales = transformers.apply_filters(
            datasets["sales"], global_filters
        )
        dashboard_comparison = _comparison_dataset(
            datasets["sales"], global_filters, "previous_period"
        )

        pnl_baseline = profitability.store_profitability(
            filtered_sales,
            datasets["fixed_costs"],
        )

        pnl_df = render_dashboard_tab(
            filtered_sales,
            dashboard_comparison,
            global_filters,
            datasets["fixed_costs"],
            datasets["inventory"],
        )
        st.session_state["latest_pnl_df"] = pnl_df

        active_tab = _render_analysis_navigation(active_tab)
        st.session_state[MAIN_TAB_KEY] = active_tab

        if active_tab == "売上":
            render_sales_tab(
                datasets["sales"],
                global_filters,
                channels,
                comparison_mode="yoy",
            )
            st.divider()
            abc_df, _ = render_products_tab(
                filtered_sales, dashboard_comparison, global_filters
            )
            st.session_state["latest_abc_df"] = abc_df
        elif active_tab == "粗利":
            pnl_view = render_profitability_tab(
                filtered_sales,
                dashboard_comparison,
                datasets["fixed_costs"],
                global_filters,
            )
            st.session_state["latest_pnl_df"] = pnl_view
        elif active_tab == "在庫":
            abc_df_cached = st.session_state.get("latest_abc_df")
            if abc_df_cached is None:
                abc_df_cached = products.abc_analysis(
                    filtered_sales, dashboard_comparison
                )
                st.session_state["latest_abc_df"] = abc_df_cached
            render_inventory_tab(
                filtered_sales,
                datasets["inventory"],
                abc_df_cached,
                global_filters,
            )
        elif active_tab == "資金":
            pnl_for_cash = st.session_state.get("latest_pnl_df", pnl_baseline)
            render_cash_tab(
                filtered_sales,
                dashboard_comparison,
                datasets["inventory"],
                pnl_for_cash,
                global_filters,
            )
    elif page_choice == "データ管理":
        integration_display = integration_result or st.session_state.get(
            "latest_api_result"
        )
        render_data_management_tab(
            validation_results,
            integration_display,
            baseline,
            sample_files_for_ui,
            templates,
        )
    else:
        render_help_settings_page()

if __name__ == "__main__":
    main()
