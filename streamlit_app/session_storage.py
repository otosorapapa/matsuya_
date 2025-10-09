"""Helpers to persist selected Streamlit session state keys across reruns."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Mapping

logger = logging.getLogger(__name__)

SESSION_STATE_PATH = Path.home() / ".matsuya_app_state.json"


def load_state() -> Dict[str, object]:
    """Load persisted session state values from disk."""

    if not SESSION_STATE_PATH.exists():
        return {}
    try:
        with SESSION_STATE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to load persisted session state")
        return {}
    return data if isinstance(data, dict) else {}


def save_state(values: Mapping[str, object]) -> None:
    """Persist a subset of session state values to disk."""

    try:
        SESSION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SESSION_STATE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(values, handle, ensure_ascii=False, indent=2)
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to persist session state")


def extract_values(keys: Iterable[str], state: Mapping[str, object]) -> Dict[str, object]:
    """Return JSON-serialisable values from ``state`` limited to ``keys``."""

    data: Dict[str, object] = {}
    for key in keys:
        if key not in state:
            continue
        value = state.get(key)
        data[key] = value
    return data
