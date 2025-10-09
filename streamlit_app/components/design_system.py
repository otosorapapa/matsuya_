"""Reusable UI helpers that embrace the shared design system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import streamlit as st

from streamlit_app.theme import get_design_tokens


@dataclass
class Metric:
    """Representation of a metric cell rendered inside a grid."""

    label: str
    value: str
    caption: Optional[str] = None


@dataclass
class Insight:
    """Structure representing an AI generated or rule based suggestion."""

    title: str
    description: str
    severity: str = "info"
    tags: Sequence[str] = ()


SEVERITY_TO_CHIP_CLASS = {
    "success": "success",
    "info": "info",
    "warning": "warning",
    "danger": "danger",
}


def render_surface(content: str, *, elevated: bool = False, muted: bool = False) -> None:
    """Render HTML content inside a styled surface container."""

    classes: List[str] = ["ds-surface"]
    if elevated:
        classes.append("elevated")
    if muted:
        classes.append("muted")
    st.markdown(
        f"""
        <div class="{' '.join(classes)}">
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(metrics: Iterable[Metric]) -> None:
    """Render a responsive grid of metrics following the design system."""

    metric_list = list(metrics)
    if not metric_list:
        return
    grid_items = []
    for metric in metric_list:
        caption_html = (
            f"<div class='caption'>{metric.caption}</div>" if metric.caption else ""
        )
        grid_items.append(
            f"""
            <div class="ds-metric">
                <div class="label">{metric.label}</div>
                <div class="value">{metric.value}</div>
                {caption_html}
            </div>
            """
        )
    grid_html = "".join(grid_items)
    render_surface(f"<div class='ds-metric-grid'>{grid_html}</div>")


def render_section_title(title: str, *, chip: Optional[str] = None, severity: str = "info") -> None:
    """Render a heading styled using the design tokens."""

    tokens = get_design_tokens()
    chip_html = ""
    if chip:
        chip_class = SEVERITY_TO_CHIP_CLASS.get(severity, "info")
        chip_html = (
            f"<span class='ds-chip {chip_class}'>{chip}</span>"
        )
    st.markdown(
        f"""
        <div class="ds-section-title" style="display:flex;align-items:center;gap:{tokens['spacingUnit']};">
            <span>{title}</span>
            {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insights(insights: Iterable[Insight]) -> None:
    """Render a collection of insights with severity chips and metadata."""

    items = list(insights)
    if not items:
        st.info("現在表示できるアドバイスはありません。")
        return
    insight_html = []
    for insight in items:
        chip_class = SEVERITY_TO_CHIP_CLASS.get(insight.severity, "info")
        tags_html = "".join(
            f"<span class='ds-chip info'>{tag}</span>" for tag in insight.tags
        )
        insight_html.append(
            f"""
            <div class="ds-insight">
                <div class="ds-chip {chip_class}" style="width:fit-content;">{insight.severity.upper()}</div>
                <strong>{insight.title}</strong>
                <div>{insight.description}</div>
                <div class="meta">{tags_html}</div>
            </div>
            """
        )
    render_surface("<div class='ds-insight-list'>" + "".join(insight_html) + "</div>")


__all__ = [
    "Metric",
    "Insight",
    "render_surface",
    "render_metric_grid",
    "render_section_title",
    "render_insights",
]
