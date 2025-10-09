"""Reusable KPI card rendering utilities for the dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Mapping, Optional, Sequence

import streamlit as st

from streamlit_app.theme import get_design_tokens


@dataclass
class KpiCard:
    """Representation of a KPI with contextual metadata."""

    label: str
    value_text: str
    yoy: Optional[float] = None
    target_diff: Optional[float] = None
    unit: str = ""
    alert: bool = False
    action: Optional[Mapping[str, object]] = None
    category: Optional[str] = None
    definition: Optional[str] = None
    formula: Optional[str] = None
    target_setting: Optional[str] = None
    yoy_label: str = "前年比"
    delta_text: Optional[str] = None
    sub_value_text: Optional[str] = None
    sub_value_muted: bool = False


@dataclass
class KpiHighlight:
    """Representation of a high impact KPI highlight."""

    label: str
    value: str
    delta: Optional[float] = None
    delta_text: Optional[str] = None
    target_label: Optional[str] = None
    tooltip: Optional[str] = None
    category: Optional[str] = None
    definition: Optional[str] = None
    formula: Optional[str] = None
    target_setting: Optional[str] = None


def _build_card_body(card: KpiCard) -> str:
    yoy = card.yoy
    if yoy is None:
        delta_text = card.delta_text or f"{card.yoy_label}: データ不足"
        delta_class = "neutral"
    else:
        arrow = "▲" if yoy >= 0 else "▼"
        delta_value = f"{arrow} {yoy * 100:.1f}%"
        delta_text = card.delta_text or f"{card.yoy_label}: {delta_value}"
        delta_class = "positive" if yoy >= 0 else "negative"

    target_diff = card.target_diff or 0.0
    target_class = "positive" if target_diff >= 0 else "negative"
    unit_suffix = f" {escape(card.unit)}" if card.unit else ""

    classes = ["kpi-card"]
    if card.alert:
        classes.append("alert")
    elif yoy is not None and yoy <= -0.05:
        classes.append("alert")
    elif target_diff < 0:
        classes.append("caution")

    info_parts = []
    if card.category:
        info_parts.append(f"分類: {escape(card.category)}")
    if card.definition:
        info_parts.append(f"定義: {escape(card.definition)}")
    if card.formula:
        info_parts.append(f"計算式: {escape(card.formula)}")
    if card.target_setting:
        info_parts.append(f"目標設定: {escape(card.target_setting)}")
    if info_parts:
        info_attr_value = " &#10;".join(info_parts)
        info_attr = f' title="{info_attr_value}"'
    else:
        info_attr = ""
    category_chip = (
        f"<span class='kpi-card__chip'>{escape(card.category)}</span>"
        if card.category
        else ""
    )
    info_icon = ""
    if info_parts:
        info_icon = "<span class='kpi-card__info'>ℹ</span>"

    sub_value = ""
    if card.sub_value_text:
        sub_classes = "sub-value muted" if card.sub_value_muted else "sub-value"
        sub_value = (
            f"<div class='{sub_classes}'>{escape(card.sub_value_text)}</div>"
        )

    return (
        f"""
        <article class="{' '.join(classes)}"{info_attr}>
            <div class="kpi-card__header">
                <span class="label">{escape(card.label)}</span>
                {category_chip}
                {info_icon}
            </div>
            <div class="value">{escape(card.value_text)}</div>
            {sub_value}
            <div class="delta {delta_class}">{escape(delta_text)}</div>
            <div class="target {target_class}">
                目標差: {target_diff:+,.0f}{unit_suffix}
            </div>
        </article>
        """
    )


def render_kpi_cards(cards: Sequence[KpiCard]) -> None:
    """Render a responsive grid of KPI cards."""

    items = list(cards)
    if not items:
        return

    grid_items = [_build_card_body(card) for card in items]
    tokens = get_design_tokens()
    grid_html = "".join(grid_items)
    st.markdown(
        f"""
        <div class="kpi-card-grid" style="--gap:{tokens['spacingUnit']};">
            {grid_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    for card in items:
        action = card.action or {}
        action_type = action.get("type")
        if action and action_type != "link":
            st.button(
                action.get("label", "詳細"),
                key=action.get("key") or f"kpi-action-{card.label}",
                help=action.get("help"),
                on_click=action.get("on_click"),
                args=action.get("args", ()),
                kwargs=action.get("kwargs", {}),
                type=action.get("button_type", "secondary"),
                use_container_width=True,
            )
        elif action_type == "link":
            url = action.get("url")
            if url:
                st.markdown(
                    f"<a class='kpi-card__link' href='{escape(url)}' target='_blank'>{escape(action.get('label', '詳細を見る'))}</a>",
                    unsafe_allow_html=True,
                )


def render_kpi_highlights(highlights: Sequence[KpiHighlight]) -> None:
    """Render a responsive group of KPI highlight cards."""

    items = list(highlights)
    if not items:
        return

    tokens = get_design_tokens()
    cards_html = []
    for highlight in items:
        tooltip_parts = []
        if highlight.category:
            tooltip_parts.append(f"分類: {escape(highlight.category)}")
        if highlight.definition:
            tooltip_parts.append(f"定義: {escape(highlight.definition)}")
        if highlight.formula:
            tooltip_parts.append(f"計算式: {escape(highlight.formula)}")
        if highlight.target_setting:
            tooltip_parts.append(f"目標設定: {escape(highlight.target_setting)}")
        if tooltip_parts:
            tooltip_attr_value = " &#10;".join(tooltip_parts)
            tooltip_attr = f' title="{tooltip_attr_value}"'
        else:
            tooltip_attr = ""
        delta = highlight.delta
        if delta is None:
            delta_text = highlight.delta_text or "比較データなし"
            delta_color = "var(--color-warning)"
        else:
            arrow = "▲" if delta >= 0 else "▼"
            default_text = f"{arrow} {abs(delta) * 100:.1f}pt"
            delta_text = highlight.delta_text or default_text
            delta_color = (
                "var(--color-success)" if delta >= 0 else "var(--color-error)"
            )
        target_label = (
            f"<div class='kpi-highlight__target'>{escape(highlight.target_label)}" "</div>"
            if highlight.target_label
            else ""
        )
        category_chip = (
            f"<span class='kpi-highlight__chip'>{escape(highlight.category)}</span>"
            if highlight.category
            else ""
        )
        cards_html.append(
            f"""
            <div class="kpi-highlight"{tooltip_attr}>
                <div class="kpi-highlight__meta">
                    <span class="label">{escape(highlight.label)}</span>
                    {category_chip}
                </div>
                <div class="value">{escape(highlight.value)}</div>
                <div class="delta" style="color:{delta_color};">{escape(delta_text)}</div>
                {target_label}
            </div>
            """
        )

    st.markdown(
        f"""
        <section class="kpi-highlight-wrapper">
            <header>主要KPIハイライト</header>
            <div class="kpi-highlight-grid" style="--gap:{tokens['spacingUnit']};">
                {''.join(cards_html)}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_glossary(cards: Sequence[KpiCard]) -> None:
    """Render an info box summarising KPI definitions grouped by category."""

    if not cards:
        return

    grouped: dict[str, list[KpiCard]] = {}
    for card in cards:
        key = card.category or "その他"
        grouped.setdefault(key, []).append(card)

    sections = []
    for category, values in grouped.items():
        bullet_items = []
        for card in values:
            parts = [escape(card.definition or "定義未設定")]
            if card.formula:
                parts.append(f"計算式: {escape(card.formula)}")
            if card.target_setting:
                parts.append(f"目標: {escape(card.target_setting)}")
            bullet_items.append(
                f"<li><strong>{escape(card.label)}</strong> — {' ／ '.join(parts)}</li>"
            )
        sections.append(
            f"<h5>{escape(category)}</h5><ul>{''.join(bullet_items)}</ul>"
        )

    st.info("\n".join([f"- {card.label}: {card.definition or '定義未設定'}" for card in cards]))
    st.markdown(
        f"<div class='kpi-glossary'>{''.join(sections)}</div>",
        unsafe_allow_html=True,
    )


__all__ = [
    "KpiCard",
    "KpiHighlight",
    "render_kpi_cards",
    "render_kpi_highlights",
    "render_glossary",
]
