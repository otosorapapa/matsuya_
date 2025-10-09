"""Design tokens and theme helpers for the Matsya Streamlit app."""
from __future__ import annotations

from textwrap import dedent
from typing import Dict, Optional

import streamlit as st


BASE_TOKENS = {
    "primaryColor": "#0B1F3B",
    "secondaryColor": "#5A6B7A",
    "accentColor": "#1E88E5",
    "backgroundColor": "#F7F8FA",
    "cardBackground": "#FFFFFF",
    "surfaceMuted": "#EEF2F6",
    "successColor": "#3FB27E",
    "warningColor": "#E9A13B",
    "errorColor": "#E55353",
    "infoColor": "#2563EB",
    "fontFamilyBase": "'Inter', 'Source Sans 3', sans-serif",
    "fontFamilyNumeric": "'Roboto Mono', monospace",
    "heading1Size": "28px",
    "heading2Size": "22px",
    "heading3Size": "18px",
    "bodySize": "16px",
    "smallTextSize": "14px",
    "microTextSize": "12px",
    "spacingUnit": "8px",
    "radiusSm": "6px",
    "radiusMd": "12px",
    "radiusLg": "18px",
    "shadowSoft": "0 2px 6px rgba(15, 23, 42, 0.08)",
    "shadowLifted": "0 12px 30px rgba(15, 23, 42, 0.16)",
}

DARK_OVERRIDES = {
    "backgroundColor": "#111827",
    "cardBackground": "#1F2937",
    "surfaceMuted": "#374151",
    "primaryColor": "#C7D2FE",
    "secondaryColor": "#9CA3AF",
    "accentColor": "#60A5FA",
    "shadowSoft": "0 2px 6px rgba(0, 0, 0, 0.3)",
    "shadowLifted": "0 12px 30px rgba(0, 0, 0, 0.45)",
}

COLORBLIND_OVERRIDES = {
    "accentColor": "#0072B2",
    "successColor": "#009E73",
    "warningColor": "#F0E442",
    "errorColor": "#D55E00",
    "infoColor": "#56B4E9",
}

DESIGN_TOKENS = dict(BASE_TOKENS)


def resolve_design_tokens(mode: str = "light", palette: str = "default") -> Dict[str, str]:
    tokens = dict(BASE_TOKENS)
    if mode == "dark":
        tokens.update(DARK_OVERRIDES)
    if palette == "colorblind":
        tokens.update(COLORBLIND_OVERRIDES)
    return tokens


def get_design_tokens(
    mode: Optional[str] = None,
    palette: Optional[str] = None,
) -> Dict[str, str]:
    """Return design tokens taking into account user preferences."""

    preferences = st.session_state.get("ui_preferences", {})
    resolved_mode = mode or preferences.get("theme", "light")
    resolved_palette = palette or preferences.get("palette", "default")
    tokens = resolve_design_tokens(resolved_mode, resolved_palette)
    st.session_state["_current_theme_tokens"] = tokens
    return tokens


def build_custom_css(mode: str = "light", palette: str = "default") -> str:
    """Return the CSS snippet that applies the shared design tokens."""

    tokens = resolve_design_tokens(mode, palette)
    return dedent(
        f"""
        <style>
        :root {{
            --color-primary: {tokens['primaryColor']};
            --color-secondary: {tokens['secondaryColor']};
            --color-accent: {tokens['accentColor']};
            --color-background: {tokens['backgroundColor']};
            --color-surface: {tokens['cardBackground']};
            --color-surface-muted: {tokens['surfaceMuted']};
            --color-success: {tokens['successColor']};
            --color-warning: {tokens['warningColor']};
            --color-error: {tokens['errorColor']};
            --color-info: {tokens['infoColor']};
            --font-base: {tokens['fontFamilyBase']};
            --font-numeric: {tokens['fontFamilyNumeric']};
            --heading-1: {tokens['heading1Size']};
            --heading-2: {tokens['heading2Size']};
            --heading-3: {tokens['heading3Size']};
            --text-body: {tokens['bodySize']};
            --text-small: {tokens['smallTextSize']};
            --text-micro: {tokens['microTextSize']};
            --spacing-unit: {tokens['spacingUnit']};
            --radius-sm: {tokens['radiusSm']};
            --radius-md: {tokens['radiusMd']};
            --radius-lg: {tokens['radiusLg']};
            --shadow-soft: {tokens['shadowSoft']};
            --shadow-lifted: {tokens['shadowLifted']};
        }}

        html, body, .stApp {{
            background-color: var(--color-background);
            color: #1A1A1A;
            font-family: var(--font-base) !important;
            font-size: var(--text-body);
            line-height: 1.5;
        }}

        h1 {{
            font-size: var(--heading-1) !important;
            font-weight: 700 !important;
            color: var(--color-primary) !important;
            letter-spacing: 0.01em;
        }}

        h2 {{
            font-size: var(--heading-2) !important;
            font-weight: 600 !important;
            color: var(--color-primary) !important;
        }}

        h3, h4, h5, h6 {{
            color: var(--color-primary) !important;
            font-weight: 600;
        }}

        p, span, li {{
            font-size: var(--text-body);
            color: #1A1A1A;
        }}

        small {{
            font-size: var(--text-small);
            color: var(--color-secondary);
        }}

        a {{
            color: var(--color-accent);
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        /* Card styling */
        .stApp [data-testid="stVerticalBlock"] > div {{
            border-radius: 12px;
        }}

        [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{
            font-family: var(--font-numeric);
        }}

        [data-testid="metric-container"] {{
            background-color: var(--color-surface);
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            padding: calc(var(--spacing-unit) * 2);
        }}

        span[data-testid="delta-positive"] {{
            color: var(--color-success) !important;
        }}

        span[data-testid="delta-negative"] {{
            color: var(--color-error) !important;
        }}

        [data-testid="stHorizontalBlock"] > div {{
            gap: calc(var(--spacing-unit) * 2);
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: var(--color-surface);
            border-radius: 8px;
            padding: calc(var(--spacing-unit) * 1.5);
            gap: calc(var(--spacing-unit));
        }}

        .stTabs [data-baseweb="tab"] button {{
            color: var(--color-secondary);
            font-weight: 600;
        }}

        .stTabs [data-baseweb="tab"]:hover button {{
            color: var(--color-primary);
        }}

        .stTabs [aria-selected="true"] button {{
            color: var(--color-primary);
        }}

        /* Buttons */
        .stButton > button {{
            background-color: var(--color-primary);
            color: #FFFFFF;
            border-radius: 8px;
            padding: calc(var(--spacing-unit) * 1) calc(var(--spacing-unit) * 1.5);
            border: none;
            transition: background-color 0.2s ease, box-shadow 0.2s ease;
        }}

        .stButton > button:hover {{
            background-color: var(--color-accent);
            box-shadow: 0 4px 12px rgba(14, 31, 59, 0.18);
        }}

        .stButton > button:focus {{
            outline: 2px solid var(--color-accent);
            outline-offset: 2px;
        }}

        /* Inputs */
        .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput, .stDateInput {{
            width: 100%;
        }}

        .stSelectbox label, .stMultiSelect label, .stTextInput label,
        .stNumberInput label, .stDateInput label {{
            font-weight: 600;
            color: var(--color-secondary);
        }}

        .stSelectbox [data-baseweb="select"] {{
            border-radius: 8px;
        }}

        .stTooltip-content {{
            font-size: var(--text-small);
            background-color: var(--color-primary);
            color: #FFFFFF;
        }}

        .ds-surface {{
            background-color: var(--color-surface);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-soft);
            padding: calc(var(--spacing-unit) * 2);
            border: 1px solid rgba(11, 31, 59, 0.05);
        }}

        .ds-surface.elevated {{
            box-shadow: var(--shadow-lifted);
        }}

        .ds-surface.muted {{
            background-color: var(--color-surface-muted);
        }}

        .ds-section-title {{
            font-size: var(--heading-3);
            font-weight: 600;
            color: var(--color-primary);
            margin-bottom: calc(var(--spacing-unit) * 1.5);
        }}

        .ds-metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: calc(var(--spacing-unit) * 1.5);
        }}

        .ds-metric {{
            display: flex;
            flex-direction: column;
            gap: calc(var(--spacing-unit) * 0.5);
        }}

        .ds-metric .label {{
            font-size: var(--text-small);
            color: var(--color-secondary);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        .ds-metric .value {{
            font-size: 1.6rem;
            font-weight: 600;
            color: var(--color-primary);
        }}

        .ds-metric .caption {{
            font-size: var(--text-small);
            color: rgba(11, 31, 59, 0.7);
        }}

        .ds-chip {{
            display: inline-flex;
            align-items: center;
            gap: calc(var(--spacing-unit) * 0.5);
            padding: calc(var(--spacing-unit) * 0.5) calc(var(--spacing-unit) * 0.75);
            border-radius: 999px;
            font-size: var(--text-small);
            font-weight: 500;
            background-color: var(--color-surface-muted);
            color: var(--color-primary);
        }}

        .ds-chip.success {{
            background-color: rgba(63, 178, 126, 0.16);
            color: var(--color-success);
        }}

        .ds-chip.warning {{
            background-color: rgba(233, 161, 59, 0.18);
            color: var(--color-warning);
        }}

        .ds-chip.danger {{
            background-color: rgba(229, 83, 83, 0.18);
            color: var(--color-error);
        }}

        .ds-chip.info {{
            background-color: rgba(30, 136, 229, 0.14);
            color: var(--color-info);
        }}

        .ds-insight-list {{
            display: flex;
            flex-direction: column;
            gap: calc(var(--spacing-unit) * 1.5);
        }}

        .ds-insight {{
            display: flex;
            flex-direction: column;
            gap: calc(var(--spacing-unit) * 0.6);
            padding: calc(var(--spacing-unit) * 1.5);
            border-radius: var(--radius-md);
            border: 1px solid rgba(11, 31, 59, 0.08);
            background-color: var(--color-surface);
        }}

        .ds-insight strong {{
            font-size: 1rem;
            color: var(--color-primary);
        }}

        .ds-insight .meta {{
            display: flex;
            gap: calc(var(--spacing-unit));
            flex-wrap: wrap;
            font-size: var(--text-small);
            color: var(--color-secondary);
        }}
        </style>
        """
    ).strip()


def inject_custom_css(mode: str = "light", palette: str = "default") -> None:
    """Inject the custom CSS into the Streamlit app respecting the current theme."""

    key = f"{mode}:{palette}"
    cached_key = st.session_state.get("_theme_css_key")
    if cached_key == key:
        return
    st.markdown(build_custom_css(mode, palette), unsafe_allow_html=True)
    st.session_state["_theme_css_key"] = key
    st.session_state["_theme_css_injected"] = True


__all__ = [
    "DESIGN_TOKENS",
    "resolve_design_tokens",
    "get_design_tokens",
    "build_custom_css",
    "inject_custom_css",
]
