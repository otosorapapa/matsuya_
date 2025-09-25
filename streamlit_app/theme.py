"""Design tokens and theme helpers for the Matsya Streamlit app."""
from __future__ import annotations

from textwrap import dedent

import streamlit as st


DESIGN_TOKENS = {
    "primaryColor": "#0B1F3B",
    "secondaryColor": "#5A6B7A",
    "accentColor": "#1E88E5",
    "backgroundColor": "#F7F8FA",
    "cardBackground": "#FFFFFF",
    "successColor": "#3FB27E",
    "warningColor": "#E9A13B",
    "errorColor": "#E55353",
    "fontFamilyBase": "'Inter', 'Source Sans 3', sans-serif",
    "fontFamilyNumeric": "'Roboto Mono', monospace",
    "heading1Size": "28px",
    "heading2Size": "22px",
    "bodySize": "16px",
    "smallTextSize": "14px",
    "spacingUnit": "8px",
}


def build_custom_css() -> str:
    """Return the CSS snippet that applies the shared design tokens."""

    return dedent(
        f"""
        <style>
        :root {{
            --color-primary: {DESIGN_TOKENS['primaryColor']};
            --color-secondary: {DESIGN_TOKENS['secondaryColor']};
            --color-accent: {DESIGN_TOKENS['accentColor']};
            --color-background: {DESIGN_TOKENS['backgroundColor']};
            --color-surface: {DESIGN_TOKENS['cardBackground']};
            --color-success: {DESIGN_TOKENS['successColor']};
            --color-warning: {DESIGN_TOKENS['warningColor']};
            --color-error: {DESIGN_TOKENS['errorColor']};
            --font-base: {DESIGN_TOKENS['fontFamilyBase']};
            --font-numeric: {DESIGN_TOKENS['fontFamilyNumeric']};
            --heading-1: {DESIGN_TOKENS['heading1Size']};
            --heading-2: {DESIGN_TOKENS['heading2Size']};
            --text-body: {DESIGN_TOKENS['bodySize']};
            --text-small: {DESIGN_TOKENS['smallTextSize']};
            --spacing-unit: {DESIGN_TOKENS['spacingUnit']};
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
        </style>
        """
    ).strip()


def inject_custom_css() -> None:
    """Inject the custom CSS into the Streamlit app if it has not been added yet."""

    if st.session_state.get("_theme_css_injected"):
        return

    st.markdown(build_custom_css(), unsafe_allow_html=True)
    st.session_state["_theme_css_injected"] = True


__all__ = ["DESIGN_TOKENS", "build_custom_css", "inject_custom_css"]
