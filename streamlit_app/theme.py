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


def get_design_tokens() -> Dict[str, str]:
    """Return a copy of the global design token dictionary."""

    return dict(DESIGN_TOKENS)


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
            --color-surface-muted: {DESIGN_TOKENS['surfaceMuted']};
            --color-success: {DESIGN_TOKENS['successColor']};
            --color-warning: {DESIGN_TOKENS['warningColor']};
            --color-error: {DESIGN_TOKENS['errorColor']};
            --color-info: {DESIGN_TOKENS['infoColor']};
            --font-base: {DESIGN_TOKENS['fontFamilyBase']};
            --font-numeric: {DESIGN_TOKENS['fontFamilyNumeric']};
            --heading-1: {DESIGN_TOKENS['heading1Size']};
            --heading-2: {DESIGN_TOKENS['heading2Size']};
            --heading-3: {DESIGN_TOKENS['heading3Size']};
            --text-body: {DESIGN_TOKENS['bodySize']};
            --text-small: {DESIGN_TOKENS['smallTextSize']};
            --text-micro: {DESIGN_TOKENS['microTextSize']};
            --spacing-unit: {DESIGN_TOKENS['spacingUnit']};
            --radius-sm: {DESIGN_TOKENS['radiusSm']};
            --radius-md: {DESIGN_TOKENS['radiusMd']};
            --radius-lg: {DESIGN_TOKENS['radiusLg']};
            --shadow-soft: {DESIGN_TOKENS['shadowSoft']};
            --shadow-lifted: {DESIGN_TOKENS['shadowLifted']};
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

        .kpi-card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: calc(var(--spacing-unit) * 1.5);
            margin-bottom: calc(var(--spacing-unit) * 1.5);
        }}

        .kpi-card {{
            background: linear-gradient(135deg, rgba(30, 136, 229, 0.08), #fff);
            border-radius: var(--radius-md);
            padding: calc(var(--spacing-unit) * 1.5);
            border: 1px solid rgba(11, 31, 59, 0.08);
            box-shadow: var(--shadow-soft);
            display: flex;
            flex-direction: column;
            gap: calc(var(--spacing-unit) * 0.75);
            position: relative;
        }}

        .kpi-card.alert {{
            border-color: var(--color-error);
            box-shadow: 0 12px 28px rgba(229, 83, 83, 0.25);
        }}

        .kpi-card.caution {{
            border-color: var(--color-warning);
        }}

        .kpi-card__header {{
            display: flex;
            align-items: center;
            gap: calc(var(--spacing-unit) * 0.5);
            font-size: var(--text-small);
            color: var(--color-secondary);
            text-transform: none;
        }}

        .kpi-card__chip {{
            background-color: rgba(37, 99, 235, 0.12);
            color: var(--color-info);
            padding: 0 10px;
            border-radius: 999px;
            font-size: var(--text-micro);
            font-weight: 600;
        }}

        .kpi-card__info {{
            margin-left: auto;
            font-size: 0.85rem;
            color: rgba(11, 31, 59, 0.55);
        }}

        .kpi-card .value {{
            font-size: 2rem;
            font-weight: 600;
            font-family: var(--font-numeric);
            color: var(--color-primary);
        }}

        .kpi-card .sub-value {{
            font-size: var(--text-small);
            color: rgba(11, 31, 59, 0.75);
        }}

        .kpi-card .sub-value.muted {{
            color: rgba(11, 31, 59, 0.45);
        }}

        .kpi-card .delta {{
            font-size: 0.95rem;
            font-weight: 600;
        }}

        .kpi-card .delta.positive {{
            color: var(--color-success);
        }}

        .kpi-card .delta.negative {{
            color: var(--color-error);
        }}

        .kpi-card .target {{
            font-size: var(--text-small);
            color: rgba(11, 31, 59, 0.7);
        }}

        .kpi-card .target.positive {{
            color: var(--color-success);
        }}

        .kpi-card .target.negative {{
            color: var(--color-error);
        }}

        .kpi-card__link {{
            display: inline-block;
            margin-top: calc(var(--spacing-unit));
            font-weight: 600;
        }}

        .kpi-highlight-wrapper {{
            border-radius: var(--radius-lg);
            padding: calc(var(--spacing-unit) * 2);
            background: linear-gradient(135deg, rgba(63, 178, 126, 0.15), rgba(255, 255, 255, 0.95));
            border: 1px solid rgba(63, 178, 126, 0.35);
            margin-bottom: calc(var(--spacing-unit) * 2);
        }}

        .kpi-highlight-wrapper > header {{
            font-weight: 600;
            color: var(--color-primary);
            margin-bottom: calc(var(--spacing-unit) * 1.2);
        }}

        .kpi-highlight-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: calc(var(--spacing-unit) * 1.2);
        }}

        .kpi-highlight {{
            background-color: rgba(255, 255, 255, 0.92);
            border-radius: var(--radius-md);
            padding: calc(var(--spacing-unit) * 1.5);
            border: 1px solid rgba(63, 178, 126, 0.25);
            display: flex;
            flex-direction: column;
            gap: calc(var(--spacing-unit) * 0.6);
        }}

        .kpi-highlight__meta {{
            display: flex;
            align-items: center;
            gap: calc(var(--spacing-unit) * 0.5);
            font-size: var(--text-small);
            color: rgba(11, 31, 59, 0.7);
        }}

        .kpi-highlight__chip {{
            background-color: rgba(63, 178, 126, 0.14);
            color: var(--color-success);
            padding: 0 10px;
            border-radius: 999px;
            font-size: var(--text-micro);
            font-weight: 600;
        }}

        .kpi-highlight .value {{
            font-size: 1.8rem;
            font-weight: 600;
            color: var(--color-primary);
        }}

        .kpi-highlight__target {{
            font-size: var(--text-small);
            color: rgba(11, 31, 59, 0.65);
        }}

        .kpi-glossary {{
            margin-top: calc(var(--spacing-unit) * 1.5);
        }}

        .kpi-glossary h5 {{
            margin-bottom: calc(var(--spacing-unit));
            color: var(--color-primary);
        }}

        .kpi-glossary ul {{
            margin-top: 0;
            margin-bottom: calc(var(--spacing-unit) * 1.5);
            padding-left: calc(var(--spacing-unit) * 2);
        }}

        .sidebar-anchor-nav {{
            display: flex;
            flex-direction: column;
            gap: calc(var(--spacing-unit));
            margin-bottom: calc(var(--spacing-unit) * 2);
        }}

        .sidebar-anchor-nav a {{
            padding: calc(var(--spacing-unit) * 0.75) calc(var(--spacing-unit));
            border-radius: var(--radius-sm);
            background-color: rgba(11, 31, 59, 0.05);
            color: var(--color-primary);
            font-weight: 600;
        }}

        .sidebar-anchor-nav a:hover {{
            background-color: rgba(30, 136, 229, 0.15);
        }}

        .alert-card.high {{
            border-left: 6px solid var(--color-error);
        }}

        .alert-card.medium {{
            border-left: 6px solid var(--color-warning);
        }}

        .alert-card.low {{
            border-left: 6px solid var(--color-success);
        }}

        .alert-card {{
            background-color: var(--color-surface);
            border-radius: var(--radius-md);
            padding: calc(var(--spacing-unit) * 1.2);
            box-shadow: var(--shadow-soft);
            border: 1px solid rgba(11, 31, 59, 0.08);
            display: flex;
            flex-direction: column;
            gap: calc(var(--spacing-unit) * 0.5);
        }}

        .alert-card__title {{
            font-weight: 600;
            color: var(--color-primary);
        }}

        .alert-card__message {{
            color: rgba(11, 31, 59, 0.75);
        }}

        .alert-card__cause {{
            font-size: var(--text-small);
            color: rgba(11, 31, 59, 0.7);
        }}

        .alert-card__actions {{
            font-size: var(--text-small);
            color: rgba(11, 31, 59, 0.65);
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


__all__ = [
    "DESIGN_TOKENS",
    "get_design_tokens",
    "build_custom_css",
    "inject_custom_css",
]
