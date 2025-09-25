"""Utility helpers for the Streamlit app package."""

from __future__ import annotations

from typing import Callable

import streamlit as st


def _get_rerun_callable() -> Callable[[], None]:
    """Return the appropriate Streamlit rerun function.

    Streamlit deprecated ``st.experimental_rerun`` in favour of ``st.rerun``
    starting from version 1.32.0. Deployments that have not yet upgraded
    still rely on ``st.experimental_rerun`` being available. To maintain
    compatibility across versions we dynamically fetch whichever API exists
    and fall back gracefully.
    """

    if hasattr(st, "rerun"):
        return getattr(st, "rerun")
    return getattr(st, "experimental_rerun")


def rerun() -> None:
    """Trigger a rerun of the current Streamlit script.

    This helper shields the rest of the code base from changes to the
    Streamlit API surface while still allowing us to call ``st.rerun`` when
    available. Once the minimum supported Streamlit version provides
    ``st.rerun`` we can collapse this helper.
    """

    _get_rerun_callable()()


__all__ = ["rerun"]

