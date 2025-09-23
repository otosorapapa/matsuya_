"""Utilities for exporting tables as CSV/PDF within Streamlit."""
from __future__ import annotations

from io import BytesIO
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

PDF_FONT = "HeiseiMin-W3"
_pdf_registered = False


def _register_pdf_font() -> None:
    global _pdf_registered
    if not _pdf_registered:
        pdfmetrics.registerFont(UnicodeCIDFont(PDF_FONT))
        _pdf_registered = True


def csv_download(label: str, df: pd.DataFrame, file_name: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
    )


def _format_number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def pdf_download(label: str, title: str, df: pd.DataFrame, file_name: str) -> None:
    pdf_bytes = create_pdf_report(title, df.head(30))
    st.download_button(
        label=label,
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
    )


def create_pdf_report(title: str, df: pd.DataFrame) -> bytes:
    _register_pdf_font()
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    pdf.setFont(PDF_FONT, 16)
    pdf.drawString(margin, y, title)
    y -= 24

    pdf.setFont(PDF_FONT, 11)
    header = ["No."] + df.columns.tolist()
    pdf.drawString(margin, y, " | ".join(header))
    y -= 18

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        if y < margin:
            pdf.showPage()
            y = height - margin
            pdf.setFont(PDF_FONT, 11)
        values = [_format_number(row[col]) for col in df.columns]
        pdf.drawString(margin, y, " | ".join([str(idx), *values]))
        y -= 16

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()
