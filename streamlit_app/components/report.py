"""Utilities for exporting tables as CSV/PDF within Streamlit."""
from __future__ import annotations

from io import BytesIO
import textwrap
from typing import Sequence, Tuple

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


def build_markdown_report(
    title: str, sections: Sequence[Tuple[str, Sequence[str]]]
) -> str:
    """Build a Markdown report string from the provided sections."""

    lines = [f"# {title}", ""]
    for heading, contents in sections:
        if heading:
            lines.append(f"## {heading}")
            lines.append("")
        for paragraph in contents:
            lines.append(paragraph)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def create_text_pdf(title: str, sections: Sequence[Tuple[str, Sequence[str]]]) -> bytes:
    """Create a simple text-centric PDF from Markdown-like content."""

    _register_pdf_font()
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    def _new_page() -> None:
        nonlocal y
        pdf.showPage()
        pdf.setFont(PDF_FONT, 16)
        y = height - margin
        pdf.drawString(margin, y, title)
        y -= 32

    pdf.setFont(PDF_FONT, 16)
    pdf.drawString(margin, y, title)
    y -= 32

    text_width = max(36, int((width - margin * 2) / 12))

    for heading, contents in sections:
        if heading:
            if y < margin + 40:
                _new_page()
            pdf.setFont(PDF_FONT, 13)
            pdf.drawString(margin, y, heading)
            y -= 22
        pdf.setFont(PDF_FONT, 11)
        for paragraph in contents:
            raw_lines = paragraph.splitlines() or [""]
            for raw_line in raw_lines:
                line = raw_line.rstrip()
                is_bullet = line.strip().startswith("- ")
                content = line.strip()[2:].strip() if is_bullet else line.strip()
                if not content:
                    if y < margin + 20:
                        _new_page()
                        if heading:
                            pdf.setFont(PDF_FONT, 13)
                            pdf.drawString(margin, y, heading)
                            y -= 22
                            pdf.setFont(PDF_FONT, 11)
                    y -= 12
                    continue
                wrapped = textwrap.wrap(content, width=text_width) or [content]
                for idx, wrapped_line in enumerate(wrapped):
                    if y < margin + 20:
                        _new_page()
                        if heading:
                            pdf.setFont(PDF_FONT, 13)
                            pdf.drawString(margin, y, heading)
                            y -= 22
                            pdf.setFont(PDF_FONT, 11)
                    prefix = "• " if is_bullet and idx == 0 else ("  " if is_bullet else "")
                    pdf.drawString(margin, y, f"{prefix}{wrapped_line}")
                    y -= 16
                y -= 6
        y -= 6

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def render_dashboard_report_downloads(
    title: str, sections: Sequence[Tuple[str, Sequence[str]]], base_file_name: str
) -> None:
    """Render download buttons for Markdown/PDF dashboard reports."""

    markdown = build_markdown_report(title, sections)
    pdf_bytes = create_text_pdf(title, sections)
    col1, col2 = st.columns(2)
    col1.download_button(
        "Markdownでダウンロード",
        data=markdown.encode("utf-8"),
        file_name=f"{base_file_name}.md",
        mime="text/markdown",
    )
    col2.download_button(
        "PDFでダウンロード",
        data=pdf_bytes,
        file_name=f"{base_file_name}.pdf",
        mime="application/pdf",
    )
