"""
pdf_report.py — drop-in replacement for the old create_pdf_report in app.py

Usage in app.py:
    from pdf_report import create_pdf_report   # replace the old inline function
"""

import tempfile
import os
import io
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ── Palette ──────────────────────────────────────────────────────────────────
DARK_BG    = colors.HexColor("#0f172a")
ACCENT     = colors.HexColor("#6366f1")
ACCENT2    = colors.HexColor("#22d3ee")
TEXT_MAIN  = colors.HexColor("#f1f5f9")
TEXT_MUTED = colors.HexColor("#94a3b8")
CARD_BG    = colors.HexColor("#1e293b")
BORDER     = colors.HexColor("#334155")

# ── Styles ────────────────────────────────────────────────────────────────────
def _styles():
    return {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=26,
            textColor=TEXT_MAIN,
            spaceAfter=2,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=10,
            textColor=TEXT_MUTED,
            spaceAfter=6,
        ),
        "section": ParagraphStyle(
            "section",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=ACCENT2,
            spaceBefore=14,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=10,
            textColor=TEXT_MAIN,
            leading=15,
            spaceAfter=4,
        ),
        "code": ParagraphStyle(
            "code",
            fontName="Courier",
            fontSize=9,
            textColor=colors.HexColor("#a5f3fc"),
            backColor=CARD_BG,
            leading=13,
            leftIndent=8,
            rightIndent=8,
            spaceAfter=4,
        ),
        "label": ParagraphStyle(
            "label",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=TEXT_MUTED,
            alignment=TA_CENTER,
        ),
        "metric": ParagraphStyle(
            "metric",
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=ACCENT,
            alignment=TA_CENTER,
        ),
    }


# ── Chart generator ───────────────────────────────────────────────────────────
def _make_chart(df: pd.DataFrame) -> str | None:
    """Return path to a PNG chart, or None if data isn't chartable."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols    = df.select_dtypes(exclude="number").columns.tolist()

    if not numeric_cols:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    palette = ["#6366f1", "#22d3ee", "#f59e0b", "#10b981", "#f43f5e"]

    if text_cols:
        categories = df[text_cols[0]].astype(str).tolist()
        x = range(len(categories))
        bar_w = 0.7 / max(len(numeric_cols), 1)

        for i, col in enumerate(numeric_cols):
            offset = (i - len(numeric_cols) / 2) * bar_w + bar_w / 2
            bars = ax.bar(
                [xi + offset for xi in x],
                df[col],
                width=bar_w * 0.9,
                color=palette[i % len(palette)],
                alpha=0.92,
                label=col,
                zorder=3
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(df[col]) * 0.01,
                    f"{bar.get_height():,.0f}",
                    ha="center", va="bottom",
                    color="#f1f5f9", fontsize=8
                )

        ax.set_xticks(list(x))
        ax.set_xticklabels(categories, color="#94a3b8", fontsize=9)
    else:
        for i, col in enumerate(numeric_cols[:1]):
            ax.plot(df[col], color=palette[0], linewidth=2, marker="o", markersize=4)
            ax.fill_between(range(len(df[col])), df[col], alpha=0.15, color=palette[0])

    ax.yaxis.set_tick_params(labelcolor="#94a3b8", labelsize=9)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.grid(axis="y", color="#334155", linestyle="--", linewidth=0.6, zorder=0)
    ax.yaxis.label.set_color("#94a3b8")

    if len(numeric_cols) > 1:
        ax.legend(
            facecolor="#1e293b", edgecolor="#334155",
            labelcolor="#f1f5f9", fontsize=8
        )

    plt.tight_layout(pad=0.5)
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    return path


# ── KPI card row ──────────────────────────────────────────────────────────────
def _kpi_table(rows: int, cols: int, styles_map: dict) -> Table:
    data = [
        [
            Paragraph(str(rows),  styles_map["metric"]),
            Paragraph(str(cols),  styles_map["metric"]),
            Paragraph("Success",  styles_map["metric"]),
        ],
        [
            Paragraph("ROWS",     styles_map["label"]),
            Paragraph("COLUMNS",  styles_map["label"]),
            Paragraph("STATUS",   styles_map["label"]),
        ],
    ]
    t = Table(data, colWidths=[55*mm, 55*mm, 55*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [CARD_BG, CARD_BG]),
        ("BOX",        (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID",  (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
    ]))
    return t


# ── Results table ─────────────────────────────────────────────────────────────
def _result_table(df: pd.DataFrame, styles_map: dict) -> Table:
    display = df.head(20)
    header  = [Paragraph(c, ParagraphStyle(
                    "th", fontName="Helvetica-Bold", fontSize=9,
                    textColor=TEXT_MAIN, alignment=TA_CENTER))
               for c in display.columns]

    rows = [[Paragraph(str(v), ParagraphStyle(
                 "td", fontName="Helvetica", fontSize=9,
                 textColor=TEXT_MAIN, alignment=TA_CENTER))
             for v in row]
            for row in display.itertuples(index=False)]

    col_w = 170 * mm / max(len(display.columns), 1)
    t = Table([header] + rows, colWidths=[col_w] * len(display.columns))
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  ACCENT),
        ("BACKGROUND",  (0, 1), (-1, -1), CARD_BG),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CARD_BG, colors.HexColor("#263348")]),
        ("BOX",         (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID",   (0, 0), (-1, -1), 0.3, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


# ── Page template (dark background on every page) ────────────────────────────
def _page_bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, 0, A4[0], A4[1], fill=True, stroke=False)

    # top accent stripe
    canvas.setFillColor(ACCENT)
    canvas.rect(0, A4[1] - 4*mm, A4[0], 4*mm, fill=True, stroke=False)

    # footer
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawString(20*mm, 10*mm, "AI Data Copilot — Confidential")
    canvas.drawRightString(A4[0] - 20*mm, 10*mm, f"Page {doc.page}")
    canvas.restoreState()


# ── Main entry point ──────────────────────────────────────────────────────────
def create_pdf_report(question: str, sql: str, result, insight: str) -> str:
    """
    Build a dark-themed, chart-enhanced PDF report.
    Returns the path to the generated PDF file.
    """
    S = _styles()
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=22*mm,  bottomMargin=20*mm,
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("🤖 AI Data Copilot", S["title"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8))

    # ── Question ──────────────────────────────────────────────────────────────
    story.append(Paragraph("QUESTION", S["section"]))
    story.append(Paragraph(question, S["body"]))

    # ── KPI cards ─────────────────────────────────────────────────────────────
    if isinstance(result, pd.DataFrame):
        story.append(Spacer(1, 6))
        story.append(_kpi_table(len(result), len(result.columns), S))

    # ── Chart ─────────────────────────────────────────────────────────────────
    if isinstance(result, pd.DataFrame):
        chart_path = _make_chart(result)
        if chart_path:
            story.append(Paragraph("CHART", S["section"]))
            story.append(RLImage(chart_path, width=170*mm, height=75*mm))
            story.append(Spacer(1, 4))

    # ── SQL ───────────────────────────────────────────────────────────────────
    story.append(Paragraph("GENERATED SQL", S["section"]))
    for line in sql.strip().split("\n"):
        story.append(Paragraph(line or " ", S["code"]))

    # ── Results table ─────────────────────────────────────────────────────────
    if isinstance(result, pd.DataFrame):
        story.append(Paragraph("RESULTS", S["section"]))
        story.append(_result_table(result, S))
        if len(result) > 20:
            story.append(Paragraph(
                f"Showing top 20 of {len(result)} rows.",
                ParagraphStyle("note", fontName="Helvetica-Oblique",
                               fontSize=8, textColor=TEXT_MUTED)
            ))

    # ── Insight ───────────────────────────────────────────────────────────────
    story.append(Paragraph("AI INSIGHT", S["section"]))
    for para in insight.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), S["body"]))
            story.append(Spacer(1, 4))

    doc.build(story, onFirstPage=_page_bg, onLaterPages=_page_bg)

    # clean up chart temp file
    if isinstance(result, pd.DataFrame):
        try:
            if chart_path:
                os.unlink(chart_path)
        except Exception:
            pass

    return out_path