"""Render the editable Markdown as a compact PDF; all outputs stay in this folder."""

import csv
from html import escape
import importlib.metadata
import json
from pathlib import Path
import re

import pymupdf as fitz
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = A4
MARGIN = 42
CONTENT_WIDTH = WIDTH - MARGIN * 2
NAVY = colors.HexColor("#172E43")
TEAL = colors.HexColor("#166675")
GRAY = colors.HexColor("#53616D")
LIGHT = colors.HexColor("#F0F4F7")


def fonts():
    mac = Path("/System/Library/Fonts/Supplemental")
    linux = Path("/usr/share/fonts/truetype/dejavu")
    if (mac / "Arial.ttf").exists():
        paths = [mac / name for name in ("Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf")]
    elif (linux / "DejaVuSans.ttf").exists():
        paths = [linux / name for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf")]
    else:
        raise RuntimeError("Install Arial or DejaVu Sans TrueType fonts before rendering.")
    for name, path in zip(("Body", "BodyBold", "BodyItalic", "BodyBoldItalic"), paths):
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="BodyItalic", boldItalic="BodyBoldItalic")


def inline(text):
    text = escape(text)
    text = re.sub(r"`([^`]+)`", lambda m: '<font name="Courier" size="8.1">' + m.group(1) + '</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8E0E6"))
    canvas.line(MARGIN, HEIGHT - 29, WIDTH - MARGIN, HEIGHT - 29)
    canvas.setFont("Body", 7.2)
    canvas.setFillColor(GRAY)
    canvas.drawString(MARGIN, HEIGHT - 22, "EZ-PENTA  /  ENGINEERING RESEARCH")
    canvas.drawRightString(WIDTH - MARGIN, HEIGHT - 22, "Evidence review: 05 Nov 2026")
    canvas.line(MARGIN, 29, WIDTH - MARGIN, 29)
    canvas.drawString(MARGIN, 17, "Local report  |  Automated guard labels are not human ground truth")
    canvas.drawRightString(WIDTH - MARGIN, 17, f"{document.page}")
    canvas.restoreState()


def main():
    fonts()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyTextReport", fontName="Body", fontSize=9.25, leading=12.15,
                              textColor=NAVY, spaceAfter=6, splitLongWords=True))
    styles.add(ParagraphStyle("TitleReport", parent=styles["BodyTextReport"], fontName="BodyBold",
                              fontSize=20, leading=23, spaceAfter=8, keepWithNext=True))
    styles.add(ParagraphStyle("H2Report", parent=styles["BodyTextReport"], fontName="BodyBold",
                              fontSize=12.2, leading=15, textColor=TEAL, spaceBefore=9,
                              spaceAfter=6, keepWithNext=True))
    styles.add(ParagraphStyle("H3Report", parent=styles["BodyTextReport"], fontName="BodyBold",
                              fontSize=10.3, leading=13, spaceBefore=7, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle("TableReport", parent=styles["BodyTextReport"], fontSize=8.1,
                              leading=10.3, spaceAfter=0))
    styles.add(ParagraphStyle("ListReport", parent=styles["BodyTextReport"], leftIndent=11,
                              firstLineIndent=-11, spaceAfter=5))
    lines = (HERE / "report.md").read_text().splitlines()
    story, paragraph, table_lines = [], [], []
    table_number = 0

    def flush_paragraph():
        if paragraph:
            text = " ".join(paragraph)
            style = styles["ListReport"] if re.match(r"^\d+\. ", text) else styles["BodyTextReport"]
            story.append(Paragraph(inline(text), style))
            paragraph.clear()

    def flush_table():
        nonlocal table_number
        if not table_lines:
            return
        parsed = [[part.strip() for part in line.strip().strip("|").split("|")]
                  for line in table_lines]
        data = [parsed[0], *parsed[2:]]
        widths = {
            0: [95, 280, CONTENT_WIDTH - 375],
            1: [178, 112, CONTENT_WIDTH - 290],
            2: [130, 132, 132, CONTENT_WIDTH - 394],
            3: [112, 135, 57, 138, CONTENT_WIDTH - 442],
            4: [180, 165, CONTENT_WIDTH - 345],
        }[table_number]
        cells = [[Paragraph(("<b>" if index == 0 else "") + inline(cell) + ("</b>" if index == 0 else ""),
                            styles["TableReport"]) for cell in row] for index, row in enumerate(data)]
        table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE9EF")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#9CB6C4")),
        ]))
        story.extend([table, Spacer(1, 7)])
        table_number += 1
        table_lines.clear()

    for line in lines:
        if line.startswith("|"):
            flush_paragraph()
            table_lines.append(line)
            continue
        flush_table()
        if line == "<!-- pagebreak -->":
            flush_paragraph()
            story.append(PageBreak())
        elif line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline(line[4:]), styles["H3Report"]))
        elif line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(inline(line[3:]), styles["H2Report"]))
        elif line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(inline(line[2:]), styles["TitleReport"]))
        elif re.match(r"^\d+\. ", line):
            flush_paragraph()
            story.append(Paragraph(inline(line), styles["ListReport"]))
        elif not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line.strip())
    flush_paragraph()
    flush_table()
    output = HERE / "report.pdf"
    document = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=MARGIN, leftMargin=MARGIN,
                                 topMargin=39, bottomMargin=42, title="ez-penta: Multilingual Safety Progress Report",
                                 author="ez-penta", subject="Read-only audit of the 28,800-answer Japanese-English safety experiment")
    document.build(story, onFirstPage=footer, onLaterPages=footer)

    doc = fitz.open(output)
    previews = HERE / "previews"
    previews.mkdir(exist_ok=True)
    checks = []
    for index, page in enumerate(doc):
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.3, 1.3))
        pixmap.save(previews / f"page-{index + 1}.png")
        outside = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    if x0 < 10 or y0 < 4 or x1 > page.rect.width - 10 or y1 > page.rect.height - 4:
                        outside.append(span["text"])
        text = page.get_text()
        assert "\ufffd" not in text
        checks.append({"page": index + 1, "text_characters": len(text), "outside_page_bounds": outside})
    assert len(doc) == 5, f"Expected 5 pages, got {len(doc)}; inspect pagination."
    assert not any(page["outside_page_bounds"] for page in checks)
    extracted = "\n".join(page.get_text() for page in doc)
    for text in ("28,800", "27,355", "6.33", "114", "487", "51-test", "0.0153"):
        assert text in extracted, f"Missing key text: {text}"
    (HERE / "pdf-validation.json").write_text(json.dumps({"pages": checks, "page_count": len(doc),
        "reportlab_version": importlib.metadata.version("reportlab"),
        "pymupdf_version": importlib.metadata.version("pymupdf")}, indent=2) + "\n")
    stats = json.loads((HERE / "statistics.json").read_text())
    for key in ("variant_language", "language_comparisons", "variant_changes", "risk_comparisons", "original_by_split"):
        with (HERE / f"{key}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(stats[key][0]))
            writer.writeheader()
            writer.writerows(stats[key])
    print(f"Rendered {len(doc)} pages: {output}")
    print("Page text lengths:", [page["text_characters"] for page in checks])


if __name__ == "__main__":
    main()
