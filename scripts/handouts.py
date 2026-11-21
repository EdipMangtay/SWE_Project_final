"""Build the technical report and recording handout from their Markdown sources."""

import re
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from matplotlib import get_data_path
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from sentinel.config import ROOT

DOCS = ROOT / "docs"
INK = "#152E36"
TEAL = "#09665F"
MUTED = "#546769"
WIDTH = A4[0] - 38 * mm


def inline(text):
    parts = re.split(r"(\[[^\]]+\]\(https?://[^)]+\)|\*\*.+?\*\*|`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            out.append("<b>" + escape(part[2:-2]) + "</b>")
        elif part.startswith("`") and part.endswith("`"):
            out.append("<font color='#095B57'>" + escape(part[1:-1]) + "</font>")
        elif part.startswith("[") and "](" in part:
            label, url = part[1:-1].split("](", 1)
            out.append(f"<a color='{TEAL}' href={quoteattr(url)}><u>{escape(label)}</u></a>")
        else:
            out.append(escape(part))
    return "".join(out)


def style(name, size, leading, **kwargs):
    return ParagraphStyle(name, fontName=kwargs.pop("fontName", "DocumentSans"), fontSize=size,
                          leading=leading, textColor=colors.HexColor(INK), **kwargs)


BODY = style("body", 10.2, 15, spaceAfter=9)
TITLE = style("title", 31, 36, fontName="DocumentBold", spaceAfter=7)
SECTION = style("section", 20, 25, fontName="DocumentBold", spaceAfter=13)
SUB = style("sub", 12, 16, fontName="DocumentBold", spaceBefore=5, spaceAfter=4)
CELL = style("cell", 8.3, 11.5)
CUE = style("cue", 9.5, 13.3)
RIGHT = style("right", 8.3, 11.5, alignment=TA_RIGHT)
HEAD = style("thead", 8.1, 11, fontName="DocumentBold")
HEAD.textColor = colors.white
FRACTIONS = {
    "Split": [0.16, 0.13, 0.11, 0.13, 0.47],
    "Süre": [0.135, 0.435, 0.43],
    "Slayt": [0.115, 0.23, 0.655],
    "Macro-F1": [0.29, 0.13, 0.14, 0.14, 0.15, 0.15],
    "Offensive precision": [0.29, 0.18, 0.17, 0.17, 0.19],
}


def make_table(lines):
    matrix = [[v.strip() for v in line.strip("|").split("|")] for line in lines]
    header, rules, *data = matrix
    widths = FRACTIONS.get(header[0]) or FRACTIONS[header[1]]
    body_style = CUE if header[0] == "Süre" else CELL
    cells = [[Paragraph(escape(v), HEAD) for v in header]]
    for row in data:
        cells.append([Paragraph(inline(v), RIGHT if rule.endswith(":") else body_style)
                      for v, rule in zip(row, rules)])
    table = Table(cells, colWidths=[w * WIDTH for w in widths], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TEAL)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EAF3F1"), colors.HexColor("#F7F9F8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def render(source, destination, title):
    story = []
    lines = (DOCS / source).read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        if line == "<!-- page -->":
            story.append(PageBreak())
        elif line.startswith("|"):
            group = [line]
            while index < len(lines) and lines[index].startswith("|"):
                group.append(lines[index])
                index += 1
            story.extend([make_table(group), Spacer(1, 10)])
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
            figure = Image(str(DOCS / match[2]), width=WIDTH, height=190, kind="proportional")
            story.extend([figure, Spacer(1, 6)])
        else:
            prefix = next((p for p in ("### ", "## ", "# ") if line.startswith(p)), "")
            text = line[len(prefix):]
            story.append(Paragraph(inline(text), {"# ": TITLE, "## ": SECTION, "### ": SUB}.get(prefix, BODY)))
    pdf = SimpleDocTemplate(str(DOCS / destination), pagesize=A4, title=title,
                            author="Ali Edip Mangtay", leftMargin=19*mm, rightMargin=19*mm,
                            topMargin=23*mm, bottomMargin=19*mm)

    def frame(canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(colors.HexColor(TEAL))
        canvas.setFont("DocumentBold", 9)
        canvas.drawString(19*mm, h-13*mm, "SENTINELAI  /  SWE015")
        canvas.setFont("DocumentSans", 8)
        canvas.drawRightString(w-19*mm, h-13*mm, title)
        canvas.setStrokeColor(colors.HexColor("#BCD5D0"))
        canvas.line(19*mm, h-16*mm, w-19*mm, h-16*mm)
        canvas.line(19*mm, 15*mm, w-19*mm, 15*mm)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(19*mm, 10.5*mm, "Ali Edip Mangtay  ·  210911044")
        canvas.drawRightString(w-19*mm, 10.5*mm, str(doc.page))
        canvas.restoreState()

    pdf.build(story, onFirstPage=frame, onLaterPages=frame)


if __name__ == "__main__":
    fonts = Path(get_data_path()) / "fonts/ttf"
    pdfmetrics.registerFont(TTFont("DocumentSans", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DocumentBold", str(fonts / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DocumentSans", normal="DocumentSans", bold="DocumentBold")
    render("experiment_report.md", "technical_report.pdf", "Technical report")
    render("video_rehberi.md", "video_rehberi.pdf", "Çekim provası")
