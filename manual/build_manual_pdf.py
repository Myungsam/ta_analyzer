"""Render Manual_Quick.md and Manual_Detailed.md to PDF.

Uses reportlab with Malgun Gothic so Korean glyphs render correctly. The
Markdown dialect supported is the subset actually used by the manuals:
    - Heading 1/2/3 (#, ##, ###)
    - Numbered lists (1.) and bullet lists (-)
    - Pipe tables (| col | col |)
    - Blockquotes (>)
    - Bold (**...**) and inline code (`...`)
    - Horizontal rules (---)
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, ListFlowable, ListItem, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle, KeepTogether,
)


# ----------------------------------------------------------------------
# Fonts (Malgun Gothic for Korean)
# ----------------------------------------------------------------------
FONT_REG = 'Malgun'
FONT_BOLD = 'MalgunBold'
FONT_MONO = 'Courier'

_FONT_DIR = Path(r'C:\Windows\Fonts')
pdfmetrics.registerFont(TTFont(FONT_REG, str(_FONT_DIR / 'malgun.ttf')))
pdfmetrics.registerFont(TTFont(FONT_BOLD, str(_FONT_DIR / 'malgunbd.ttf')))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily(FONT_REG, normal=FONT_REG, bold=FONT_BOLD,
                   italic=FONT_REG, boldItalic=FONT_BOLD)


# ----------------------------------------------------------------------
# Styles
# ----------------------------------------------------------------------
_styles = getSampleStyleSheet()

H1 = ParagraphStyle(
    'H1', parent=_styles['Heading1'],
    fontName=FONT_BOLD, fontSize=20, leading=26,
    textColor=colors.HexColor('#1f2a55'),
    spaceBefore=8, spaceAfter=10,
)
H2 = ParagraphStyle(
    'H2', parent=_styles['Heading2'],
    fontName=FONT_BOLD, fontSize=15, leading=20,
    textColor=colors.HexColor('#264f8e'),
    spaceBefore=14, spaceAfter=6,
)
H3 = ParagraphStyle(
    'H3', parent=_styles['Heading3'],
    fontName=FONT_BOLD, fontSize=12.5, leading=17,
    textColor=colors.HexColor('#2c5f8a'),
    spaceBefore=10, spaceAfter=4,
)
H4 = ParagraphStyle(
    'H4', parent=_styles['Heading4'],
    fontName=FONT_BOLD, fontSize=11, leading=15,
    textColor=colors.HexColor('#365a78'),
    spaceBefore=8, spaceAfter=3,
)
BODY = ParagraphStyle(
    'Body', parent=_styles['BodyText'],
    fontName=FONT_REG, fontSize=10, leading=15,
    spaceBefore=2, spaceAfter=4,
    alignment=TA_LEFT,
)
LIST_ITEM = ParagraphStyle(
    'ListItem', parent=BODY, leftIndent=0, spaceBefore=0, spaceAfter=2)
QUOTE = ParagraphStyle(
    'Quote', parent=BODY,
    leftIndent=12, rightIndent=8,
    fontName=FONT_REG, fontSize=9.5, leading=14,
    textColor=colors.HexColor('#555'),
    borderColor=colors.HexColor('#bbb'),
    borderPadding=(2, 6, 2, 6),
    backColor=colors.HexColor('#f5f7fb'),
    spaceBefore=4, spaceAfter=6,
)
TABLE_HEADER = ParagraphStyle(
    'TH', parent=BODY, fontName=FONT_BOLD, fontSize=9.5, leading=12,
    textColor=colors.white, alignment=TA_LEFT)
TABLE_CELL = ParagraphStyle(
    'TD', parent=BODY, fontName=FONT_REG, fontSize=9.5, leading=12,
    alignment=TA_LEFT, spaceBefore=0, spaceAfter=0)


# ----------------------------------------------------------------------
# Inline-markdown -> reportlab mini-HTML
# ----------------------------------------------------------------------
def _escape(text: str) -> str:
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;'))


def inline(text: str) -> str:
    """Convert a subset of inline markdown to reportlab's tiny HTML."""
    # 1) Protect inline code spans with placeholders
    codes: list[str] = []

    def _grab_code(m: re.Match) -> str:
        codes.append(_escape(m.group(1)))
        return f'\x00CODE{len(codes) - 1}\x00'

    text = re.sub(r'`([^`]+)`', _grab_code, text)

    # 2) Now escape everything else
    text = _escape(text)

    # 3) **bold**
    text = re.sub(r'\*\*([^*]+)\*\*',
                  rf'<font name="{FONT_BOLD}">\1</font>', text)

    # 4) Restore code spans with a monospace font + light background
    def _put_code(m: re.Match) -> str:
        idx = int(m.group(1))
        return (f'<font name="{FONT_MONO}" size="9" '
                f'backColor="#eef0f3">{codes[idx]}</font>')

    text = re.sub(r'\x00CODE(\d+)\x00', _put_code, text)
    return text


def P(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(inline(text), style)


# ----------------------------------------------------------------------
# Block parser
# ----------------------------------------------------------------------
def _make_table(rows: list[list[str]]) -> Table:
    """rows[0] = header, rows[1:] = body."""
    n_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < n_cols:
            r.append('')
    cells: list[list[Paragraph]] = []
    for i, r in enumerate(rows):
        style = TABLE_HEADER if i == 0 else TABLE_CELL
        cells.append([Paragraph(inline(c), style) for c in r])

    # Roughly proportional column widths from header text length
    header_lens = [max(1, len(c)) for c in rows[0]]
    total = sum(header_lens)
    avail = 165 * mm
    col_widths = [avail * (L / total) for L in header_lens]

    t = Table(cells, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#264f8e')),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#9aa6b8')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#f3f5f9')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


_table_row_re = re.compile(r'^\s*\|(.+)\|\s*$')
_table_sep_re = re.compile(r'^\s*\|?[\s\-:|]+\|?\s*$')


def _split_row(line: str) -> list[str]:
    m = _table_row_re.match(line)
    if not m:
        return []
    cells = m.group(1).split('|')
    return [c.strip() for c in cells]


def md_to_flowables(md_text: str) -> list:
    """Parse the markdown into reportlab flowables."""
    out: list = []
    lines = md_text.splitlines()
    i = 0
    n = len(lines)

    def flush_paragraph(buf: list[str]):
        text = ' '.join(s.strip() for s in buf).strip()
        if text:
            out.append(P(text))

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            i += 1
            continue

        # Headings
        if stripped.startswith('# '):
            out.append(P(stripped[2:].strip(), H1))
            out.append(HRFlowable(
                width='100%', thickness=0.8,
                color=colors.HexColor('#264f8e'),
                spaceBefore=2, spaceAfter=10))
            i += 1
            continue
        if stripped.startswith('## '):
            out.append(P(stripped[3:].strip(), H2))
            i += 1
            continue
        if stripped.startswith('### '):
            out.append(P(stripped[4:].strip(), H3))
            i += 1
            continue
        if stripped.startswith('#### '):
            out.append(P(stripped[5:].strip(), H4))
            i += 1
            continue

        # Horizontal rule
        if stripped in ('---', '***', '___'):
            out.append(HRFlowable(
                width='100%', thickness=0.4,
                color=colors.HexColor('#888'),
                spaceBefore=6, spaceAfter=6))
            i += 1
            continue

        # Pipe table  (header row, then separator row, then body rows)
        if (_table_row_re.match(line) and i + 1 < n
                and _table_sep_re.match(lines[i + 1])):
            header = _split_row(line)
            i += 2  # skip header + separator
            body: list[list[str]] = []
            while i < n and _table_row_re.match(lines[i]):
                body.append(_split_row(lines[i]))
                i += 1
            out.append(_make_table([header] + body))
            out.append(Spacer(1, 4))
            continue

        # Blockquote
        if stripped.startswith('> '):
            quote_buf: list[str] = []
            while i < n and lines[i].lstrip().startswith('>'):
                quote_buf.append(lines[i].lstrip()[1:].lstrip())
                i += 1
            qtext = ' '.join(s for s in quote_buf if s)
            if not qtext.strip():
                qtext = ' '
            out.append(P(qtext, QUOTE))
            continue

        # Numbered list — honor source numbers and fold any indented
        # sub-bullets into the preceding numbered item as continuation
        # lines (one bullet glyph per sub-bullet).  This keeps the count
        # correct without having to nest ListFlowables (which made
        # Platypus loop on this content).
        m_num = re.match(r'^(\d+)\.\s+(.*)$', line)
        if m_num is not None:
            start_num = int(m_num.group(1))
            items: list[ListItem] = []
            while i < n:
                ln = lines[i]
                m = re.match(r'^(\d+)\.\s+(.*)$', ln)
                if m is None:
                    break
                body_lines = [m.group(2).rstrip()]
                i += 1
                while i < n and re.match(r'^[ \t]+[-*]\s+', lines[i]):
                    sub_text = re.sub(r'^[ \t]+[-*]\s+', '',
                                      lines[i]).rstrip()
                    body_lines.append(f'<br/>&nbsp;&nbsp;&#8226;&nbsp;'
                                      f'{inline(sub_text)}')
                    i += 1
                # body_lines[0] still needs inline-rendering; subsequent
                # entries are pre-rendered so we don't double-escape.
                rendered = inline(body_lines[0]) + ''.join(body_lines[1:])
                items.append(ListItem(
                    Paragraph(rendered, LIST_ITEM), leftIndent=14))
            out.append(ListFlowable(
                items, bulletType='1', start=start_num, leftIndent=18,
                bulletFontName=FONT_REG, bulletFontSize=10,
                spaceBefore=2, spaceAfter=4))
            continue

        # Bullet list (- or *).  Allow indented bullets too — any
        # indented bullets that belonged to a numbered list were already
        # consumed inside the numbered-list handler above, so by the time
        # we reach here an indented bullet is a top-level orphan that we
        # should still render rather than loop forever on.
        if re.match(r'^\s*[-*]\s+', line):
            items = []
            while i < n and re.match(r'^\s*[-*]\s+', lines[i]):
                item_text = re.sub(r'^\s*[-*]\s+', '',
                                   lines[i]).rstrip()
                items.append(ListItem(P(item_text, LIST_ITEM),
                                      leftIndent=14))
                i += 1
            out.append(ListFlowable(
                items, bulletType='bullet', leftIndent=18,
                bulletFontName=FONT_REG, bulletFontSize=10,
                spaceBefore=2, spaceAfter=4))
            continue

        # Paragraph: collect adjacent non-blank, non-block lines
        para_buf: list[str] = []
        while i < n:
            ln = lines[i]
            ls = ln.strip()
            if (not ls or ls.startswith('#')
                    or ls.startswith('>')
                    or _table_row_re.match(ln)
                    or re.match(r'^\s*\d+\.\s+', ln)
                    or re.match(r'^\s*[-*]\s+', ln)
                    or ls in ('---', '***', '___')):
                break
            para_buf.append(ln)
            i += 1
        flush_paragraph(para_buf)

    return out


# ----------------------------------------------------------------------
# Document
# ----------------------------------------------------------------------
def _page_decoration(canvas, doc):
    """Footer with page number."""
    canvas.saveState()
    canvas.setFont(FONT_REG, 8)
    canvas.setFillColor(colors.HexColor('#888'))
    canvas.drawRightString(
        A4[0] - 18 * mm, 12 * mm,
        f'TA Analyzer Manual   -   {doc.page}')
    canvas.restoreState()


def render(md_path: Path, pdf_path: Path):
    text = md_path.read_text(encoding='utf-8')
    flowables = md_to_flowables(text)
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title=md_path.stem, author='TA Analyzer')
    doc.build(flowables, onFirstPage=_page_decoration,
              onLaterPages=_page_decoration)
    print(f'Built {pdf_path}  ({pdf_path.stat().st_size:,} bytes)')


if __name__ == '__main__':
    here = Path(__file__).parent
    for stem in ('Manual_Quick', 'Manual_Detailed',
                 'Manual_Quick_EN', 'Manual_Detailed_EN'):
        md = here / f'{stem}.md'
        pdf = here / f'{stem}.pdf'
        if not md.exists():
            print(f'SKIP: {md} not found')
            continue
        render(md, pdf)
