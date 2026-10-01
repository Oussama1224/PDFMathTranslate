"""Low-level PDF parsing with PyMuPDF.

Produces, per page, the raw material the layout stage works with: text lines
(with per-glyph geometry and style), image placements, vector drawing
statistics, figure clusters and table cells.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pymupdf

from pt2en.layout.math import build_runs
from pt2en.model import Glyph, ImageElement, Line, Rect, TextStyle, rect_area

log = logging.getLogger(__name__)

if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()

_SUBSET_RE = re.compile(r"^[A-Z]{6}\+")
_BOLD_RE = re.compile(
    r"(bold|black|heavy|semibold|demi|extrabold|ultrabold|\bbd\b)", re.I
)
_ITALIC_RE = re.compile(r"(italic|oblique|\bit\b|slanted|kursiv)", re.I)
_MONO_RE = re.compile(
    r"(mono|courier|consolas|menlo|code|typewriter|cmtt|lmmono)", re.I
)
_SERIF_FALLBACK_RE = re.compile(
    r"(times|georgia|garamond|cambria|book|serif|cmr|lmroman|palatino|minion)", re.I
)
_SANS_RE = re.compile(
    r"(arial|helvetica|calibri|verdana|tahoma|segoe|sans|gothic|futura|roboto|open ?sans|lato|carlito)",
    re.I,
)

CHAR_STRIKEOUT = 1
CHAR_UNDERLINE = 2
CHAR_BOLD = 8
CHAR_FILLED = 16
CHAR_STROKED = 32

TEXT_FLAGS = (
    pymupdf.TEXT_PRESERVE_WHITESPACE
    | pymupdf.TEXT_MEDIABOX_CLIP
    | getattr(pymupdf, "TEXT_COLLECT_STYLES", 0)
)


@dataclass
class RawLine:
    line: Line
    block_no: int
    line_no: int


@dataclass
class RawPage:
    number: int
    width: float
    height: float
    lines: list[RawLine] = field(default_factory=list)
    images: list[ImageElement] = field(default_factory=list)
    drawings_count: int = 0
    drawing_rects: list[Rect] = field(default_factory=list)
    figure_rects: list[Rect] = field(default_factory=list)
    table_cells: list[list[Rect]] = field(default_factory=list)
    table_rects: list[Rect] = field(default_factory=list)
    text_chars: int = 0
    invisible_chars: int = 0
    image_coverage: float = 0.0


def clean_font_name(name: str) -> str:
    return _SUBSET_RE.sub("", name or "")


def style_from_span(span: dict, char_flags: int | None = None) -> TextStyle:
    font = clean_font_name(span.get("font", ""))
    flags = span.get("flags", 0)
    cflags = span.get("char_flags", 0) if char_flags is None else char_flags
    bold = bool(flags & 16) or bool(cflags & CHAR_BOLD) or bool(_BOLD_RE.search(font))
    italic = bool(flags & 2) or bool(_ITALIC_RE.search(font))
    mono = bool(flags & 8) or bool(_MONO_RE.search(font))
    serif = bool(flags & 4)
    if not serif and not _SANS_RE.search(font) and _SERIF_FALLBACK_RE.search(font):
        serif = True
    if _SANS_RE.search(font):
        serif = False
    alpha = span.get("alpha", 255)
    return TextStyle(
        font=font,
        size=float(span.get("size", 10.0)),
        bold=bold,
        italic=italic,
        serif=serif,
        mono=mono,
        color=int(span.get("color", 0) or 0),
        alpha=(alpha / 255.0) if isinstance(alpha, int) else 1.0,
        superscript=bool(flags & 1),
        underline=bool(cflags & CHAR_UNDERLINE),
        strike=bool(cflags & CHAR_STRIKEOUT),
    )


def _char_visible(span: dict) -> bool:
    if "char_flags" in span:
        cf = span["char_flags"]
        if not (cf & (CHAR_FILLED | CHAR_STROKED)):
            return False
    if span.get("alpha", 255) == 0:
        return False
    return True


def parse_page(
    doc: pymupdf.Document, page: pymupdf.Page, *, detect_tables: bool = True
) -> RawPage:
    raw = RawPage(number=page.number, width=page.rect.width, height=page.rect.height)
    data = page.get_text("rawdict", flags=TEXT_FLAGS)

    for block_no, block in enumerate(data.get("blocks", [])):
        if block.get("type", 0) != 0:
            continue
        for line_no, ldata in enumerate(block.get("lines", [])):
            glyphs: list[Glyph] = []
            styles: list[TextStyle] = []
            for span in ldata.get("spans", []):
                chars = span.get("chars", [])
                if not chars:
                    continue
                if not _char_visible(span):
                    raw.invisible_chars += len(chars)
                    continue
                style = style_from_span(span)
                for ch in chars:
                    c = ch.get("c", "")
                    if not c:
                        continue
                    glyphs.append(
                        Glyph(
                            c=c,
                            bbox=tuple(ch["bbox"]),
                            origin=tuple(ch["origin"]),
                            size=style.size,
                        )
                    )
                    styles.append(style)
            if not glyphs or not any(not g.c.isspace() for g in glyphs):
                continue
            direction = tuple(ldata.get("dir", (1.0, 0.0)))
            horizontal = abs(direction[0] - 1.0) < 1e-3
            pieces = [(glyphs, styles)]
            if horizontal:
                pieces = [
                    _insert_gap_spaces(g, s) for g, s in _split_tab_gaps(glyphs, styles)
                ]
            for sub, (glyphs, styles) in enumerate(pieces):
                # Trim leading/trailing whitespace glyphs (they confuse geometry).
                while glyphs and glyphs[0].c.isspace():
                    glyphs.pop(0)
                    styles.pop(0)
                while glyphs and glyphs[-1].c.isspace():
                    glyphs.pop()
                    styles.pop()
                if not glyphs:
                    continue
                raw.text_chars += sum(1 for g in glyphs if not g.c.isspace())
                bbox = (
                    min(g.bbox[0] for g in glyphs),
                    min(g.bbox[1] for g in glyphs),
                    max(g.bbox[2] for g in glyphs),
                    max(g.bbox[3] for g in glyphs),
                )
                sizes = sorted(s.size for s in styles)
                line = Line(
                    runs=build_runs(glyphs, styles),
                    bbox=bbox,
                    direction=direction,
                    wmode=ldata.get("wmode", 0),
                    baseline=_dominant_baseline(glyphs, styles),
                    size=sizes[len(sizes) // 2],
                )
                raw.lines.append(
                    RawLine(line=line, block_no=block_no, line_no=line_no * 100 + sub)
                )

    # ----------------------------------------------------------- images
    page_area = max(rect_area(tuple(page.rect)), 1.0)
    covered = 0.0
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:  # pragma: no cover
        infos = []
    for idx, info in enumerate(infos):
        bbox = tuple(info.get("bbox", (0, 0, 0, 0)))
        clipped = (
            max(bbox[0], 0),
            max(bbox[1], 0),
            min(bbox[2], raw.width),
            min(bbox[3], raw.height),
        )
        area = rect_area(clipped)
        if area <= 0:
            continue
        covered += area
        xref = int(info.get("xref", 0) or 0)
        smask = 0
        if xref:
            try:
                key = doc.xref_get_key(xref, "SMask")
                if key[0] == "xref":
                    smask = int(key[1].split()[0])
            except Exception:
                smask = 0
        raw.images.append(
            ImageElement(
                id=f"p{page.number}-img{idx}",
                page=page.number,
                xref=xref,
                bbox=clipped,
                width=int(info.get("width", 0)),
                height=int(info.get("height", 0)),
                smask=smask,
                transform=tuple(info.get("transform", (1, 0, 0, 1, 0, 0))),
            )
        )
    raw.image_coverage = min(1.0, covered / page_area)

    # --------------------------------------------------------- drawings
    try:
        drawings = page.get_drawings()
    except Exception:  # pragma: no cover
        drawings = []
    raw.drawings_count = len(drawings)
    raw.drawing_rects = [
        tuple(d["rect"]) for d in drawings if d.get("rect") is not None
    ]
    if drawings:
        try:
            clusters = page.cluster_drawings(drawings=drawings)
        except Exception:  # pragma: no cover
            clusters = []
        for rect in clusters:
            r = tuple(rect)
            w, h = r[2] - r[0], r[3] - r[1]
            if w < 30 or h < 20:
                continue
            if w > raw.width * 0.97 and h > raw.height * 0.97:
                continue  # page background
            inside = sum(
                1
                for d in raw.drawing_rects
                if d[0] >= r[0] - 1
                and d[1] >= r[1] - 1
                and d[2] <= r[2] + 1
                and d[3] <= r[3] + 1
            )
            if inside >= 4:
                raw.figure_rects.append(r)

    # ----------------------------------------------------------- tables
    if detect_tables:
        try:
            tables = page.find_tables()
            for table in tables.tables:
                cells = [tuple(c) for c in table.cells if c is not None]
                if len(cells) < 2:
                    continue
                raw.table_cells.append(cells)
                raw.table_rects.append(tuple(table.bbox))
        except Exception as exc:  # pragma: no cover - table finder is heuristic
            log.debug("table detection failed on page %s: %s", page.number, exc)

    # Figures that are really table grids are not figures.
    raw.figure_rects = [
        f
        for f in raw.figure_rects
        if not any(
            _mostly_inside(f, t) or _mostly_inside(t, f) for t in raw.table_rects
        )
    ]
    return raw


def _split_tab_gaps(glyphs: list[Glyph], styles: list[TextStyle]):
    """Split a line where text is separated by tab-stop sized gaps (columns, table cells)."""
    pieces = []
    start = 0
    last_ink = None
    for i, g in enumerate(glyphs):
        if g.c.isspace():
            continue
        if last_ink is not None:
            prev = glyphs[last_ink]
            gap = g.bbox[0] - prev.bbox[2]
            if gap > 2.2 * max(g.size, prev.size):
                pieces.append((glyphs[start:i], styles[start:i]))
                start = i
        last_ink = i
    pieces.append((glyphs[start:], styles[start:]))
    return [(list(g), list(s)) for g, s in pieces if g]


def _insert_gap_spaces(glyphs: list[Glyph], styles: list[TextStyle]):
    """Insert synthetic spaces where words are separated only by positioning."""
    out_g: list[Glyph] = []
    out_s: list[TextStyle] = []
    for g, s in zip(glyphs, styles):
        if out_g and not g.c.isspace() and not out_g[-1].c.isspace():
            prev = out_g[-1]
            gap = g.bbox[0] - prev.bbox[2]
            if gap > 0.22 * max(s.size, out_s[-1].size):
                out_g.append(
                    Glyph(
                        c=" ",
                        bbox=(prev.bbox[2], prev.bbox[1], g.bbox[0], prev.bbox[3]),
                        origin=(prev.bbox[2], prev.origin[1]),
                        size=prev.size,
                    )
                )
                out_s.append(out_s[-1])
        out_g.append(g)
        out_s.append(s)
    return out_g, out_s


def _mostly_inside(a: Rect, b: Rect) -> bool:
    inter = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    area = rect_area(a)
    return area > 0 and rect_area(inter) / area > 0.85


def _dominant_baseline(glyphs: list[Glyph], styles: list[TextStyle]) -> float:
    """Baseline of the largest-sized, non-superscript glyphs of a line."""
    candidates = [
        (s.size, g.origin[1])
        for g, s in zip(glyphs, styles)
        if not s.superscript and not g.c.isspace()
    ]
    if not candidates:
        candidates = [(s.size, g.origin[1]) for g, s in zip(glyphs, styles)]
    max_size = max(c[0] for c in candidates)
    ys = sorted(y for size, y in candidates if size >= max_size * 0.9)
    return ys[len(ys) // 2]
