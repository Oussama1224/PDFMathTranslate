"""Element classification and free-space computation.

Runs after every page has been segmented so document-wide statistics (body
font size, repeated headers/footers) can be used.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Optional

from pt2en.layout.detector import LayoutHints
from pt2en.layout.math import math_ratio
from pt2en.model import (
    DocumentModel,
    ElementKind,
    PageModel,
    Rect,
    TextBlock,
    overlap_ratio,
    rect_area,
    rect_center,
    rect_contains_point,
)

PAGE_NUMBER_RE = re.compile(
    r"^\s*[-–—]?\s*(?:(?:p[áa]g(?:ina)?\.?|p\.)\s*)?(\d{1,4}|[ivxlcdm]{1,7})"
    r"(?:\s*(?:/|de|of)\s*\d{1,4})?\s*[-–—]?\s*$",
    re.IGNORECASE,
)
NUMERIC_PAGE_RE = re.compile(
    r"^\s*[-–—]?\s*(\d{1,4}|[ivxlcdm]{1,7})\s*[-–—]?\s*$", re.I
)
CAPTION_RE = re.compile(
    r"^\s*(figura|fig\.|tabela|tab\.|quadro|gr[áa]fico|esquema|imagem|ilustra[çc][ãa]o"
    r"|diagrama|mapa|fonte|exemplo|anexo|equa[çc][ãa]o)\s*(\d|[ivx]+\b|[:.\-–—])",
    re.IGNORECASE,
)
_LETTER_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")


def text_of(block: TextBlock) -> str:
    return " ".join(
        "".join(r.text for r in line.runs if not r.is_math) for line in block.lines
    ).strip()


def has_translatable_text(block: TextBlock) -> bool:
    return len(_LETTER_RE.findall(text_of(block))) >= 2


def estimate_body_size(doc: DocumentModel) -> float:
    counter: Counter = Counter()
    for block in doc.blocks():
        if block.kind in (ElementKind.FORMULA, ElementKind.LABEL):
            continue
        for line in block.lines:
            for run in line.runs:
                if not run.is_math:
                    counter[round(run.style.size * 2) / 2] += len(run.text.strip())
    if not counter:
        return 10.0
    return float(counter.most_common(1)[0][0])


class ElementClassifier:
    def __init__(self, hints: Optional[dict[int, LayoutHints]] = None):
        self.hints = hints or {}

    def classify(self, doc: DocumentModel) -> None:
        doc.body_size = estimate_body_size(doc)
        self._headers_footers(doc)
        for page in doc.pages:
            for block in page.blocks:
                self._classify_block(page, block, doc.body_size)
        self._heading_levels(doc)
        self._sections(doc)
        for page in doc.pages:
            compute_regions(page)
        if not doc.title:
            headings = [b for b in doc.blocks() if b.kind == ElementKind.HEADING]
            if headings:
                doc.title = text_of(headings[0])[:200]

    # ------------------------------------------------------------------
    def _headers_footers(self, doc: DocumentModel) -> None:
        zone_counts: Counter = Counter()
        entries: list[tuple[TextBlock, tuple]] = []
        for page in doc.pages:
            top_zone = page.height * 0.1
            bottom_zone = page.height * 0.9
            for block in page.blocks:
                if block.kind in (ElementKind.TABLE_CELL, ElementKind.FORMULA):
                    continue
                if len(block.lines) > 3:
                    continue
                position = None
                if block.bbox[3] <= top_zone:
                    position = "top"
                elif block.bbox[1] >= bottom_zone:
                    position = "bottom"
                if not position:
                    continue
                norm = re.sub(r"\d+", "#", text_of(block).lower()).strip()
                key = (position, norm, round(block.bbox[1] / 12))
                zone_counts[key] += 1
                entries.append((block, key))
        threshold = max(2, int(0.3 * doc.page_count))
        for block, key in entries:
            text = text_of(block) or block.text
            if PAGE_NUMBER_RE.match(text):
                block.kind = ElementKind.PAGE_NUMBER
                if NUMERIC_PAGE_RE.match(text):
                    block.translatable = False
                continue
            if zone_counts[key] >= threshold or doc.page_count == 1:
                block.kind = (
                    ElementKind.HEADER if key[0] == "top" else ElementKind.FOOTER
                )

    def _classify_block(self, page: PageModel, block: TextBlock, body: float) -> None:
        text = text_of(block)
        if block.kind == ElementKind.FORMULA:
            block.translatable = False
            return
        if not has_translatable_text(block):
            block.translatable = False
        if block.kind in (
            ElementKind.HEADER,
            ElementKind.FOOTER,
            ElementKind.PAGE_NUMBER,
        ):
            return

        hints = self.hints.get(page.number)
        if hints:
            label = hints.label_for(block.bbox)
            if (
                label == "isolate_formula"
                and math_ratio(r for ln in block.lines for r in ln.runs) > 0.3
            ):
                block.kind = ElementKind.FORMULA
                block.translatable = False
                return
            if label == "title" and block.kind == ElementKind.PARAGRAPH:
                block.kind = ElementKind.HEADING
            if (
                label in ("figure",)
                and block.kind == ElementKind.PARAGRAPH
                and len(text.split()) <= 15
            ):
                block.kind = ElementKind.LABEL
            if label in ("figure_caption", "table_caption", "formula_caption"):
                block.kind = ElementKind.CAPTION

        if block.kind == ElementKind.TABLE_CELL:
            return
        words = len(text.split())
        if block.kind == ElementKind.PARAGRAPH and CAPTION_RE.match(text):
            block.kind = ElementKind.CAPTION
            return
        in_figure = any(
            rect_contains_point(_figure_zone(f), *rect_center(block.bbox))
            and rect_area(f) < page.width * page.height * 0.9
            for f in page.figure_rects + [img.bbox for img in page.images]
        )
        if (
            in_figure
            and words <= 15
            and len(block.lines) <= 3
            and block.kind != ElementKind.LIST_ITEM
        ):
            block.kind = ElementKind.LABEL
            return
        if block.kind in (ElementKind.PARAGRAPH, ElementKind.LIST_ITEM):
            size = block.style.size
            if (
                block.bbox[1] > page.height * 0.7
                and size <= body * 0.88
                and re.match(
                    r"^\s*(\d+|\*|†|[¹²³⁴⁵⁶⁷⁸⁹])", block.prefix_text + block.text
                )
            ):
                block.kind = ElementKind.FOOTNOTE
                return
        numbered = bool(block.prefix_text) and block.prefix_text[0].isdigit()
        if (
            (
                block.kind == ElementKind.PARAGRAPH
                or (block.kind == ElementKind.LIST_ITEM and numbered)
            )
            and len(block.lines) <= 3
            and words <= 25
        ):
            bold = block.style.bold
            if block.style.size >= body * 1.15 or (
                bold
                and words <= 15
                and len(block.lines) <= 2
                and not text.rstrip().endswith(".")
            ):
                block.kind = ElementKind.HEADING

    def _heading_levels(self, doc: DocumentModel) -> None:
        sizes = sorted(
            {
                round(b.style.size)
                for b in doc.blocks()
                if b.kind == ElementKind.HEADING
            },
            reverse=True,
        )
        for block in doc.blocks():
            if block.kind == ElementKind.HEADING:
                block.heading_level = sizes.index(round(block.style.size)) + 1

    def _sections(self, doc: DocumentModel) -> None:
        current = ""
        for block in doc.blocks():
            if block.kind == ElementKind.HEADING:
                current = (block.prefix_text + " " + text_of(block)).strip()[:160]
            block.section = current


def compute_regions(page: PageModel) -> None:
    """Determine alignment anchors and how much room each block may use."""
    blocks = page.blocks
    obstacles: list[tuple[str, Rect]] = [(b.id, b.bbox) for b in blocks]
    obstacles += [
        (img.id, img.bbox)
        for img in page.images
        if rect_area(img.bbox) < page.width * page.height * 0.6
    ]
    # Tables and figures (their borders and shapes) are obstacles too.
    obstacles += [(f"table{i}", r) for i, r in enumerate(page.table_rects)]
    obstacles += [(f"figure{i}", r) for i, r in enumerate(page.figure_rects)]
    content = [
        b.bbox for b in blocks if b.kind not in (ElementKind.HEADER, ElementKind.FOOTER)
    ]
    content += [
        img.bbox
        for img in page.images
        if rect_area(img.bbox) < page.width * page.height * 0.6
    ]
    if content:
        content_left = min(r[0] for r in content)
        content_right = max(r[2] for r in content)
    else:
        content_left, content_right = 36.0, page.width - 36.0
    left_limit = max(8.0, min(content_left, page.width - content_right))
    right_limit = min(page.width - 8.0, max(content_right, page.width - content_left))
    bottom_limit = page.height - 12.0

    for block in blocks:
        x0, y0, x1, y1 = block.bbox
        size = max(block.style.size, 4.0)
        if len(block.lines) == 1 and not block.rotation:
            block.align = _single_line_alignment(
                page, block, content_left, content_right
            )
        elif (
            block.align == "justify"
            and len(block.lines) == 2
            and x1 < content_right - 2 * size
        ):
            block.align = "left"
        if block.region is not None:
            continue
        others = [
            r
            for oid, r in obstacles
            if oid != block.id and not _contains(r, block.bbox)
        ]

        # Downward room: nearest obstacle below that overlaps horizontally.
        max_y = bottom_limit
        for r in others:
            if r[1] >= y1 - 0.25 * size and r[0] < x1 - 0.5 and r[2] > x0 + 0.5:
                max_y = min(max_y, r[1] - 0.2 * size)
        max_y = max(max_y, y1)

        # Horizontal room for single-line blocks (headings, labels...).
        rx0, rx1 = x0, x1
        if len(block.lines) == 1:
            band = (y0, max(y1, y0 + size))
            lim_left, lim_right = left_limit, right_limit
            for r in others:
                if r[3] <= band[0] + 0.5 or r[1] >= band[1] - 0.5:
                    continue
                if r[2] <= x0 + 0.5:
                    lim_left = max(lim_left, r[2] + 0.5 * size)
                elif r[0] >= x1 - 0.5:
                    lim_right = min(lim_right, r[0] - 0.5 * size)
            rx0, rx1 = min(x0, lim_left), max(x1, lim_right)
            if block.kind == ElementKind.LABEL:
                width = max(x1 - x0, 4 * size)
                rx0 = max(rx0, x0 - width * 0.6)
                rx1 = min(rx1, x1 + width * 0.6)
        block.region = (min(rx0, x0), y0, max(rx1, x1), max_y)


def _single_line_alignment(
    page: PageModel, block: TextBlock, left: float, right: float
) -> str:
    x0, y0, x1, y1 = block.bbox
    size = max(block.style.size, 4.0)
    if block.kind == ElementKind.TABLE_CELL and block.region is not None:
        cx0, _, cx1, _ = block.region
        left_gap, right_gap = x0 - cx0, cx1 - x1
        if abs(left_gap - right_gap) <= max(2.0, 0.15 * size) and left_gap > 2.0:
            return "center"
        if right_gap < left_gap * 0.5 and left_gap > 3 * size:
            return "right"
        return "left"
    if block.kind == ElementKind.LABEL:
        for d in page.drawing_rects:
            w, h = d[2] - d[0], d[3] - d[1]
            if w > 3 * size or h > 3 * size:
                continue
            vertical = min(d[3], y1) - max(d[1], y0)
            if vertical > 0 and 0 <= x0 - d[2] <= 1.5 * size:
                return "left"
        return "center"
    width = right - left
    cx = (x0 + x1) / 2
    if (
        width > 0
        and abs(cx - (left + right) / 2) < 0.03 * width
        and x0 - left > 3 * size
    ):
        return "center"
    if abs(right - x1) < 2.0 and (x0 - left) > 0.3 * width:
        return "right"
    return "left"


def _figure_zone(rect: Rect) -> Rect:
    """Figures own the labels placed just around them (axis titles, ticks)."""
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    m = min(28.0, 0.12 * max(w, h))
    return (rect[0] - m, rect[1] - m, rect[2] + m, rect[3] + m)


def _contains(outer: Rect, inner: Rect) -> bool:
    return (
        overlap_ratio(outer, inner) > 0.95 and rect_area(outer) > rect_area(inner) * 1.5
    )


def group_by_page(blocks) -> dict[int, list[TextBlock]]:
    out: dict[int, list[TextBlock]] = defaultdict(list)
    for block in blocks:
        out[block.page].append(block)
    return out
