"""Paragraph segmentation: turns raw text lines into structured text blocks."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from pt2en.layout.math import BULLET_CHARS, is_formula_line
from pt2en.model import (
    ElementKind,
    Glyph,
    Line,
    MathRegion,
    Rect,
    Run,
    TextBlock,
    TextStyle,
    expand_rect,
    intersection_area,
    rect_area,
    rect_contains_point,
    union_rects,
)
from pt2en.parsing.pdf_parser import RawLine, RawPage

ENUM_PREFIX_RE = re.compile(
    r"^(\(?\d{1,3}[.)º°]|\(?[a-zA-Z]\)|\(?[ivxIVX]{1,5}[.)]|\d+(?:\.\d+)+\.?|[A-Z]\.)(?=\s)"
)


def _line_text(line: Line) -> str:
    return line.text


def _all_bold(line: Line) -> bool:
    text_runs = [r for r in line.runs if not r.is_math and r.text.strip()]
    return bool(text_runs) and all(r.style.bold for r in text_runs)


def _word_count(line: Line) -> int:
    return len(line.text.split())


FOOTNOTE_MARKS = set("¹²³⁴⁵⁶⁷⁸⁹*†‡§")


def _starts_list_item(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    if stripped[0] in BULLET_CHARS and (len(stripped) == 1 or stripped[1].isspace()):
        return True
    if 0xE000 <= ord(stripped[0]) <= 0xF8FF:
        return True
    return bool(ENUM_PREFIX_RE.match(stripped))


def _vertical_overlap(a: Rect, b: Rect) -> float:
    inter = min(a[3], b[3]) - max(a[1], b[1])
    h = min(a[3] - a[1], b[3] - b[1])
    return inter / h if h > 0 else 0.0


def _cell_index(line: Line, cells: list[tuple[int, Rect]]) -> Optional[int]:
    cx = (line.bbox[0] + line.bbox[2]) / 2
    cy = (line.bbox[1] + line.bbox[3]) / 2
    for idx, rect in cells:
        if rect_contains_point(rect, cx, cy, tol=0.5):
            return idx
    return None


class BlockBuilder:
    """Builds :class:`TextBlock` objects for one page."""

    def __init__(self, raw: RawPage):
        self.raw = raw
        self.cells: list[tuple[int, Rect]] = []
        idx = 0
        for table in raw.table_cells:
            for cell in table:
                self.cells.append((idx, cell))
                idx += 1
        self.cell_rects = dict(self.cells)

    # ------------------------------------------------------------ grouping
    def build(self) -> list[TextBlock]:
        groups: list[list[Line]] = []
        group_cells: list[Optional[int]] = []
        rotated: list[Line] = []

        current_block = None
        current: list[RawLine] = []
        ordered = list(self.raw.lines)
        for rl in ordered:
            if not rl.line.is_horizontal or rl.line.wmode:
                rotated.append(rl.line)
                continue
            if current and rl.block_no != current_block:
                self._split_block([r.line for r in current], groups, group_cells)
                current = []
            current_block = rl.block_no
            current.append(rl)
        if current:
            self._split_block([r.line for r in current], groups, group_cells)

        groups, group_cells = self._merge_continuations(groups, group_cells)

        blocks: list[TextBlock] = []
        for lines, cell in zip(groups, group_cells):
            block = self._make_block(lines, len(blocks))
            if cell is not None:
                block.kind = ElementKind.TABLE_CELL
                block.region = expand_rect(self.cell_rects[cell], -1.0)
            blocks.append(block)
        for line in rotated:
            block = self._make_block([line], len(blocks))
            dx, dy = line.direction
            block.rotation = math.degrees(math.atan2(dy, dx))
            block.kind = ElementKind.LABEL
            blocks.append(block)
        return blocks

    def _split_block(self, lines: list[Line], groups, group_cells) -> None:
        if not lines:
            return
        lines = _merge_same_row(lines, lambda ln: _cell_index(ln, self.cells))
        x0 = min(line.bbox[0] for line in lines)
        x1 = max(line.bbox[2] for line in lines)
        formula_flags = [is_formula_line(line.runs) for line in lines]

        # A formula line that overlaps a neighbouring text line vertically is an
        # inline construct (e.g. a fraction inside a sentence), not a display.
        display = []
        for i, line in enumerate(lines):
            if not formula_flags[i]:
                display.append(False)
                continue
            inline = False
            for j in (i - 1, i + 1):
                if 0 <= j < len(lines) and not formula_flags[j]:
                    if _vertical_overlap(line.bbox, lines[j].bbox) > 0.3:
                        inline = True
            display.append(not inline)

        current: list[Line] = [lines[0]]
        current_cell = _cell_index(lines[0], self.cells)
        for i in range(1, len(lines)):
            prev, cur = lines[i - 1], lines[i]
            cell = _cell_index(cur, self.cells)
            brk = (
                cell != current_cell
                or (self._in_figure(cur) and not cur.text.lstrip()[:1].islower())
                or display[i] != display[i - 1]
                or (
                    display[i]
                    and display[i - 1]
                    and cur.bbox[1] - prev.bbox[3] > 0.5 * cur.size
                )
                or self._should_break(prev, cur, x0, x1)
            )
            if brk:
                groups.append(current)
                group_cells.append(current_cell)
                current = [cur]
                current_cell = cell
            else:
                current.append(cur)
        groups.append(current)
        group_cells.append(current_cell)

    def _in_figure(self, line: Line) -> bool:
        """Lines inside charts/diagrams are separate labels unless clearly wrapped."""
        cx = (line.bbox[0] + line.bbox[2]) / 2
        cy = (line.bbox[1] + line.bbox[3]) / 2
        zones = self.raw.figure_rects + [img.bbox for img in self.raw.images]
        page_area = self.raw.width * self.raw.height
        return any(
            rect_contains_point(z, cx, cy)
            and (z[2] - z[0]) * (z[3] - z[1]) < 0.85 * page_area
            for z in zones
        )

    @staticmethod
    def _should_break(prev: Line, cur: Line, gx0: float, gx1: float) -> bool:
        size = max(prev.size, cur.size, 1.0)
        if abs(cur.baseline - prev.baseline) < 0.35 * size:
            # Same visual row: separate when there is a clear horizontal gap.
            return (
                cur.bbox[0] - prev.bbox[2] > 1.2 * size or cur.bbox[2] <= prev.bbox[0]
            )
        if _vertical_overlap(prev.bbox, cur.bbox) > 0.3:
            return False  # sub/superscript or inline fraction pieces
        gap = cur.bbox[1] - prev.bbox[3]
        if gap > 0.9 * size:
            return True
        if abs(cur.size - prev.size) > 0.18 * size:
            return True
        if (
            _all_bold(prev) != _all_bold(cur)
            and min(_word_count(prev), _word_count(cur)) <= 14
        ):
            return True
        if _starts_list_item(cur.text):
            return True
        width = gx1 - gx0
        if width > 0:
            prev_fill = (prev.bbox[2] - gx0) / width
            first = cur.text.lstrip()[:1]
            if prev_fill < 0.7 and (first.isupper() or first.isdigit()):
                return True
            if cur.bbox[0] - prev.bbox[0] > 1.0 * size and prev_fill < 0.92:
                return True
        return False

    def _merge_continuations(self, groups, cells):
        if not groups:
            return groups, cells
        out_g = [groups[0]]
        out_c = [cells[0]]
        for lines, cell in zip(groups[1:], cells[1:]):
            prev = out_g[-1]
            if self._is_continuation(prev, lines, out_c[-1], cell):
                prev.extend(lines)
            else:
                out_g.append(lines)
                out_c.append(cell)
        return out_g, out_c

    @staticmethod
    def _is_continuation(a: list[Line], b: list[Line], ca, cb) -> bool:
        if ca != cb:
            return False
        if any(is_formula_line(line.runs) for line in (a[-1], b[0])):
            return False
        last, first = a[-1], b[0]
        size = max(last.size, first.size, 1.0)
        if abs(last.size - first.size) > 0.1 * size:
            return False
        gap = first.bbox[1] - last.bbox[3]
        if gap < -0.2 * size or gap > 0.6 * size:
            return False
        if _all_bold(last) != _all_bold(first):
            return False
        ax0 = min(line.bbox[0] for line in a)
        ax1 = max(line.bbox[2] for line in a)
        bx0 = min(line.bbox[0] for line in b)
        bx1 = max(line.bbox[2] for line in b)
        overlap = min(ax1, bx1) - max(ax0, bx0)
        narrow = min(ax1 - ax0, bx1 - bx0)
        if narrow <= 0 or overlap / narrow < 0.7 or abs(ax0 - bx0) > 1.5 * size:
            return False
        if _starts_list_item(first.text):
            return False
        if last.bbox[2] < max(ax1, bx1) - 2.5 * size:
            return False  # previous paragraph ended with a short line
        text_end = last.text.rstrip()[-1:]
        first_char = first.text.lstrip()[:1]
        return first_char.islower() or text_end not in ".:!?"

    # ---------------------------------------------------------- block data
    def _make_block(self, lines: list[Line], index: int) -> TextBlock:
        page = self.raw.number
        bbox = union_rects(line.bbox for line in lines)
        style = dominant_style(lines)
        block = TextBlock(
            id=f"p{page}-b{index}",
            page=page,
            kind=ElementKind.PARAGRAPH,
            bbox=bbox,
            lines=lines,
            style=style,
        )
        if all(is_formula_line(line.runs) for line in lines):
            block.kind = ElementKind.FORMULA
            block.translatable = False
        self._detect_prefix(block)
        self._measure(block)
        self._math_regions(block)
        return block

    @staticmethod
    def _detect_prefix(block: TextBlock) -> None:
        first = block.lines[0]
        glyphs = first.glyphs
        text = "".join(g.c for g in glyphs)
        stripped = text.lstrip()
        lead = len(text) - len(stripped)
        prefix_len = 0
        if stripped and (
            (
                stripped[0] in BULLET_CHARS
                and (len(stripped) == 1 or stripped[1].isspace())
            )
            or 0xE000 <= ord(stripped[0]) <= 0xF8FF
        ):
            prefix_len = 1
        elif stripped and stripped[0] in FOOTNOTE_MARKS:
            prefix_len = 1
            while prefix_len < len(stripped) and stripped[prefix_len] in FOOTNOTE_MARKS:
                prefix_len += 1
        else:
            m = ENUM_PREFIX_RE.match(stripped)
            if m:
                prefix_len = len(m.group(1))
        if not prefix_len:
            return
        k = lead + prefix_len
        rest = glyphs[k:]
        while rest and rest[0].c.isspace():
            rest = rest[1:]
        if not rest:
            return  # the line is only a prefix (e.g. a lone number)
        prefix_glyphs = glyphs[lead : lead + prefix_len]
        block.prefix_text = "".join(g.c for g in prefix_glyphs)
        block.prefix_bbox = union_rects(g.bbox for g in prefix_glyphs)
        block.text_x0 = rest[0].bbox[0]
        # Remove prefix glyphs from the runs: they stay untouched on the page.
        prefix_ids = {id(g) for g in prefix_glyphs}
        new_runs: list[Run] = []
        for run in first.runs:
            kept = [g for g in run.glyphs if id(g) not in prefix_ids]
            if kept:
                new_runs.append(Run(glyphs=kept, style=run.style, is_math=run.is_math))
        # Drop whitespace-only leading runs.
        while new_runs and not new_runs[0].text.strip():
            new_runs.pop(0)
        first.runs = new_runs
        first_char = block.prefix_text[:1]
        if (
            block.kind == ElementKind.PARAGRAPH
            and first_char
            and first_char not in FOOTNOTE_MARKS
            and (
                first_char in BULLET_CHARS
                or 0xE000 <= ord(first_char) <= 0xF8FF
                or ENUM_PREFIX_RE.match(block.prefix_text + " ")
            )
        ):
            block.kind = ElementKind.LIST_ITEM

    @staticmethod
    def _measure(block: TextBlock) -> None:
        lines = block.lines
        x0 = block.text_x0 if block.text_x0 is not None else block.bbox[0]
        block_x0 = min([x0] + [line.bbox[0] for line in lines[1:]])
        block.first_indent = x0 - block_x0
        if len(lines) > 1:
            block.rest_indent = min(line.bbox[0] for line in lines[1:]) - block_x0
            baselines = [line.baseline for line in lines]
            diffs = sorted(
                b - a
                for a, b in zip(baselines, baselines[1:])
                if b - a > 0.5 * lines[0].size
            )
            block.line_pitch = diffs[len(diffs) // 2] if diffs else 0.0
        block.align = detect_alignment(lines, block.bbox)

    def _with_math_art(self, bbox: Rect, size: float) -> Rect:
        """Extend a formula box with its vector parts (fraction bars, overlines, roots)."""
        probe = expand_rect(bbox, 0.25 * size, 0.25 * size)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        extra = [
            d
            for d in self.raw.drawing_rects
            if _touches(d, probe)
            and (d[2] - d[0]) <= w * 1.5 + 2
            and (d[3] - d[1]) <= h * 1.5 + 2
        ]
        return union_rects([bbox] + extra) if extra else bbox

    def _math_regions(self, block: TextBlock) -> None:
        if block.kind == ElementKind.FORMULA:
            return
        items: list[tuple[int, Run]] = []
        for li, line in enumerate(block.lines):
            for run in line.runs:
                if run.is_math and run.text.strip():
                    items.append((li, run))
        if not items:
            return
        parent = list(range(len(items)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                li, ri = items[i]
                lj, rj = items[j]
                if li == lj:
                    continue
                size = max(ri.style.size, rj.style.size)
                a = expand_rect(ri.bbox, 0.15 * size, 0.05 * size)
                if intersection_area(a, rj.bbox) > 0:
                    parent[find(i)] = find(j)
        groups: dict[int, list[tuple[int, Run]]] = {}
        for i, item in enumerate(items):
            groups.setdefault(find(i), []).append(item)
        for n, members in enumerate(
            sorted(groups.values(), key=lambda m: (m[0][0], m[0][1].bbox[0]))
        ):
            rid = f"m{n + 1}"
            bbox = union_rects(run.bbox for _, run in members)
            bbox = self._with_math_art(bbox, max(run.style.size for _, run in members))
            # Baseline: the line whose vertical centre is closest to the region's.
            cy = (bbox[1] + bbox[3]) / 2
            main_line = min(
                (block.lines[li] for li, _ in members),
                key=lambda line: abs((line.bbox[1] + line.bbox[3]) / 2 - cy),
            )
            region = MathRegion(
                id=rid,
                page=block.page,
                bbox=bbox,
                text=" ".join(run.text.strip() for _, run in members),
                baseline=main_line.baseline,
                glyph_count=sum(len(run.glyphs) for _, run in members),
            )
            block.math[rid] = region
            for _, run in members:
                run.region_id = rid


def _touches(a: Rect, b: Rect) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _merge_same_row(lines: list[Line], cell_of=None) -> list[Line]:
    """Join fragments that MuPDF reported as separate lines on one baseline."""
    out: list[Line] = []
    for line in lines:
        if out and (cell_of is None or cell_of(out[-1]) == cell_of(line)):
            prev = out[-1]
            size = max(prev.size, line.size, 1.0)
            gap = line.bbox[0] - prev.bbox[2]
            if (
                abs(line.baseline - prev.baseline) < 0.2 * size
                and -0.1 * size <= gap < 1.2 * size
            ):
                runs = list(prev.runs)
                if gap > 0.15 * size and runs and not runs[-1].text.endswith(" "):
                    last = runs[-1]
                    space = Glyph(
                        c=" ",
                        bbox=(prev.bbox[2], prev.bbox[1], line.bbox[0], prev.bbox[3]),
                        origin=(prev.bbox[2], prev.baseline),
                        size=last.style.size,
                    )
                    runs.append(Run(glyphs=[space], style=last.style, is_math=False))
                runs.extend(line.runs)
                out[-1] = Line(
                    runs=runs,
                    bbox=union_rects([prev.bbox, line.bbox]),
                    direction=prev.direction,
                    wmode=prev.wmode,
                    baseline=prev.baseline if prev.size >= line.size else line.baseline,
                    size=max(prev.size, line.size),
                )
                continue
        out.append(line)
    return out


def dominant_style(lines: list[Line]) -> TextStyle:
    counter: Counter = Counter()
    styles: dict[tuple, TextStyle] = {}
    for line in lines:
        for run in line.runs:
            if run.is_math:
                continue
            n = sum(1 for g in run.glyphs if not g.c.isspace())
            if not n:
                continue
            key = run.style.visual_key() + (round(run.style.size, 1),)
            counter[key] += n
            styles.setdefault(key, run.style)
    if not counter:
        for line in lines:
            for run in line.runs:
                return run.style
        return TextStyle()
    return styles[counter.most_common(1)[0][0]]


def detect_alignment(lines: list[Line], bbox: Rect) -> str:
    if len(lines) == 1:
        return "left"
    size = max(line.size for line in lines)
    tol = max(1.5, 0.25 * size)
    lefts = [line.bbox[0] for line in lines[1:]]
    rights = [line.bbox[2] for line in lines[:-1]]
    centers = [(line.bbox[0] + line.bbox[2]) / 2 for line in lines]
    left_aligned = max(lefts) - min(lefts) <= tol
    right_aligned = max(rights) - min(rights) <= tol
    mean_c = sum(centers) / len(centers)
    centered = all(abs(c - mean_c) <= tol for c in centers)
    widths = [line.bbox[2] - line.bbox[0] for line in lines]
    if centered and (max(widths) - min(widths)) > 2 * tol:
        return "center"
    if left_aligned and right_aligned and len(lines) >= 3:
        return "justify"
    if left_aligned and right_aligned and len(lines) == 2:
        # Two lines: justified only if the first line spans the full width.
        return "justify" if lines[0].bbox[2] >= bbox[2] - tol else "left"
    if right_aligned and not left_aligned and lines[-1].bbox[2] >= bbox[2] - tol:
        return "right"
    return "left"


def block_area(block: TextBlock) -> float:
    return rect_area(block.bbox)
