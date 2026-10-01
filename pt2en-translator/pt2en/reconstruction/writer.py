"""Document reconstruction: replaces Portuguese text in-place.

For each page the writer:

1. removes the glyphs of translated blocks with precise redactions that touch
   only those glyphs (images, vector graphics, formulas and untouched text
   stay byte-identical);
2. re-typesets the English text inside each block's frame;
3. re-inserts inline formulas as vector clips of the *source* page, so every
   symbol, fraction bar and accent is reproduced exactly.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Optional

import pymupdf

from pt2en.model import (
    BlockStatus,
    ElementKind,
    Glyph,
    Issue,
    MathRegion,
    PageModel,
    Rect,
    Severity,
    TextBlock,
    expand_rect,
)
from pt2en.reconstruction.fonts import FontResolver
from pt2en.reconstruction.layout_engine import Frame, MathBox, Typesetter, TypesetResult
from pt2en.translation import markup as mk

log = logging.getLogger(__name__)

MATH_MARGIN = 1.5


def glyph_redaction_rects(glyphs: list[Glyph], rotated: bool = False) -> list[Rect]:
    """Small rectangles that intersect each glyph's ink but not its neighbours'."""
    rects: list[Rect] = []
    last_band: Optional[tuple[float, float]] = None
    for g in glyphs:
        if g.c.isspace():
            continue
        x0, y0, x1, y1 = g.bbox
        w = max(x1 - x0, 0.3 * g.size)
        h = max(y1 - y0, 0.3 * g.size)
        if rotated:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rects.append((cx - 0.22 * w, cy - 0.22 * h, cx + 0.22 * w, cy + 0.22 * h))
            last_band = None
            continue
        base = g.origin[1]
        r = (g.cx - 0.2 * w, base - 0.7 * g.size, g.cx + 0.2 * w, base + 0.12 * g.size)
        band = (round(r[1], 1), round(r[3], 1))
        if rects and last_band == band and r[0] - rects[-1][2] < 0.8 * g.size:
            prev = rects[-1]
            rects[-1] = (prev[0], prev[1], r[2], prev[3])
        else:
            rects.append(r)
        last_band = band
    return rects


def block_text_glyphs(block: TextBlock) -> list[tuple[list[Glyph], bool]]:
    """Glyph groups (non-math runs) of a block, with their rotation flag."""
    groups = []
    for line in block.lines:
        rotated = not line.is_horizontal
        for run in line.runs:
            if not run.is_math:
                groups.append((run.glyphs, rotated))
    return groups


class MathClipSource:
    """Builds isolated vector clips of formula regions from the source PDF."""

    def __init__(self, src_bytes: bytes, pages: list[PageModel]):
        self.src_bytes = src_bytes
        self.pages = {p.number: p for p in pages}
        self.clips = pymupdf.open()
        self._textless: dict[int, pymupdf.Document] = {}
        self._index: dict[tuple, tuple[int, Rect]] = {}

    def _textless_page(self, pno: int) -> pymupdf.Document:
        if pno in self._textless:
            return self._textless[pno]
        doc = pymupdf.open(stream=self.src_bytes, filetype="pdf")
        doc.select([pno])
        page = doc[0]
        pm = self.pages[pno]
        count = 0
        for block in pm.blocks:
            for glyphs, rotated in block_text_glyphs(block):
                for r in glyph_redaction_rects(glyphs, rotated):
                    page.add_redact_annot(r, fill=False)
                    count += 1
            if block.prefix_bbox:
                cx, cy = (block.prefix_bbox[0] + block.prefix_bbox[2]) / 2, (
                    block.prefix_bbox[1] + block.prefix_bbox[3]
                ) / 2
                page.add_redact_annot(
                    (cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5), fill=False
                )
                count += 1
        if count:
            page.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_NONE,
                graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                text=pymupdf.PDF_REDACT_TEXT_REMOVE,
            )
        self._textless[pno] = doc
        return doc

    def clip_for(self, region: MathRegion, block_id: str) -> tuple[int, Rect]:
        key = (region.page, block_id, region.id)
        if key in self._index:
            return self._index[key]
        src = self._textless_page(region.page)
        self.clips.insert_pdf(src)
        idx = self.clips.page_count - 1
        page = self.clips[idx]
        pr = page.rect
        r = expand_rect(region.bbox, MATH_MARGIN)
        outside = [
            (pr.x0, pr.y0, pr.x1, r[1]),
            (pr.x0, r[3], pr.x1, pr.y1),
            (pr.x0, r[1], r[0], r[3]),
            (r[2], r[1], pr.x1, r[3]),
        ]
        for o in outside:
            if o[2] - o[0] > 0 and o[3] - o[1] > 0:
                page.add_redact_annot(o, fill=False)
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_REMOVE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )
        self._index[key] = (idx, r)
        return idx, r

    def close(self) -> None:
        for d in self._textless.values():
            d.close()
        self.clips.close()


class PageWriter:
    def __init__(
        self,
        fonts: FontResolver,
        math_source: MathClipSource,
        *,
        min_scale: float = 0.72,
        hard_min_scale: float = 0.5,
    ):
        self.fonts = fonts
        self.typesetter = Typesetter(fonts)
        self.math_source = math_source
        self.min_scale = min_scale
        self.hard_min_scale = hard_min_scale
        self._writers: dict[tuple, pymupdf.TextWriter] = {}

    # ----------------------------------------------------------------
    def write(self, page: pymupdf.Page, pm: PageModel, issues: list[Issue]) -> None:
        blocks = []
        for block in pm.blocks:
            if block.status != BlockStatus.TRANSLATED or block.translation is None:
                continue
            tokens = self._tokens(block, issues)
            if tokens is None:
                continue
            blocks.append((block, tokens))
        if not blocks:
            return

        # Phase A: formulas that move with the text (glyphs + their vector art).
        # OCR blocks have no glyphs in the PDF: their pixels are cleaned by the scan processor.
        native = [(b, t) for b, t in blocks if b.source != "ocr"]
        count = 0
        for block, _ in native:
            for region in block.math.values():
                page.add_redact_annot(expand_rect(region.bbox, 0.2), fill=False)
                count += 1
        if count:
            page.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_NONE,
                graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                text=pymupdf.PDF_REDACT_TEXT_REMOVE,
            )

        # Phase A2: underline / strike-through decorations (vector only, text untouched).
        count = 0
        for block, _ in native:
            for line in block.lines:
                if not line.is_horizontal:
                    continue
                for run in line.runs:
                    if run.is_math or not (run.style.underline or run.style.strike):
                        continue
                    x0, _, x1, _ = run.bbox
                    size = run.style.size
                    base = line.baseline
                    pad = 0.6 * size
                    if run.style.underline:
                        page.add_redact_annot(
                            (
                                x0 - pad,
                                base - 0.05 * size,
                                x1 + pad,
                                base + 0.45 * size,
                            ),
                            fill=False,
                        )
                        count += 1
                    if run.style.strike:
                        page.add_redact_annot(
                            (x0 - pad, base - 0.6 * size, x1 + pad, base - 0.1 * size),
                            fill=False,
                        )
                        count += 1
        if count:
            page.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_NONE,
                graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                text=getattr(pymupdf, "PDF_REDACT_TEXT_NONE", 1),
            )

        # Phase B: the Portuguese glyphs themselves (graphics untouched).
        count = 0
        for block, _ in native:
            for glyphs, rotated in block_text_glyphs(block):
                for r in glyph_redaction_rects(glyphs, rotated):
                    page.add_redact_annot(r, fill=False)
                    count += 1
        if count:
            page.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_NONE,
                graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                text=pymupdf.PDF_REDACT_TEXT_REMOVE,
            )

        self._writers: dict[tuple, pymupdf.TextWriter] = {}
        for block, tokens in blocks:
            try:
                if block.rotation and abs(block.rotation) > 0.5:
                    self._write_rotated(page, block, tokens, issues)
                else:
                    self._write_block(page, pm, block, tokens, issues)
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("failed to write block %s", block.id)
                issues.append(
                    Issue(
                        Severity.ERROR,
                        "rendering",
                        f"The translated text of this element could not be rendered ({exc}).",
                        page=block.page,
                        bbox=block.bbox,
                        element_id=block.id,
                        source_text=block.source_text[:300],
                    )
                )
        for (color, alpha), tw in self._writers.items():
            rgb = (
                ((color >> 16) & 255) / 255.0,
                ((color >> 8) & 255) / 255.0,
                (color & 255) / 255.0,
            )
            tw.write_text(page, color=rgb, opacity=alpha if alpha < 1 else -1)
        self._writers = {}

    # ----------------------------------------------------------------
    def _tokens(self, block: TextBlock, issues: list[Issue]):
        try:
            return mk.decode(
                block.translation,
                block.style,
                block.styles,
                block.protected,
                set(block.math),
            )
        except mk.MarkupError:
            try:
                return mk.decode(
                    mk.strip_tags(block.translation),
                    block.style,
                    {},
                    block.protected,
                    set(block.math),
                )
            except mk.MarkupError as exc:
                block.status = BlockStatus.FAILED
                issues.append(
                    Issue(
                        Severity.ERROR,
                        "formatting",
                        f"Translated markup is invalid ({exc}); the original text was kept.",
                        page=block.page,
                        bbox=block.bbox,
                        element_id=block.id,
                        source_text=block.source_text[:300],
                    )
                )
                return None

    def _frame(self, block: TextBlock) -> Frame:
        lines = block.lines
        text_x0 = block.text_x0 if block.text_x0 is not None else block.bbox[0]
        region = block.region or block.bbox
        single = len(lines) == 1
        size = block.style.size
        if single:
            align = block.align
            if align == "center":
                cx = (text_x0 + block.bbox[2]) / 2
                half = max(
                    min(cx - region[0], region[2] - cx), (block.bbox[2] - text_x0) / 2
                )
                x0, x1, anchor = cx - half, cx + half, cx
            elif align == "right":
                x0, x1, anchor = min(region[0], text_x0), block.bbox[2], block.bbox[2]
            else:
                x0, x1, anchor = text_x0, max(region[2], block.bbox[2]), None
            return Frame(
                x0=x0,
                x1=x1,
                top=block.bbox[1],
                bottom=max(region[3], block.bbox[3]),
                first_baseline=lines[0].baseline,
                align=align,
                line_pitch=0.0,
                size=size,
                single_line=True,
                anchor_x=anchor,
            )
        col_x0 = min([text_x0] + [ln.bbox[0] for ln in lines[1:]])
        col_x1 = max(block.bbox[2], col_x0 + 4 * size)
        return Frame(
            x0=col_x0,
            x1=col_x1,
            top=block.bbox[1],
            bottom=max(region[3], block.bbox[3]),
            first_baseline=lines[0].baseline,
            first_indent=text_x0 - col_x0,
            rest_indent=max(0.0, block.rest_indent),
            align=block.align,
            line_pitch=block.line_pitch,
            size=size,
        )

    @staticmethod
    def _math_boxes(block: TextBlock) -> dict[str, MathBox]:
        boxes = {}
        for rid, region in block.math.items():
            x0, y0, x1, y1 = region.bbox
            boxes[rid] = MathBox(
                rid,
                x1 - x0,
                max(region.baseline - y0, 0.0),
                max(y1 - region.baseline, 0.0),
            )
        return boxes

    def _write_block(
        self, page, pm: PageModel, block: TextBlock, tokens, issues
    ) -> None:
        frame = self._frame(block)
        maths = self._math_boxes(block)
        res = self.typesetter.fit(
            tokens, frame, block.style, maths, self.min_scale, self.hard_min_scale
        )
        self._draw(page, block, res)
        block.placed_bbox = res.bbox
        block.font_scale = res.scale
        self._report_fit(block, res, frame, issues)

    def _draw(
        self, page: pymupdf.Page, block: TextBlock, res: TypesetResult, morph=None
    ) -> None:
        # Page-level writers are flushed once per page (one font resource per font);
        # rotated text needs its own writer because the morph applies to a whole writer.
        writers: dict[tuple, pymupdf.TextWriter] = (
            {} if morph is not None else self._writers
        )
        for pf in res.fragments:
            frag = pf.fragment
            key = (frag.style.color, round(frag.style.alpha, 2))
            tw = writers.get(key)
            if tw is None:
                tw = writers[key] = pymupdf.TextWriter(page.rect)
            tw.append(
                (pf.x, pf.baseline), frag.text, font=frag.font, fontsize=frag.size
            )
            if (frag.style.underline or frag.style.strike) and morph is None:
                width = max(0.4, frag.size * 0.06)
                color = frag.style.rgb
                if frag.style.underline:
                    y = pf.baseline + frag.size * 0.12
                    page.draw_line(
                        (pf.x, y), (pf.x + frag.width, y), color=color, width=width
                    )
                if frag.style.strike:
                    y = pf.baseline - frag.size * 0.3
                    page.draw_line(
                        (pf.x, y), (pf.x + frag.width, y), color=color, width=width
                    )
        if morph is not None:
            for (color, alpha), tw in writers.items():
                style = block.style.copy(color=color)
                tw.write_text(
                    page,
                    color=style.rgb,
                    opacity=alpha if alpha < 1 else -1,
                    morph=morph,
                )
        for pm_ in res.maths:
            region = block.math[pm_.box.region_id]
            try:
                idx, clip = self.math_source.clip_for(region, block.id)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("formula clip failed for %s: %s", block.id, exc)
                continue
            scale = res.scale
            m = MATH_MARGIN * scale
            x0, y0, x1, y1 = pm_.rect
            target = pymupdf.Rect(x0 - m, y0 - m, x1 + m, y1 + m)
            page.show_pdf_page(
                target,
                self.math_source.clips,
                idx,
                clip=pymupdf.Rect(clip),
                keep_proportion=False,
            )

    def _write_rotated(self, page, block: TextBlock, tokens, issues) -> None:
        line = block.lines[0]
        glyphs = [g for g in line.glyphs if not g.c.isspace()]
        if not glyphs:
            return
        dx, dy = line.direction
        origin = glyphs[0].origin
        last = glyphs[-1]
        length = abs((last.bbox[2] - glyphs[0].bbox[0]) * dx) + abs(
            (max(g.bbox[3] for g in glyphs) - min(g.bbox[1] for g in glyphs)) * dy
        )
        length = max(length, block.style.size)
        frame = Frame(
            x0=0.0,
            x1=length * 1.6,
            top=-block.style.size,
            bottom=block.style.size * 0.4,
            first_baseline=0.0,
            align="left",
            size=block.style.size,
            single_line=True,
        )
        res = self.typesetter.fit(
            tokens, frame, block.style, {}, self.min_scale, self.hard_min_scale
        )
        width = res.bbox[2] - res.bbox[0] if res.fragments else 0.0
        offset = (length - width) / 2  # keep the label centred on its original span
        for pf in res.fragments:
            pf.x = origin[0] + offset + pf.x
            pf.baseline = origin[1] + pf.baseline
        angle = math.degrees(math.atan2(dy, dx))
        morph = (pymupdf.Point(origin), pymupdf.Matrix(-angle))
        self._draw(page, block, res, morph=morph)
        block.placed_bbox = block.bbox
        block.font_scale = res.scale
        if res.scale < self.min_scale - 1e-6 or res.overflow:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "layout",
                    f"Rotated label compressed to {res.scale:.0%} of its original size.",
                    page=block.page,
                    bbox=block.bbox,
                    element_id=block.id,
                    source_text=block.source_text[:200],
                )
            )

    def _report_fit(
        self, block: TextBlock, res: TypesetResult, frame: Frame, issues: list[Issue]
    ) -> None:
        if res.overflow:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "overflow",
                    "The English text needs more space than the original layout provides; it was "
                    f"compressed to {res.scale:.0%} and may overlap neighbouring content.",
                    page=block.page,
                    bbox=res.bbox,
                    element_id=block.id,
                    source_text=block.source_text[:300],
                    suggestion="Review this area; consider shortening the translation.",
                )
            )
        elif res.scale < self.min_scale - 1e-6:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "font_scaled",
                    f"Font size reduced to {res.scale:.0%} of the original to fit the English text.",
                    page=block.page,
                    bbox=res.bbox,
                    element_id=block.id,
                    source_text=block.source_text[:300],
                )
            )
        elif res.scale < 0.9:
            block.notes.append(f"font scaled to {res.scale:.0%}")


def group_blocks_by_page(blocks: list[TextBlock]) -> dict[int, list[TextBlock]]:
    out: dict[int, list[TextBlock]] = defaultdict(list)
    for b in blocks:
        out[b.page].append(b)
    return out


def is_rewritable(block: TextBlock) -> bool:
    return block.kind != ElementKind.FORMULA and block.status == BlockStatus.TRANSLATED
