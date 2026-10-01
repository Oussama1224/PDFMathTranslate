"""Scanned pages: OCR, paragraph reconstruction and background cleaning.

The page image is OCR'd (with a denoised/binarised second attempt for poor
scans), paragraphs become regular text blocks (so they go through the same
translation, typesetting and QA as native text), and the Portuguese text is
erased from the scan. The rebuilt page is the cleaned scan as background with
searchable English text on top. Formulas, figures and anything that cannot be
read reliably stay as untouched pixels.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pymupdf
from PIL import Image

from pt2en.config import Settings
from pt2en.layout.blocks import BlockBuilder, detect_alignment
from pt2en.layout.classifier import PAGE_NUMBER_RE, compute_regions
from pt2en.layout.math import unicode_is_math
from pt2en.model import (
    BlockStatus,
    ElementKind,
    Glyph,
    Issue,
    Line,
    OCRWord,
    PageModel,
    Run,
    Severity,
    TextBlock,
    TextStyle,
    union_rects,
)
from pt2en.ocr.base import OCREngine
from pt2en.visual import text_render as tr

log = logging.getLogger(__name__)


@dataclass
class _ScanWork:
    rgb: np.ndarray
    dpi: int
    words: dict[str, list[OCRWord]]
    gray: bool


def _score(words: list[OCRWord]) -> float:
    good = [w.confidence for w in words if w.confidence >= 0]
    if not good:
        return 0.0
    return float(np.mean(good)) * np.sqrt(len(good))


class ScannedPageProcessor:
    def __init__(self, ocr: OCREngine, settings: Settings):
        self.ocr = ocr
        self.settings = settings
        self._work: dict[int, _ScanWork] = {}

    # ------------------------------------------------------------ analyse
    def analyze(
        self, doc: pymupdf.Document, pm: PageModel, issues: list[Issue]
    ) -> None:
        if not self.ocr.available():
            issues.append(
                Issue(
                    Severity.ERROR,
                    "ocr",
                    "This page is a scanned image but no OCR engine is available; it was not translated.",
                    page=pm.number,
                    suggestion="Install Tesseract with the Portuguese language pack (tesseract-ocr-por).",
                )
            )
            return
        dpi = int(min(max(pm.scan_dpi or 200, 150), self.settings.ocr_dpi))
        page = doc[pm.number]
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        rgb = (
            np.frombuffer(pix.samples, np.uint8)
            .reshape(pix.height, pix.width, pix.n)[:, :, :3]
            .copy()
        )
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        is_gray = bool(
            np.mean(np.abs(rgb.astype(np.int16) - gray[..., None].astype(np.int16))) < 4
        )

        rotation = self.ocr.detect_rotation(gray)
        if rotation:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "ocr",
                    f"The scanned page appears rotated by {rotation}°; its text was not translated.",
                    page=pm.number,
                    suggestion="Rotate the page upright in the source PDF and translate again.",
                )
            )
            return

        words = self.ocr.recognize(gray, psm=3)
        if not words or np.mean([w.confidence for w in words]) < 80:
            cleaned = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)
            binar = cv2.adaptiveThreshold(
                cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
            )
            alt = self.ocr.recognize(binar, psm=3)
            if _score(alt) > _score(words):
                words = alt
                pm.notes.append(
                    "OCR used the denoised version of this low-quality scan."
                )
        words = [w for w in words if w.text.strip()]

        groups: dict[tuple[int, int], list[OCRWord]] = {}
        for w in words:
            groups.setdefault((w.block, w.paragraph), []).append(w)

        blocks: list[TextBlock] = []
        min_conf = self.settings.ocr_min_confidence
        for key in sorted(groups, key=lambda k: min(w.bbox[1] for w in groups[k])):
            pwords = groups[key]
            block = self._make_block(pm, pwords, dpi, rgb, len(blocks))
            if block is None:
                continue
            conf = block.ocr_confidence or 0.0
            text = block.text
            letters = sum(c.isalpha() for c in text)
            mathy = sum(unicode_is_math(c) for c in text) > max(2, 0.15 * len(text))
            if letters < 2:
                block.translatable = False
            elif mathy:
                block.kind = ElementKind.FORMULA
                block.translatable = False
                issues.append(
                    Issue(
                        Severity.INFO,
                        "formula",
                        "A formula on this scanned page was kept as part of the image (unchanged).",
                        page=pm.number,
                        bbox=block.bbox,
                        element_id=block.id,
                    )
                )
            elif conf < min_conf:
                block.translatable = False
                issues.append(
                    Issue(
                        Severity.WARNING,
                        "ocr_low_confidence",
                        f"Scanned text could not be read reliably (OCR confidence {conf:.0f}%); "
                        "it was left untranslated in the image.",
                        page=pm.number,
                        bbox=block.bbox,
                        element_id=block.id,
                        source_text=text[:300],
                        suggestion="Translate this paragraph manually.",
                    )
                )
            self._work.setdefault(pm.number, _ScanWork(rgb, dpi, {}, is_gray)).words[
                block.id
            ] = pwords
            blocks.append(block)
        if pm.number not in self._work:
            self._work[pm.number] = _ScanWork(rgb, dpi, {}, is_gray)
        self._classify(pm, blocks)
        pm.blocks = blocks
        compute_regions(pm)

    def _make_block(
        self, pm: PageModel, words: list[OCRWord], dpi: int, rgb: np.ndarray, idx: int
    ) -> Optional[TextBlock]:
        f = 72.0 / dpi
        by_line: dict[int, list[OCRWord]] = {}
        for w in words:
            by_line.setdefault(w.line, []).append(w)
        lines: list[Line] = []
        colors = []
        for ln in sorted(by_line, key=lambda k: min(w.bbox[1] for w in by_line[k])):
            lw = sorted(by_line[ln], key=lambda w: w.bbox[0])
            x0 = min(w.bbox[0] for w in lw) * f
            y0 = min(w.bbox[1] for w in lw) * f
            x1 = max(w.bbox[2] for w in lw) * f
            y1 = max(w.bbox[3] for w in lw) * f
            text_line = " ".join(w.text for w in lw)
            has_desc = any(c in "gjpqy,;ç" for c in text_line)
            has_asc = any(
                c.isupper() or c in "bdfhklt'\"áéíóúàâêôãõ" or c.isdigit()
                for c in text_line
            )
            ink = y1 - y0
            # Ink height relative to the em: ascender+descender lines ~0.93, x-height only ~0.5.
            ratio = (
                0.93
                if (has_desc and has_asc)
                else 0.72 if has_asc else 0.7 if has_desc else 0.5
            )
            size = max(4.0, ink / ratio)
            baseline = y1 - (0.22 * size if has_desc else 0.0)
            glyphs: list[Glyph] = []
            for k, w in enumerate(lw):
                if k:
                    prev = lw[k - 1]
                    glyphs.append(
                        Glyph(
                            " ",
                            (prev.bbox[2] * f, y0, w.bbox[0] * f, y1),
                            (prev.bbox[2] * f, baseline),
                            size,
                        )
                    )
                n = max(1, len(w.text))
                wx0, wx1 = w.bbox[0] * f, w.bbox[2] * f
                step = (wx1 - wx0) / n
                for i, ch in enumerate(w.text):
                    glyphs.append(
                        Glyph(
                            ch,
                            (wx0 + i * step, y0, wx0 + (i + 1) * step, y1),
                            (wx0 + i * step, baseline),
                            size,
                        )
                    )
                bx = tuple(int(v) for v in w.bbox)
                if bx[2] > bx[0] and bx[3] > bx[1]:
                    bg = tr.ring_background(rgb, bx)
                    mask = tr.text_mask(rgb, bx, bg)
                    px = rgb[bx[1] : bx[3], bx[0] : bx[2]][mask]
                    if px.size:
                        colors.append(np.median(px, axis=0))
            style = TextStyle(
                font="", size=size, serif=self.settings.scan_font_family != "sans"
            )
            lines.append(
                Line(
                    runs=[Run(glyphs=glyphs, style=style)],
                    bbox=(x0, y0, x1, y1),
                    baseline=baseline,
                    size=size,
                    ocr_confidence=float(np.mean([w.confidence for w in lw])),
                )
            )
        if not lines:
            return None
        color = 0
        if colors:
            c = np.median(np.array(colors), axis=0).astype(int)
            # Scans are rarely pure black: keep the measured tone but avoid washed-out text.
            c = np.clip(c - 25, 0, 255)
            color = (int(c[0]) << 16) | (int(c[1]) << 8) | int(c[2])
        sizes = sorted(ln.size for ln in lines)
        size = sizes[len(sizes) // 2]
        serif = self.settings.scan_font_family != "sans"
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        longest = max(
            (
                (
                    ln,
                    " ".join(
                        w.text for w in sorted(by_line[k], key=lambda w: w.bbox[0])
                    ),
                )
                for ln, k in zip(
                    lines,
                    sorted(by_line, key=lambda k: min(w.bbox[1] for w in by_line[k])),
                )
            ),
            key=lambda t: len(t[1]),
        )
        lb = tuple(int(v / f) for v in longest[0].bbox)
        bold = tr.looks_bold(
            g[lb[1] : lb[3], lb[0] : lb[2]], longest[1], lb[3] - lb[1], 1, serif
        )
        base = TextStyle(font="", size=size, serif=serif, color=color, bold=bold)
        for ln in lines:
            for run in ln.runs:
                run.style = base.copy(
                    size=ln.size if abs(ln.size - size) > 0.15 * size else size
                )
        block = TextBlock(
            id=f"p{pm.number}-ocr{idx}",
            page=pm.number,
            kind=ElementKind.PARAGRAPH,
            bbox=union_rects(ln.bbox for ln in lines),
            lines=lines,
            style=base,
            source="ocr",
            ocr_confidence=float(np.mean([w.confidence for w in words])),
        )
        BlockBuilder._detect_prefix(block)
        BlockBuilder._measure(block)
        block.align = detect_alignment(lines, block.bbox)
        return block

    @staticmethod
    def _classify(pm: PageModel, blocks: list[TextBlock]) -> None:
        sizes = sorted(b.style.size for b in blocks if b.translatable) or [10.0]
        body = sizes[len(sizes) // 2]
        for b in blocks:
            text = b.text.strip()
            if b.bbox[1] > pm.height * 0.9 and PAGE_NUMBER_RE.match(text):
                b.kind = ElementKind.PAGE_NUMBER
            elif (
                b.kind == ElementKind.PARAGRAPH
                and len(b.lines) <= 2
                and b.style.size >= body * 1.15
            ):
                b.kind = ElementKind.HEADING

    # ------------------------------------------------------------ rebuild
    def rebuild(self, doc: pymupdf.Document, pm: PageModel) -> bool:
        work = self._work.get(pm.number)
        if work is None:
            return False
        translated = [
            b for b in pm.blocks if b.status == BlockStatus.TRANSLATED and b.translation
        ]
        if not translated:
            return False
        rgb = work.rgb.copy()
        f = 72.0 / work.dpi
        for block in translated:
            for w in work.words.get(block.id, []):
                if (
                    block.prefix_bbox
                    and block.prefix_text
                    and w.text.startswith(block.prefix_text)
                ):
                    wx = (w.bbox[0] * f + w.bbox[2] * f) / 2
                    if block.prefix_bbox[0] - 1 <= wx <= block.prefix_bbox[2] + 1:
                        continue  # the enumerator/bullet stays as in the scan
                bx = tuple(int(round(v)) for v in w.bbox)
                tr.erase_box(rgb, bx, bx[3] - bx[1])
        page = doc[pm.number]
        page.add_redact_annot(page.rect, fill=False)
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_REMOVE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )
        img = Image.fromarray(rgb)
        if work.gray:
            img = img.convert("L")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85, optimize=True)
        page.insert_image(
            page.rect, stream=buf.getvalue(), keep_proportion=False, overlay=False
        )
        return True
