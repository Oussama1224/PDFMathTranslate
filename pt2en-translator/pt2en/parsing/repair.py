"""Repair of characters that the PDF does not map to Unicode.

Word processors often draw ligatures (Calibri's "ti", "tt", "ft"...) with
glyphs that have no Unicode mapping; extraction then yields U+FFFD
("Obje�vos"). Each affected word is rendered and OCR'd to recover the
real letters; when OCR is unavailable or inconclusive the most likely
ligature for the context is used and the word is reported for review.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pymupdf

from pt2en.model import Glyph, TextStyle
from pt2en.ocr.base import OCREngine
from pt2en.parsing.pdf_parser import RawPage
from pt2en.layout.math import build_runs

log = logging.getLogger(__name__)

UNKNOWN = "�"


def _guess_one(prev: str, nxt: str) -> str:
    """Most plausible ligature for an unmapped glyph when OCR cannot help."""
    if prev == "h" and nxt == "p":
        return "tt"  # https
    if prev == "o" and nxt == "w":
        return "ft"  # software
    return "ti"  # by far the most frequent unmapped ligature (Calibri "ti")


def _ocr_word(page: pymupdf.Page, bbox, ocr: OCREngine) -> Optional[str]:
    rect = pymupdf.Rect(bbox) + (-1.5, -1.5, 1.5, 1.5)
    try:
        pix = page.get_pixmap(
            clip=rect, dpi=400, colorspace=pymupdf.csGRAY, alpha=False
        )
    except Exception:
        return None
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    arr = np.pad(arr, 20, constant_values=255)
    words = ocr.recognize(arr, psm=8)
    if not words:
        return None
    return "".join(w.text for w in words)


def repair_unmapped(page: pymupdf.Page, raw: RawPage, ocr: Optional[OCREngine]) -> int:
    """Replace U+FFFD glyphs in ``raw`` lines; returns the number of repairs."""
    repaired = 0
    for rl in raw.lines:
        line = rl.line
        glyphs = line.glyphs
        if not any(g.c == UNKNOWN for g in glyphs):
            continue
        styles = [run.style for run in line.runs for _ in run.glyphs]
        new_glyphs: list[Glyph] = []
        new_styles: list[TextStyle] = []
        i = 0
        while i < len(glyphs):
            if glyphs[i].c.isspace():
                new_glyphs.append(glyphs[i])
                new_styles.append(styles[i])
                i += 1
                continue
            j = i
            while j < len(glyphs) and not glyphs[j].c.isspace():
                j += 1
            word_glyphs = glyphs[i:j]
            word = "".join(g.c for g in word_glyphs)
            if UNKNOWN not in word:
                new_glyphs.extend(word_glyphs)
                new_styles.extend(styles[i:j])
                i = j
                continue
            fixed = _expand_word(page, word_glyphs, ocr)
            for g, st, text in zip(word_glyphs, styles[i:j], fixed):
                if g.c != UNKNOWN or len(text) == 1:
                    new_glyphs.append(Glyph(text, g.bbox, g.origin, g.size))
                    new_styles.append(st)
                    continue
                # Split the ligature glyph box between its letters.
                n = len(text)
                x0, y0, x1, y1 = g.bbox
                step = (x1 - x0) / n
                for k, ch in enumerate(text):
                    new_glyphs.append(
                        Glyph(
                            ch,
                            (x0 + k * step, y0, x0 + (k + 1) * step, y1),
                            (x0 + k * step, g.origin[1]),
                            g.size,
                        )
                    )
                    new_styles.append(st)
            repaired += 1
            i = j
        line.runs = build_runs(new_glyphs, new_styles)
    return repaired


def _expand_word(
    page: pymupdf.Page, glyphs: list[Glyph], ocr: Optional[OCREngine]
) -> list[str]:
    """Return, per glyph, the text it stands for."""
    word = "".join(g.c for g in glyphs)
    pattern = (
        "^"
        + "".join("([a-zA-Z]{1,3})" if c == UNKNOWN else re.escape(c) for c in word)
        + "$"
    )
    if ocr is not None and ocr.available():
        bbox = (
            min(g.bbox[0] for g in glyphs),
            min(g.bbox[1] for g in glyphs),
            max(g.bbox[2] for g in glyphs),
            max(g.bbox[3] for g in glyphs),
        )
        text = _ocr_word(page, bbox, ocr)
        if text:
            m = re.match(pattern, text.strip(".,;:!?\"'()"))
            if m:
                parts = iter(m.groups())
                return [next(parts) if g.c == UNKNOWN else g.c for g in glyphs]
    out = []
    for i, g in enumerate(glyphs):
        if g.c != UNKNOWN:
            out.append(g.c)
            continue
        prev = glyphs[i - 1].c if i else ""
        nxt = glyphs[i + 1].c if i + 1 < len(glyphs) else ""
        out.append(_guess_one(prev, nxt))
    return out
