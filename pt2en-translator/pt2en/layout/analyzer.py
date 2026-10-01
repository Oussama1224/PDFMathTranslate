"""Document analysis: parsing, page strategy routing, segmentation, classification."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pymupdf

from pt2en.layout.blocks import BlockBuilder
from pt2en.layout.classifier import ElementClassifier
from pt2en.layout.detector import LayoutDetector, LayoutHints, NullLayoutDetector
from pt2en.model import DocumentModel, PageModel, PageStrategy, rect_area
from pt2en.ocr.base import OCREngine
from pt2en.parsing.pdf_parser import RawPage, parse_page
from pt2en.parsing.repair import repair_unmapped

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


def decide_strategy(raw: RawPage) -> PageStrategy:
    """Route a page to native-text processing or full-page OCR."""
    page_area = max(raw.width * raw.height, 1.0)
    largest = max((rect_area(img.bbox) for img in raw.images), default=0.0)
    if raw.image_coverage >= 0.6 and largest >= 0.5 * page_area and raw.text_chars < 30:
        return PageStrategy.SCANNED
    return PageStrategy.NATIVE


def scan_dpi(raw: RawPage) -> int:
    """Approximate native resolution of the dominant page image."""
    if not raw.images:
        return 0
    img = max(raw.images, key=lambda i: rect_area(i.bbox))
    width_in = max((img.bbox[2] - img.bbox[0]) / 72.0, 0.1)
    return int(img.width / width_in) if img.width else 0


def analyze_document(
    doc: pymupdf.Document,
    *,
    detector: Optional[LayoutDetector] = None,
    progress: Optional[ProgressFn] = None,
    cancel_check: Optional[Callable[[], None]] = None,
    ocr: Optional[OCREngine] = None,
) -> DocumentModel:
    detector = detector or NullLayoutDetector()
    model = DocumentModel(page_count=doc.page_count, metadata=dict(doc.metadata or {}))
    model.title = (model.metadata.get("title") or "").strip()
    hints: dict[int, LayoutHints] = {}

    for page in doc:
        if cancel_check:
            cancel_check()
        raw = parse_page(doc, page)
        repaired = repair_unmapped(page, raw, ocr)
        pm = PageModel(
            number=page.number,
            width=raw.width,
            height=raw.height,
            strategy=decide_strategy(raw),
            images=raw.images,
            drawings_count=raw.drawings_count,
            figure_rects=raw.figure_rects,
            table_rects=raw.table_rects,
            drawing_rects=[
                r
                for r in raw.drawing_rects
                if (r[2] - r[0]) < 60 and (r[3] - r[1]) < 60
            ][:2000],
            text_chars=raw.text_chars,
            invisible_chars=raw.invisible_chars,
            image_coverage=raw.image_coverage,
        )
        if repaired:
            pm.notes.append(
                f"{repaired} word(s) with unmapped ligature glyphs were repaired."
            )
        if pm.strategy == PageStrategy.SCANNED:
            pm.scan_dpi = scan_dpi(raw)
            pm.notes.append("Scanned page: text will be recognised with OCR.")
        else:
            pm.blocks = BlockBuilder(raw).build()
            if not isinstance(detector, NullLayoutDetector):
                try:
                    pix = page.get_pixmap(dpi=96)
                    arr = np.frombuffer(pix.samples, np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )[:, :, :3]
                    hints[page.number] = detector.detect(arr, raw.width, raw.height)
                except Exception as exc:  # pragma: no cover - optional model
                    log.warning(
                        "layout detection failed on page %s: %s", page.number, exc
                    )
        model.pages.append(pm)
        if progress:
            progress(
                (page.number + 1) / doc.page_count,
                f"Analysed page {page.number + 1}/{doc.page_count}",
            )

    ElementClassifier(hints).classify(model)
    return model
