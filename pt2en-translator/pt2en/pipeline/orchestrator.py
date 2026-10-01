"""End-to-end translation pipeline.

    Analyzing -> Extracting -> OCR -> Translating -> Rebuilding -> Validating -> Finalizing

Every stage is a separate, replaceable component; this module only wires
them together, tracks progress and handles cancellation and QA repairs.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import pymupdf

from pt2en.config import QAReviewMode, Settings
from pt2en.errors import JobCancelled, PipelineError
from pt2en.ingestion.loader import load_pdf
from pt2en.layout.analyzer import analyze_document
from pt2en.layout.detector import create_detector
from pt2en.model import BlockStatus, DocumentModel, Issue, PageStrategy, Severity
from pt2en.ocr.base import OCREngine
from pt2en.ocr.tesseract import create_ocr
from pt2en.pipeline.options import JobOptions
from pt2en.pipeline.progress import ProgressEvent, ProgressReporter, Stage
from pt2en.reconstruction.fonts import FontResolver
from pt2en.reconstruction.writer import MathClipSource, PageWriter
from pt2en.translation import markup as mk
from pt2en.translation.base import Translator
from pt2en.translation.engine import (
    TranslationEngine,
    TranslationItem,
    item_from_block,
    item_from_image_box,
)
from pt2en.translation.memory import TranslationMemory
from pt2en.translation.registry import create_translator
from pt2en.validation.report import build_report, build_review_pdf
from pt2en.validation.validator import QualityValidator
from pt2en.visual.image_text import ImageTextTranslator
from pt2en.visual.scanned import ScannedPageProcessor

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    output_pdf: bytes
    review_pdf: bytes
    report: dict
    model: DocumentModel
    issues: list[Issue] = field(default_factory=list)


class TranslationPipeline:
    def __init__(
        self,
        settings: Settings,
        options: Optional[JobOptions] = None,
        *,
        progress: Optional[Callable[[ProgressEvent], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        translator: Optional[Translator] = None,
        ocr: Optional[OCREngine] = None,
        memory: Optional[TranslationMemory] = None,
    ):
        self.settings = settings
        self.options = options or JobOptions.from_settings(settings)
        self.reporter = ProgressReporter(progress)
        self.cancel_event = cancel_event or threading.Event()
        self._translator = translator
        self._ocr = ocr
        self._memory = memory
        self.issues: list[Issue] = []

    # ------------------------------------------------------------------
    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled()

    def _translator_instance(self) -> Translator:
        if self._translator is None:
            self._translator = create_translator(
                self.options.resolved_provider(self.settings), self.settings
            )
        return self._translator

    def _ocr_instance(self) -> OCREngine:
        if self._ocr is None:
            engine = self.settings.ocr_engine if self.options.ocr_enabled else "none"
            self._ocr = create_ocr(
                engine, self.settings.ocr_languages, self.settings.tesseract_cmd
            )
        return self._ocr

    # ------------------------------------------------------------------
    def run(self, pdf: bytes) -> PipelineResult:
        rep = self.reporter
        settings = self.settings
        translator = self._translator_instance()  # fail fast on configuration errors
        ocr = self._ocr_instance()

        # ---------------------------------------------------------- analyze
        rep.start(Stage.ANALYZING, "Reading the PDF structure")
        loaded = load_pdf(pdf, max_pages=settings.max_pages)
        for note in loaded.notes:
            self.issues.append(Issue(Severity.INFO, "document", note))
        detector = create_detector(
            settings.layout_detector, settings.doclayout_model_path
        )
        model = analyze_document(
            loaded.doc,
            detector=detector,
            progress=lambda f, m: rep.update(f, m),
            cancel_check=self.check_cancelled,
            ocr=ocr,
        )
        for pm in model.pages:
            for note in pm.notes:
                if "ligature" in note:
                    self.issues.append(
                        Issue(Severity.INFO, "extraction", note, page=pm.number)
                    )

        # ---------------------------------------------------------- extract
        rep.start(Stage.EXTRACTING, "Extracting text, formulas, tables and images")
        native_blocks = [b for b in model.blocks() if b.translatable]
        for i, block in enumerate(native_blocks):
            mk.encode_block(block)
            if i % 50 == 0:
                rep.update(i / max(len(native_blocks), 1))
        scanned_pages = [p for p in model.pages if p.strategy == PageStrategy.SCANNED]
        images = (
            [img for img in model.images()] if self.options.translate_images else []
        )
        image_jobs = [
            img
            for img in images
            if model.pages[img.page].strategy == PageStrategy.NATIVE
        ]
        rep.reweight(
            {
                Stage.ANALYZING: 1.0 + 0.1 * model.page_count,
                Stage.EXTRACTING: 0.5,
                Stage.OCR: 0.2 + 1.5 * len(image_jobs) + 4.0 * len(scanned_pages),
                Stage.TRANSLATING: 2.0
                + sum(len(b.source_markup) for b in native_blocks)
                / 800.0
                * (1.0 if translator.is_llm else 0.1),
                Stage.REBUILDING: 0.5 + 0.2 * model.page_count,
                Stage.VALIDATING: 1.0
                + 0.4 * model.page_count
                + 1.0 * len(image_jobs)
                + (
                    sum(len(b.source_markup) for b in native_blocks) / 2500.0
                    if translator.is_llm
                    else 0
                ),
                Stage.FINALIZING: 0.5,
            }
        )
        rep.update(
            1.0,
            f"{len(native_blocks)} text blocks, {len(images)} images, {len(scanned_pages)} scanned page(s)",
        )

        # -------------------------------------------------------------- OCR
        rep.start(Stage.OCR, "Recognising text in scanned pages and images")
        scan_proc = ScannedPageProcessor(ocr, settings)
        image_proc = ImageTextTranslator(ocr, settings)
        work_units = len(scanned_pages) + len(image_jobs)
        done = 0
        for pm in scanned_pages:
            self.check_cancelled()
            scan_proc.analyze(loaded.doc, pm, self.issues)
            for block in pm.blocks:
                if block.translatable:
                    mk.encode_block(block)
            done += 1
            rep.update(
                done / max(work_units, 1), f"OCR of scanned page {pm.number + 1}"
            )
        if not ocr.available() and image_jobs:
            self.issues.append(
                Issue(
                    Severity.WARNING,
                    "ocr",
                    "OCR is not available: text inside images was not translated.",
                    suggestion="Install Tesseract OCR with Portuguese data (tesseract-ocr-por).",
                )
            )
        for img in image_jobs:
            self.check_cancelled()
            pm = model.pages[img.page]
            try:
                image_proc.analyze(loaded.doc, img, pm.width, pm.height, self.issues)
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("image analysis failed for %s", img.id)
                img.status = "failed"
                self.issues.append(
                    Issue(
                        Severity.WARNING,
                        "image_text",
                        f"Image could not be analysed ({exc}).",
                        page=img.page,
                        bbox=img.bbox,
                    )
                )
            done += 1
            rep.update(
                done / max(work_units, 1),
                f"Analysed image {done - len(scanned_pages)}/{len(image_jobs)}",
            )

        # ------------------------------------------------------- translate
        rep.start(Stage.TRANSLATING, "Building the terminology glossary")
        memory = (
            self._memory
            if self._memory is not None
            else TranslationMemory(settings.resolved_cache_path)
        )
        engine = TranslationEngine(
            translator,
            settings,
            self.options,
            memory=memory,
            issues=self.issues,
            cancel_check=self.check_cancelled,
        )
        image_texts = [
            box.text for img in images for box in img.text_boxes if box.translatable
        ]
        engine.build_document_glossary(model, image_texts)
        rep.update(
            0.05,
            f"Glossary ready ({len(engine.glossary.to_list())} document/user terms)",
        )
        block_items: list[tuple[object, TranslationItem]] = []
        for block in model.blocks():
            if block.translatable and block.kind.value != "formula":
                if not block.source_markup:
                    mk.encode_block(block)
                block_items.append((block, item_from_block(block)))
        image_items: list[tuple[object, TranslationItem]] = []
        for img in images:
            for box in img.text_boxes:
                if box.translatable:
                    image_items.append(
                        (box, item_from_image_box(box, img.page, img.bbox))
                    )
        items = [it for _, it in block_items] + [it for _, it in image_items]
        engine.translate_items(
            items, progress=lambda f, m: rep.update(0.05 + 0.95 * f, m)
        )
        for block, item in block_items:
            block.translation, block.status = item.translation, item.status
            block.notes.extend(item.notes)
        for box, item in image_items:
            box.translation = mk.unescape(mk.strip_tags(item.translation or box.text))
            box.status = item.status

        # --------------------------------------------- rebuild + validate
        validator = QualityValidator(
            settings,
            ocr,
            engine,
            review_mode=self.options.qa_review,
            progress=lambda f, m: rep.update(f, m),
        )
        repairs: dict = {"retranslated": [], "corrected": []}
        rounds = 1 + max(0, settings.qa_max_repair_rounds)
        out_doc = None
        qa = None
        for round_no in range(rounds):
            rep.start(
                Stage.REBUILDING,
                "Rebuilding the page layout" + (" (repair pass)" if round_no else ""),
            )
            build_issues: list[Issue] = []
            out_doc = self._rebuild(
                loaded.source_bytes, model, image_proc, scan_proc, build_issues
            )
            rep.start(Stage.VALIDATING, "Validating the translated document")
            src_doc = pymupdf.open(stream=loaded.source_bytes, filetype="pdf")
            qa = validator.validate(src_doc, out_doc, model, llm_review=(round_no == 0))
            src_doc.close()
            if round_no == rounds - 1 or not qa.needs_repair:
                break
            self.check_cancelled()
            changed = self._repair(model, engine, qa, repairs)
            if not changed:
                break
        assert out_doc is not None and qa is not None
        all_issues = self.issues + build_issues + qa.issues

        # ---------------------------------------------------------- finalize
        rep.start(Stage.FINALIZING, "Embedding fonts and writing the English PDF")
        meta = dict(loaded.doc.metadata or {})
        meta["producer"] = "pt2en-translator (PyMuPDF)"
        meta["title"] = self._translated_title(model) or meta.get("title", "")
        try:
            out_doc.set_metadata(
                {
                    k: v
                    for k, v in meta.items()
                    if k
                    in ("title", "author", "subject", "keywords", "creator", "producer")
                }
            )
        except Exception:  # pragma: no cover
            pass
        try:
            out_doc.subset_fonts()
        except Exception as exc:  # pragma: no cover - font subsetting is best effort
            log.warning("font subsetting failed: %s", exc)
        rep.update(0.5)
        output = out_doc.tobytes(garbage=3, deflate=True, use_objstms=1)
        out_doc.close()
        stats = dict(engine.stats)
        usage = getattr(translator, "usage", None)
        if usage:
            stats["llm_usage"] = dict(usage)
        report = build_report(
            model,
            all_issues,
            qa.checks,
            provider=translator.describe(),
            options={
                "style": self.options.style.value,
                "english_variant": self.options.english_variant.value,
                "preserve_terminology": self.options.preserve_terminology,
                "localize_numbers": self.options.localize_numbers,
                "qa_review": self.options.qa_review.value,
            },
            glossary=engine.glossary.to_list(),
            stats=stats,
            timings=rep.timings,
            repairs=repairs,
        )
        review = build_review_pdf(output, report)
        rep.finish("Translation complete")
        return PipelineResult(
            output_pdf=output,
            review_pdf=review,
            report=report,
            model=model,
            issues=all_issues,
        )

    # ------------------------------------------------------------------
    def _rebuild(
        self, source_bytes, model, image_proc, scan_proc, issues
    ) -> pymupdf.Document:
        settings = self.settings
        out = pymupdf.open(stream=source_bytes, filetype="pdf")
        fonts = FontResolver(
            out,
            extra_dirs=settings.font_dirs,
            reuse_original=settings.reuse_original_fonts,
        )
        math_src = MathClipSource(source_bytes, model.pages)
        writer = PageWriter(
            fonts,
            math_src,
            min_scale=settings.min_font_scale,
            hard_min_scale=settings.hard_min_font_scale,
        )
        try:
            for i, pm in enumerate(model.pages):
                self.check_cancelled()
                if pm.strategy == PageStrategy.SCANNED:
                    scan_proc.rebuild(out, pm)
                else:
                    for img in pm.images:
                        if img.status in ("analysed", "translated") and img.has_text:
                            image_proc.render(out, img, issues)
                writer.write(out[pm.number], pm, issues)
                self.reporter.update(
                    (i + 1) / model.page_count,
                    f"Rebuilt page {pm.number + 1}/{model.page_count}",
                )
        finally:
            # Formula clips are copied into ``out`` by show_pdf_page; the source can go.
            math_src.close()
        return out

    def _repair(
        self, model: DocumentModel, engine: TranslationEngine, qa, repairs: dict
    ) -> bool:
        changed = False
        for block_id, corrected in qa.corrections.items():
            block = model.block_by_id(block_id)
            if block is None:
                continue
            block.translation = corrected
            repairs["corrected"].append(block_id)
            changed = True
        targets = []
        for block_id, note in qa.retranslate.items():
            if block_id in qa.corrections:
                continue
            block = model.block_by_id(block_id)
            if block is None or block.status != BlockStatus.TRANSLATED:
                continue
            item = item_from_block(block)
            targets.append((block, item, note))
        if targets:
            engine.retranslate([(item, note) for _, item, note in targets])
            for block, item, _ in targets:
                if (
                    item.status == BlockStatus.TRANSLATED
                    and item.translation
                    and item.translation != block.translation
                ):
                    block.translation = item.translation
                    repairs["retranslated"].append(block.id)
                    changed = True
        return changed

    @staticmethod
    def _translated_title(model: DocumentModel) -> str:
        from pt2en.model import ElementKind

        for block in model.blocks():
            if block.kind == ElementKind.HEADING and block.translation:
                return mk.plain_text(
                    block.translation,
                    block.protected,
                    {k: v.text for k, v in block.math.items()},
                )[:200]
        return ""


def translate_file(
    pdf: bytes,
    settings: Settings,
    options: Optional[JobOptions] = None,
    progress: Optional[Callable[[ProgressEvent], None]] = None,
) -> PipelineResult:
    """Convenience wrapper used by the CLI and tests."""
    try:
        return TranslationPipeline(settings, options, progress=progress).run(pdf)
    except (PipelineError, JobCancelled):
        raise
    except Exception as exc:  # pragma: no cover - surfaced to the user
        log.exception("pipeline failed")
        raise PipelineError(
            "Unexpected error while processing the document.", detail=str(exc)
        ) from exc


__all__ = ["TranslationPipeline", "PipelineResult", "translate_file", "QAReviewMode"]
