"""Quality control of the reconstructed English PDF.

Every check produces a status (pass / warn / fail) plus issues anchored to a
page region. Checks that find fixable problems (untranslated Portuguese in a
translated block, reviewer corrections) also return *repairs* that the
pipeline applies before rebuilding the document once more.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pymupdf

from pt2en.config import QAReviewMode, Settings
from pt2en.model import (
    BlockStatus,
    DocumentModel,
    ElementKind,
    Issue,
    PageStrategy,
    Rect,
    Severity,
    expand_rect,
    intersection_area,
    overlap_ratio,
    rect_area,
)
from pt2en.ocr.base import OCREngine, group_into_lines, line_bbox
from pt2en.translation import markup as mk
from pt2en.translation.engine import TranslationEngine
from pt2en.translation.language import looks_untranslated, portuguese_residue, words_of

log = logging.getLogger(__name__)

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass
class CheckResult:
    id: str
    label: str
    status: str = "pass"  # pass | warn | fail | skipped
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "details": self.details,
        }


@dataclass
class QAResult:
    checks: list[CheckResult] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    retranslate: dict[str, str] = field(default_factory=dict)  # block id -> feedback
    corrections: dict[str, str] = field(
        default_factory=dict
    )  # block id -> corrected markup

    @property
    def needs_repair(self) -> bool:
        return bool(self.retranslate or self.corrections)


class QualityValidator:
    def __init__(
        self,
        settings: Settings,
        ocr: OCREngine,
        engine: TranslationEngine,
        *,
        review_mode: QAReviewMode = QAReviewMode.OFF,
        progress: Optional[Callable[[float, str], None]] = None,
    ):
        self.settings = settings
        self.ocr = ocr
        self.engine = engine
        self.review_mode = review_mode
        self.progress = progress or (lambda f, m: None)

    # ------------------------------------------------------------------
    def validate(
        self,
        src: pymupdf.Document,
        out: pymupdf.Document,
        model: DocumentModel,
        *,
        llm_review: bool = True,
    ) -> QAResult:
        res = QAResult()
        steps = [
            ("page_structure", "Page count, size and order", self._pages),
            ("images", "Images, charts and diagrams present", self._images),
            ("graphics", "Vector graphics preserved", self._graphics),
            ("formulas", "Formulas unchanged", self._formulas),
            ("text", "No missing text", self._text_completeness),
            ("bounds", "Text inside page boundaries", self._bounds),
            ("overlap", "No overlapping elements", self._overlaps),
            ("numbering", "Numbering and page numbers preserved", self._numbering),
            ("terminology", "Consistent terminology", self._terminology),
            (
                "portuguese_text",
                "Second pass: untranslated Portuguese (text layer)",
                self._residual_text,
            ),
            (
                "portuguese_images",
                "Second pass: untranslated Portuguese (images and scans)",
                self._residual_images,
            ),
        ]
        if (
            llm_review
            and self.review_mode != QAReviewMode.OFF
            and self.engine.translator.is_llm
        ):
            steps.append(("review", "Translation accuracy review", self._llm_review))
        for i, (cid, label, fn) in enumerate(steps):
            check = CheckResult(cid, label)
            self.progress(i / len(steps), label)
            try:
                fn(src, out, model, check, res)
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("QA check %s failed", cid)
                check.status = "warn"
                check.details = f"check could not run: {exc}"
            res.checks.append(check)
        self.progress(1.0, "Validation complete")
        return res

    # ------------------------------------------------------------ checks
    def _pages(self, src, out, model, check: CheckResult, res: QAResult) -> None:
        if src.page_count != out.page_count:
            check.status = "fail"
            check.details = (
                f"source has {src.page_count} pages, output has {out.page_count}"
            )
            res.issues.append(Issue(Severity.ERROR, "page_order", check.details))
            return
        bad = []
        for i in range(src.page_count):
            a, b = src[i].rect, out[i].rect
            if abs(a.width - b.width) > 0.5 or abs(a.height - b.height) > 0.5:
                bad.append(i)
                res.issues.append(
                    Issue(Severity.ERROR, "page_order", "Page size changed.", page=i)
                )
        order_problems = []
        for i in range(src.page_count):
            fs, fo = _fingerprint(src[i]), _fingerprint(out[i])
            if model.pages[i].strategy == PageStrategy.SCANNED:
                continue
            if not set(fs[0]).issubset(set(fo[0])) or fs[1:] != fo[1:]:
                order_problems.append(i)
                res.issues.append(
                    Issue(
                        Severity.ERROR,
                        "page_order",
                        "Page content does not match the source page.",
                        page=i,
                    )
                )
        if bad or order_problems:
            check.status = "fail"
            check.details = f"{len(bad)} page(s) resized, {len(order_problems)} page(s) out of order"
        else:
            check.details = (
                f"{out.page_count} pages, same sizes and order as the source"
            )

    def _images(self, src, out, model, check, res) -> None:
        missing = 0
        total = 0
        for i in range(src.page_count):
            s_imgs = [tuple(x["bbox"]) for x in src[i].get_image_info()]
            o_imgs = [tuple(x["bbox"]) for x in out[i].get_image_info()]
            for sb in s_imgs:
                if rect_area(sb) < 4:
                    continue
                total += 1
                if not any(_iou(sb, ob) > 0.9 for ob in o_imgs):
                    missing += 1
                    res.issues.append(
                        Issue(
                            Severity.ERROR,
                            "missing_image",
                            "An image from the source is missing.",
                            page=i,
                            bbox=sb,
                        )
                    )
        translated = sum(1 for img in model.images() if img.status == "translated")
        if missing:
            check.status = "fail"
        check.details = f"{total - missing}/{total} images present; {translated} image(s) had their text translated"

    def _graphics(self, src, out, model, check, res) -> None:
        lost_pages = []
        for pm in model.pages:
            if pm.strategy == PageStrategy.SCANNED:
                continue
            s_count = len(src[pm.number].get_drawings())
            o_count = len(out[pm.number].get_drawings())
            allowance = (
                2
                + sum(len(b.math) for b in pm.blocks)
                + sum(
                    1
                    for b in pm.blocks
                    for ln in b.lines
                    for r in ln.runs
                    if r.style.underline or r.style.strike
                )
            )
            if o_count < s_count - allowance:
                lost_pages.append(pm.number)
                res.issues.append(
                    Issue(
                        Severity.WARNING,
                        "missing_graphics",
                        f"{s_count - o_count} vector drawing element(s) fewer than in the source.",
                        page=pm.number,
                    )
                )
        if lost_pages:
            check.status = "warn"
            check.details = "pages with fewer vector elements: " + ", ".join(
                str(p + 1) for p in lost_pages
            )
        else:
            check.details = "all lines, shapes, charts and diagrams preserved"

    def _formulas(self, src, out, model, check, res) -> None:
        display = 0
        changed = 0
        inline = 0
        for pm in model.pages:
            if pm.strategy == PageStrategy.SCANNED:
                continue
            for block in pm.blocks:
                if block.kind == ElementKind.FORMULA:
                    display += 1
                    diff = _raster_diff(src[pm.number], out[pm.number], block.bbox)
                    if diff > 6.0:
                        changed += 1
                        res.issues.append(
                            Issue(
                                Severity.ERROR,
                                "formula",
                                f"A formula region differs from the source (mean pixel difference {diff:.1f}).",
                                page=pm.number,
                                bbox=block.bbox,
                                element_id=block.id,
                                source_text=block.text[:200],
                            )
                        )
                elif block.status == BlockStatus.TRANSLATED:
                    inline += len(block.math)
                    if block.translation is not None:
                        missing = set(block.math) - set(
                            mk.placeholders(block.translation)
                        )
                        if missing:
                            changed += 1
                            res.issues.append(
                                Issue(
                                    Severity.ERROR,
                                    "formula",
                                    "An inline formula is missing from the translated text.",
                                    page=pm.number,
                                    bbox=block.bbox,
                                    element_id=block.id,
                                )
                            )
        check.status = "fail" if changed else "pass"
        check.details = (
            f"{display} display formula(s) verified pixel-identical; "
            f"{inline} inline formula(s) re-inserted from the source"
        )
        if changed:
            check.details = f"{changed} formula problem(s) found"

    def _text_completeness(self, src, out, model, check, res) -> None:
        missing = leftover = 0
        for pm in model.pages:
            words_out = out[pm.number].get_text("words")
            for block in pm.blocks:
                if (
                    block.status != BlockStatus.TRANSLATED
                    or not block.placed_bbox
                    or block.rotation
                ):
                    continue
                expected = [
                    w.lower()
                    for w in words_of(
                        mk.plain_text(block.translation or "", block.protected, {})
                    )
                ]
                if not expected:
                    continue
                area = expand_rect(block.placed_bbox, 2.0)
                found = {
                    re.sub(r"\W", "", w[4].lower())
                    for w in words_out
                    if intersection_area(tuple(w[:4]), area)
                    > 0.5 * rect_area(tuple(w[:4]))
                }
                joined = " ".join(found)
                hits = sum(
                    1
                    for e in expected
                    if (k := re.sub(r"\W", "", e)) in found
                    or (len(k) > 2 and k in joined)
                )
                if hits / len(expected) < 0.85:
                    missing += 1
                    res.issues.append(
                        Issue(
                            Severity.ERROR,
                            "missing_text",
                            f"Only {hits}/{len(expected)} words of the translation are present in the output.",
                            page=pm.number,
                            bbox=block.placed_bbox,
                            element_id=block.id,
                            translated_text=mk.plain_text(
                                block.translation or "", block.protected, {}
                            )[:300],
                        )
                    )
                ratio = len(block.translation or "") / max(len(block.source_markup), 1)
                if (ratio < 0.4 or ratio > 2.2) and len(block.source_markup) > 40:
                    res.issues.append(
                        Issue(
                            Severity.WARNING,
                            "length",
                            f"The translation is {ratio:.0%} of the source length; check for omissions or additions.",
                            page=pm.number,
                            bbox=block.bbox,
                            element_id=block.id,
                            source_text=block.source_text[:300],
                            translated_text=mk.plain_text(
                                block.translation or "", block.protected, {}
                            )[:300],
                        )
                    )
                if block.source == "native":
                    src_words = {
                        w.lower() for w in words_of(block.source_text) if len(w) > 3
                    }
                    expected_set = set(expected)
                    pt_left = [
                        w
                        for w in words_out
                        if overlap_ratio(tuple(w[:4]), block.bbox) > 0.8
                        and (wl := re.sub(r"[^\w-]", "", w[4].lower())) in src_words
                        and wl not in expected_set
                    ]
                    if len(pt_left) > max(2, 0.3 * len(src_words)):
                        leftover += 1
                        res.issues.append(
                            Issue(
                                Severity.ERROR,
                                "untranslated",
                                "Original Portuguese glyphs are still present underneath the translation.",
                                page=pm.number,
                                bbox=block.bbox,
                                element_id=block.id,
                            )
                        )
        failed = sum(1 for b in model.blocks() if b.status == BlockStatus.FAILED)
        if missing or leftover:
            check.status = "fail"
        elif failed:
            check.status = "warn"
        check.details = (
            f"{missing} block(s) with missing words, {leftover} with leftover source text, "
            f"{failed} block(s) kept in Portuguese because translation failed"
        )

    def _bounds(self, src, out, model, check, res) -> None:
        bad = 0
        for pm in model.pages:
            for block in pm.blocks:
                b = block.placed_bbox
                if not b or block.rotation:
                    continue
                if (
                    b[0] < -1
                    or b[1] < -1
                    or b[2] > pm.width + 1
                    or b[3] > pm.height + 1
                ):
                    bad += 1
                    res.issues.append(
                        Issue(
                            Severity.ERROR,
                            "overflow",
                            "Translated text extends beyond the page boundary.",
                            page=pm.number,
                            bbox=b,
                            element_id=block.id,
                        )
                    )
        check.status = "fail" if bad else "pass"
        check.details = (
            f"{bad} element(s) outside the page"
            if bad
            else "all text within page boundaries"
        )

    def _overlaps(self, src, out, model, check, res) -> None:
        count = 0
        for pm in model.pages:
            placed = [
                (b, b.placed_bbox)
                for b in pm.blocks
                if b.placed_bbox and not b.rotation
            ]
            fixed = [
                (b, b.bbox) for b in pm.blocks if b.status != BlockStatus.TRANSLATED
            ]
            for i, (a, ra) in enumerate(placed):
                ra_tight = expand_rect(ra, -0.5)
                for b, rb in placed[i + 1 :] + fixed:
                    if b is a:
                        continue
                    inter = intersection_area(ra_tight, rb)
                    if inter <= 0:
                        continue
                    if (
                        inter / max(min(rect_area(ra_tight), rect_area(rb)), 1e-6)
                        > 0.12
                    ):
                        count += 1
                        res.issues.append(
                            Issue(
                                Severity.WARNING,
                                "overlap",
                                "Translated text overlaps another element.",
                                page=pm.number,
                                bbox=ra,
                                element_id=a.id,
                                suggestion="Review this area; the translation may need shortening.",
                            )
                        )
                        break
        check.status = "warn" if count else "pass"
        check.details = (
            f"{count} overlapping element(s)" if count else "no overlapping text"
        )

    def _numbering(self, src, out, model, check, res) -> None:
        problems = 0
        for block in model.blocks():
            if block.kind not in (
                ElementKind.HEADING,
                ElementKind.PAGE_NUMBER,
                ElementKind.CAPTION,
                ElementKind.LIST_ITEM,
            ):
                continue
            if block.status != BlockStatus.TRANSLATED or not block.translation:
                continue
            src_nums = _NUM_RE.findall(block.source_text)
            out_nums = _NUM_RE.findall(
                mk.plain_text(block.translation, block.protected, {})
            )
            if src_nums != out_nums and sorted(src_nums) != sorted(out_nums):
                problems += 1
                res.issues.append(
                    Issue(
                        Severity.WARNING,
                        "numbering",
                        f"Numbering changed: {' '.join(src_nums)} -> {' '.join(out_nums)}.",
                        page=block.page,
                        bbox=block.bbox,
                        element_id=block.id,
                        source_text=block.source_text[:200],
                    )
                )
        check.status = "warn" if problems else "pass"
        check.details = (
            f"{problems} heading/caption/page number(s) with changed numbering"
            if problems
            else "section numbers, list enumerators and page numbers unchanged"
        )

    def _terminology(self, src, out, model, check, res) -> None:
        glossary = self.engine.glossary
        violations = 0
        used = set()
        for block in model.blocks():
            if block.status != BlockStatus.TRANSLATED or not block.translation:
                continue
            target = mk.plain_text(block.translation, block.protected, {})
            for entry in glossary.match(block.source_text):
                if entry.strict:
                    used.add(entry.pt)
            missing = glossary.check_translation(block.source_text, target)
            if missing:
                violations += 1
        strict = len(glossary.strict_entries())
        check.status = "warn" if violations else "pass"
        check.details = (
            f"{len(used)} mandatory glossary term(s) used; {violations} segment(s) deviate from the glossary"
            f" ({strict} mandatory terms in total)"
        )

    def _allowed_words(self, model: DocumentModel) -> frozenset[str]:
        allowed = set(self.engine.glossary.kept_terms())
        for block in model.blocks():
            if block.status in (BlockStatus.KEPT,):
                for w in words_of(block.text):
                    allowed.add(w.lower())
            # Proper names stay as written.
            for w in re.findall(r"(?<=[a-zà-ÿ,;] )[A-ZÀ-Ý][\wÀ-ÿ]+", block.source_text):
                allowed.add(w.lower())
        return frozenset(allowed)

    def _residual_text(self, src, out, model, check, res) -> None:
        allowed = self._allowed_words(model)
        lexicon = self.engine.glossary.portuguese_lexicon()
        failed_rects = [
            (b.page, b.bbox) for b in model.blocks() if b.status == BlockStatus.FAILED
        ]
        found = 0
        for pm in model.pages:
            words = out[pm.number].get_text("words")
            lines: dict[tuple, list] = {}
            for w in words:
                lines.setdefault((w[5], w[6]), []).append(w)
            for ws in lines.values():
                text = " ".join(w[4] for w in ws)
                if not looks_untranslated(text, allowed, lexicon):
                    continue
                bbox = (
                    min(w[0] for w in ws),
                    min(w[1] for w in ws),
                    max(w[2] for w in ws),
                    max(w[3] for w in ws),
                )
                if any(
                    p == pm.number and overlap_ratio(bbox, r) > 0.6
                    for p, r in failed_rects
                ):
                    continue  # already reported as a failed translation
                found += 1
                residue = portuguese_residue(text, allowed, lexicon)
                owner = next(
                    (
                        b
                        for b in pm.blocks
                        if b.placed_bbox
                        and overlap_ratio(bbox, b.placed_bbox) > 0.6
                        and b.status == BlockStatus.TRANSLATED
                    ),
                    None,
                )
                if owner is not None:
                    res.retranslate[owner.id] = (
                        "The previous English version still contained Portuguese words ("
                        + ", ".join(residue[:6])
                        + "). Translate every word into English."
                    )
                res.issues.append(
                    Issue(
                        Severity.WARNING,
                        "untranslated",
                        "Possible untranslated Portuguese: " + ", ".join(residue[:8]),
                        page=pm.number,
                        bbox=bbox,
                        element_id=owner.id if owner else None,
                        translated_text=text[:300],
                    )
                )
        check.status = "warn" if found else "pass"
        check.details = (
            f"{found} line(s) with Portuguese words"
            if found
            else "no Portuguese found in the text layer"
        )

    def _residual_images(self, src, out, model, check, res) -> None:
        if not (self.settings.qa_ocr_images and self.ocr.available()):
            check.status = "skipped"
            check.details = "OCR not available"
            return
        allowed = self._allowed_words(model)
        lexicon = self.engine.glossary.portuguese_lexicon()
        found = 0
        scanned = 0
        targets: list[tuple[int, Rect]] = []
        for pm in model.pages:
            if pm.strategy == PageStrategy.SCANNED:
                targets.append((pm.number, (0, 0, pm.width, pm.height)))
                continue
            for img in pm.images:
                if (
                    img.status in ("skipped",)
                    or rect_area(img.bbox) < self.settings.min_image_area_pt
                ):
                    continue
                targets.append((pm.number, img.bbox))
            if pm.text_chars == 0 and pm.drawings_count > 30 and not pm.images:
                targets.append(
                    (pm.number, (0, 0, pm.width, pm.height))
                )  # outlined text
        for k, (pno, rect) in enumerate(targets):
            self.progress(
                k / max(len(targets), 1), "Scanning images for Portuguese text"
            )
            page = out[pno]
            pix = page.get_pixmap(clip=pymupdf.Rect(rect), dpi=220, alpha=False)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height, pix.width, pix.n
            )[:, :, :3]
            words = [
                w
                for w in self.ocr.recognize(arr, psm=11)
                if w.confidence >= self.settings.ocr_min_confidence
            ]
            scanned += 1
            sx = (rect[2] - rect[0]) / max(pix.width, 1)
            sy = (rect[3] - rect[1]) / max(pix.height, 1)
            for line in group_into_lines(words):
                text = " ".join(w.text for w in line)
                if not looks_untranslated(text, allowed, lexicon):
                    continue
                lb = line_bbox(line)
                bbox = (
                    rect[0] + lb[0] * sx,
                    rect[1] + lb[1] * sy,
                    rect[0] + lb[2] * sx,
                    rect[1] + lb[3] * sy,
                )
                # Text layer lines are handled by the text pass.
                tl = page.get_text("text", clip=pymupdf.Rect(bbox)).strip()
                if tl and looks_untranslated(tl, allowed, lexicon):
                    continue
                found += 1
                res.issues.append(
                    Issue(
                        Severity.WARNING,
                        "untranslated_image_text",
                        "Portuguese text detected inside an image: "
                        + ", ".join(portuguese_residue(text, allowed, lexicon)[:8]),
                        page=pno,
                        bbox=bbox,
                        translated_text=text[:200],
                        suggestion="This text could not be replaced automatically; edit the image manually.",
                    )
                )
        check.status = "warn" if found else "pass"
        check.details = (
            f"{found} Portuguese text line(s) found in {scanned} image region(s)"
            if found
            else f"no Portuguese text found in {scanned} image/scan region(s)"
        )

    def _llm_review(self, src, out, model, check, res) -> None:
        blocks = [
            b
            for b in model.blocks()
            if b.status == BlockStatus.TRANSLATED
            and b.translation
            and len(b.source_text) > 3
        ]
        pairs = [(b.id, b.source_markup, b.translation) for b in blocks]
        by_id = {b.id: b for b in blocks}
        findings = []
        batch = 30
        ctx = self.engine.context
        for k in range(0, len(pairs), batch):
            self.progress(k / max(len(pairs), 1), "Reviewing translation accuracy")
            try:
                findings.extend(
                    self.engine.translator.review(pairs[k : k + batch], ctx)
                )
            except Exception as exc:
                log.warning("review batch failed: %s", exc)
        fixed = 0
        for f in findings:
            block = by_id.get(f.id)
            if block is None:
                continue
            severity = {"critical": Severity.ERROR, "major": Severity.WARNING}.get(
                f.severity, Severity.INFO
            )
            can_fix = (
                self.review_mode == QAReviewMode.FIX
                and f.suggestion
                and f.severity in ("major", "critical")
                and not mk.validate_structure(
                    block.source_markup,
                    f.suggestion,
                    block.styles,
                    block.protected,
                    set(block.math),
                )
            )
            if can_fix:
                res.corrections[block.id] = f.suggestion
                fixed += 1
            res.issues.append(
                Issue(
                    Severity.INFO if can_fix else severity,
                    "translation_quality",
                    f"Reviewer ({f.category}): {f.explanation}",
                    page=block.page,
                    bbox=block.placed_bbox or block.bbox,
                    element_id=block.id,
                    source_text=block.source_text[:300],
                    translated_text=mk.plain_text(
                        block.translation, block.protected, {}
                    )[:300],
                    suggestion=(
                        mk.plain_text(f.suggestion, block.protected, {})
                        if f.suggestion
                        else None
                    ),
                    auto_fixed=bool(can_fix),
                )
            )
        serious = sum(1 for f in findings if f.severity in ("major", "critical"))
        check.status = "warn" if serious - fixed > 0 else "pass"
        check.details = f"{len(pairs)} segments reviewed; {len(findings)} finding(s), {fixed} corrected automatically"


# ------------------------------------------------------------------ helpers
def _fingerprint(page: pymupdf.Page) -> tuple:
    imgs = tuple(
        sorted(tuple(round(v) for v in x["bbox"]) for x in page.get_image_info())
    )
    return (imgs, round(page.rect.width), round(page.rect.height))


def _iou(a: Rect, b: Rect) -> float:
    inter = intersection_area(a, b)
    union = rect_area(a) + rect_area(b) - inter
    return inter / union if union > 0 else 0.0


def _raster_diff(a: pymupdf.Page, b: pymupdf.Page, rect: Rect) -> float:
    clip = pymupdf.Rect(expand_rect(rect, 0.5))
    pa = a.get_pixmap(clip=clip, dpi=144, colorspace=pymupdf.csGRAY, alpha=False)
    pb = b.get_pixmap(clip=clip, dpi=144, colorspace=pymupdf.csGRAY, alpha=False)
    if (pa.width, pa.height) != (pb.width, pb.height) or pa.width == 0:
        return 255.0
    xa = np.frombuffer(pa.samples, np.uint8).astype(np.int16)
    xb = np.frombuffer(pb.samples, np.uint8).astype(np.int16)
    return float(np.mean(np.abs(xa - xb)))
