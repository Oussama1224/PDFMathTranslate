"""Translation of text embedded in raster images (charts, diagrams, screenshots).

Workflow per image:

1. decode the image XObject (with its soft mask);
2. OCR it (horizontal text, plus 90°-rotated passes for vertical axis titles);
3. group words into labels and decide which are Portuguese;
4. after translation: erase the original text pixels (flat backgrounds are
   filled, textured ones in-painted) and draw the English label with matching
   size, colour, weight and anchoring;
5. write the image back in place (same object, same placement).

Shapes, data marks, axes and icons are never modified: only pixels that
belong to recognised text are touched.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pymupdf

from pt2en.config import Settings
from pt2en.layout.math import unicode_is_math
from pt2en.model import (
    BlockStatus,
    ImageElement,
    ImageTextBox,
    Issue,
    OCRWord,
    Severity,
    union_rects,
)
from pt2en.ocr.base import OCREngine, group_into_lines, group_lines_into_boxes
from pt2en.translation.language import is_probably_english, looks_untranslated, score
from pt2en.visual import text_render as tr
from pt2en.visual.image_io import ImageData, encode_png, read_image, write_image

log = logging.getLogger(__name__)


@dataclass
class _ImageWork:
    data: ImageData
    upscale: float  # factor applied to the stored pixels before editing
    words: dict[str, list[OCRWord]] = field(default_factory=dict)


def _is_textual(text: str) -> bool:
    letters = sum(ch.isalpha() for ch in text)
    return letters >= 2 and letters >= 0.4 * len(text.replace(" ", ""))


def _mathy(text: str) -> bool:
    chars = [c for c in text if not c.isspace()]
    return bool(chars) and sum(unicode_is_math(c) for c in chars) / len(chars) > 0.25


class ImageTextTranslator:
    def __init__(self, ocr: OCREngine, settings: Settings):
        self.ocr = ocr
        self.settings = settings
        self._work: dict[str, _ImageWork] = {}
        self._done_xrefs: dict[int, str] = {}

    # ----------------------------------------------------------- analyse
    def analyze(
        self,
        doc: pymupdf.Document,
        img: ImageElement,
        page_w: float,
        page_h: float,
        issues: list[Issue],
    ) -> None:
        s = self.settings
        bw, bh = img.bbox[2] - img.bbox[0], img.bbox[3] - img.bbox[1]
        if (
            min(img.width, img.height) < s.min_image_side_px
            or bw * bh < s.min_image_area_pt
        ):
            img.status = "skipped"
            return
        if img.xref and img.xref in self._done_xrefs:
            img.status = "shared"
            img.notes.append(f"same image as {self._done_xrefs[img.xref]}")
            return
        if not self.ocr.available():
            img.status = "skipped"
            return
        data = read_image(doc, img.xref) if img.xref else None
        if data is None:
            if img.xref:
                img.status = "skipped"
                img.notes.append("unsupported image encoding")
                return
            # Inline image: render its area from the page instead.
            data = self._render_inline(doc, img)
            if data is None:
                img.status = "skipped"
                return
        if img.xref:
            self._done_xrefs[img.xref] = img.id

        h, w = data.rgb.shape[:2]
        dpi = w / max(bw / 72.0, 0.01)
        upscale = 1.0
        if dpi < 150 and max(w, h) < 2500:
            upscale = float(min(3, math.ceil(150 / max(dpi, 1))))
        work = _ImageWork(data=data, upscale=upscale)

        ocr_factor = 3.0 if max(w, h) < 600 else 2.0 if max(w, h) < 1600 else 1.0
        gray = cv2.cvtColor(data.rgb, cv2.COLOR_RGB2GRAY)
        if data.alpha is not None:
            # Composite transparent areas on white so OCR sees the text.
            a = data.alpha.astype(np.float32) / 255.0
            gray = (gray * a + 255 * (1 - a)).astype(np.uint8)
        big = (
            cv2.resize(
                gray, None, fx=ocr_factor, fy=ocr_factor, interpolation=cv2.INTER_CUBIC
            )
            if ocr_factor > 1
            else gray
        )

        min_conf = s.ocr_min_confidence
        horizontal = [
            self._scale_word(wd, 1 / ocr_factor)
            for wd in self.ocr.recognize(big, psm=11)
            if self._plausible_word(wd, min_conf * 0.8)
        ]
        horizontal = [wd for wd in horizontal if not _is_shape(gray, wd.bbox)]
        boxes: list[ImageTextBox] = []
        lines = group_into_lines(horizontal)
        for group in group_lines_into_boxes(lines):
            boxes.append(self._make_box(img, group, 0, len(boxes)))

        # Vertical labels (axis titles): OCR the image rotated both ways.
        H, W = gray.shape
        for rotation, rot_code in (
            (90, cv2.ROTATE_90_CLOCKWISE),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ):
            rotated = cv2.rotate(big, rot_code)
            found = [
                wd
                for wd in self.ocr.recognize(rotated, psm=11)
                if wd.confidence >= min_conf + 10
                and self._plausible_word(wd, min_conf + 10)
            ]
            if not found:
                continue
            lines_r = group_into_lines(
                [self._scale_word(wd, 1 / ocr_factor) for wd in found]
            )
            lines_r = [
                ln
                for ln in lines_r
                if any(sum(c.isalpha() for c in w.text) >= 3 for w in ln)
            ]
            for group in group_lines_into_boxes(lines_r):
                mapped = [
                    [self._unrotate(wd, rotation, W, H) for wd in line]
                    for line in group
                ]
                bbox = union_rects(wd.bbox for line in mapped for wd in line)
                if any(_overlaps(bbox, b.bbox) for b in boxes):
                    continue
                box = self._make_box(img, mapped, rotation, len(boxes))
                box.text = " ".join(" ".join(wd.text for wd in line) for line in group)
                boxes.append(box)

        for box in boxes:
            text = box.text
            pt_like = score(text).pt >= 0.8 or looks_untranslated(text)
            if not _is_textual(text) or _mathy(text):
                box.translatable = False
            elif box.confidence < min_conf:
                box.translatable = False
                if pt_like:
                    issues.append(
                        Issue(
                            Severity.WARNING,
                            "image_text",
                            f"Image text '{text[:60]}' looks Portuguese but could not be read reliably "
                            f"(OCR confidence {box.confidence:.0f}%); it was left unchanged.",
                            page=img.page,
                            bbox=self.page_rect_of(img, box.bbox, work),
                            element_id=box.id,
                            source_text=text,
                            suggestion="Edit this image manually.",
                        )
                    )
            elif not pt_like and (is_probably_english(text) or box.confidence < 70):
                box.translatable = False
            else:
                box.translatable = True
        img.text_boxes = boxes
        img.has_text = any(b.translatable for b in boxes)
        if img.has_text:
            self._work[img.id] = work
        img.status = "analysed"

    @staticmethod
    def _plausible_word(wd: OCRWord, min_conf: float) -> bool:
        text = wd.text.strip()
        if wd.confidence < min_conf or not any(c.isalnum() for c in text):
            return False
        if len(text) <= 2 and not text.isdigit() and wd.confidence < 70:
            return False  # icons and bullets misread as letters
        return True

    @staticmethod
    def _scale_word(wd: OCRWord, f: float) -> OCRWord:
        x0, y0, x1, y1 = wd.bbox
        return OCRWord(
            wd.text,
            (x0 * f, y0 * f, x1 * f, y1 * f),
            wd.confidence,
            wd.block,
            wd.paragraph,
            wd.line,
        )

    @staticmethod
    def _unrotate(wd: OCRWord, rotation: int, W: int, H: int) -> OCRWord:
        x0, y0, x1, y1 = wd.bbox
        if rotation == 90:  # image was rotated clockwise for OCR
            box = (y0, H - x1, y1, H - x0)
        else:  # counter-clockwise
            box = (W - y1, x0, W - y0, x1)
        return OCRWord(wd.text, box, wd.confidence, wd.block, wd.paragraph, wd.line)

    def _make_box(
        self, img: ImageElement, lines: list[list[OCRWord]], rotation: int, idx: int
    ) -> ImageTextBox:
        words = [w for line in lines for w in line]
        return ImageTextBox(
            id=f"{img.id}-t{idx}",
            text=" ".join(" ".join(w.text for w in line) for line in lines),
            bbox=union_rects(w.bbox for w in words),
            lines=lines,
            confidence=float(np.mean([w.confidence for w in words])) if words else 0.0,
            rotation=rotation,
        )

    @staticmethod
    def _render_inline(doc: pymupdf.Document, img: ImageElement) -> Optional[ImageData]:
        try:
            page = doc[img.page]
            pix = page.get_pixmap(clip=pymupdf.Rect(img.bbox), dpi=200, alpha=False)
            arr = (
                np.frombuffer(pix.samples, np.uint8)
                .reshape(pix.height, pix.width, pix.n)[:, :, :3]
                .copy()
            )
            return ImageData(rgb=arr, alpha=None, gray=False, jpeg=False, smask=0)
        except Exception:  # pragma: no cover
            return None

    def page_rect_of(self, img: ImageElement, box, work: Optional[_ImageWork] = None):
        """Map a pixel box to page coordinates (axis-aligned placements)."""
        data = work or self._work.get(img.id)
        h, w = (data.data.rgb.shape[:2]) if data else (img.height, img.width)
        x0, y0, x1, y1 = img.bbox
        sx, sy = (x1 - x0) / max(w, 1), (y1 - y0) / max(h, 1)
        return (x0 + box[0] * sx, y0 + box[1] * sy, x0 + box[2] * sx, y0 + box[3] * sy)

    # ------------------------------------------------------------ render
    def render(
        self, doc: pymupdf.Document, img: ImageElement, issues: list[Issue]
    ) -> bool:
        work = self._work.get(img.id)
        if work is None:
            return False
        todo = [
            b
            for b in img.text_boxes
            if b.translatable
            and b.status == BlockStatus.TRANSLATED
            and b.translation
            and b.translation.strip().lower() != b.text.strip().lower()
        ]
        if not todo:
            return False
        src = work.data
        # Always start from the pristine pixels so a rebuild (QA repair round) is idempotent.
        data = ImageData(
            rgb=src.rgb.copy(),
            alpha=None if src.alpha is None else src.alpha.copy(),
            gray=src.gray,
            jpeg=src.jpeg,
            smask=src.smask,
        )
        u = work.upscale
        if u > 1:
            data.rgb = cv2.resize(
                data.rgb, None, fx=u, fy=u, interpolation=cv2.INTER_CUBIC
            )
            if data.alpha is not None:
                data.alpha = cv2.resize(
                    data.alpha, None, fx=u, fy=u, interpolation=cv2.INTER_CUBIC
                )
        rgb = data.rgb

        # 1) erase every translated label first (so neighbours don't block each other).
        styles = {}
        for box in todo:
            colors, bgs, heights = [], [], []
            for line in box.lines:
                if box.rotation:
                    heights.append(max(w.bbox[2] - w.bbox[0] for w in line) * u)
                else:
                    heights.append(max(w.bbox[3] - w.bbox[1] for w in line) * u)
            bold = self._looks_bold(
                rgb, box, u, float(np.median(heights)) if heights else 10.0
            )
            for line, lh in zip(box.lines, heights):
                for wd in line:
                    bx = tuple(int(round(v * u)) for v in wd.bbox)
                    color, bg = tr.erase_box(rgb, bx, lh)
                    colors.append(color)
                    bgs.append(bg)
            color = (
                np.median(np.array(colors), axis=0).astype(np.uint8)
                if colors
                else np.zeros(3, np.uint8)
            )
            bg = (
                bgs[0]
                if bgs
                else tr.Background(np.array([255, 255, 255], np.uint8), 0.0)
            )
            styles[box.id] = (
                color,
                bg,
                float(np.median(heights)) if heights else 10.0,
                bold,
            )

        # 2) draw the English labels.
        drawn: list[tuple[int, int, int, int]] = []
        for box in todo:
            color, bg, line_h, bold = styles[box.id]
            ok = self._draw_box(rgb, data.alpha, box, color, bg, line_h, u, drawn, bold)
            if not ok:
                issues.append(
                    Issue(
                        Severity.WARNING,
                        "image_text",
                        f"The English label '{box.translation[:60]}' had to be shrunk considerably to fit "
                        "the image; check readability.",
                        page=img.page,
                        bbox=self.page_rect_of(img, box.bbox),
                        element_id=box.id,
                        source_text=box.text,
                        translated_text=box.translation,
                    )
                )
        data.rgb = rgb
        try:
            if img.xref:
                write_image(doc, img.xref, data)
            else:
                page = doc[img.page]
                page.insert_image(
                    pymupdf.Rect(img.bbox),
                    stream=encode_png(data),
                    keep_proportion=False,
                )
        except Exception as exc:
            log.exception("could not write image %s", img.id)
            issues.append(
                Issue(
                    Severity.ERROR,
                    "image_text",
                    f"The translated image could not be written ({exc}); the original image was kept.",
                    page=img.page,
                    bbox=img.bbox,
                    element_id=img.id,
                )
            )
            return False
        img.status = "translated"
        return True

    def _draw_box(
        self,
        rgb,
        alpha,
        box: ImageTextBox,
        color,
        bg,
        line_h: float,
        u: float,
        drawn,
        bold: bool,
    ) -> bool:
        x0, y0, x1, y1 = (int(round(v * u)) for v in box.bbox)
        text = " ".join(box.translation.split())
        source_lines = [" ".join(w.text for w in line) for line in box.lines]
        n_lines = max(1, len(source_lines))
        ref = max(source_lines, key=len) if source_lines else box.text
        size = tr.font_px_for_height(ref, line_h, bold)
        if box.rotation:
            length = y1 - y0
            avail = int(length * 1.5)
            centre = ((x0 + x1) / 2, (y0 + y1) / 2)
            align, anchor = "center", centre[0]
            pitch = line_h * 1.25
            max_lines = n_lines
        else:
            left_room, right_room = tr.free_extent(
                rgb, (x0, y0, x1, y1), bg, int((x1 - x0) * 0.6 + line_h * 2), drawn
            )
            swatch = tr.has_swatch_left(rgb, (x0, y0, x1, y1), bg, line_h)
            centres = [
                ((min(w.bbox[0] for w in ln) + max(w.bbox[2] for w in ln)) / 2) * u
                for ln in box.lines
            ]
            lefts = [min(w.bbox[0] for w in ln) * u for ln in box.lines]
            if swatch or (
                n_lines > 1
                and max(lefts) - min(lefts) < line_h * 0.5
                and max(centres) - min(centres) > line_h
            ):
                align, anchor = "left", x0
                avail = (x1 - x0) + right_room
            else:
                align, anchor = "center", (x0 + x1) / 2
                half = (x1 - x0) / 2 + min(left_room, right_room)
                avail = int(2 * half)
            centre = ((x0 + x1) / 2, (y0 + y1) / 2)
            pitch = ((y1 - y0) / n_lines) if n_lines > 1 else line_h * 1.25
            max_lines = n_lines
        lines = None
        chosen = size
        ok = True
        for factor in [1.0 - 0.05 * k for k in range(0, 9)]:  # down to 60%
            px = max(4, int(size * factor))
            lines = tr.wrap_lines(text, tr.pil_font(px, bold), avail, max_lines)
            if lines:
                chosen = px
                break
        if not lines:
            for factor in (0.85, 0.75, 0.65, 0.55, 0.45):
                px = max(4, int(size * factor))
                lines = tr.wrap_lines(text, tr.pil_font(px, bold), avail, max_lines + 1)
                if lines:
                    chosen = px
                    ok = factor >= 0.6
                    break
        if not lines:
            chosen = max(4, int(size * 0.45))
            lines = [text]
            ok = False
        if len(lines) != n_lines and n_lines == 1:
            pitch = line_h * 1.15
        rect = tr.draw_lines(
            rgb,
            alpha,
            lines,
            tr.pil_font(chosen, bold),
            color,
            centre,
            pitch * (chosen / max(size, 1)) if len(lines) == n_lines else pitch,
            align,
            anchor,
            rotation=box.rotation,
        )
        drawn.append(rect)
        box.notes.append(f"drawn at {chosen / max(size, 1):.0%} of the original size")
        return ok

    @staticmethod
    def _looks_bold(rgb, box: ImageTextBox, u: float, line_h: float) -> bool:
        x0, y0, x1, y1 = (int(round(v * u)) for v in box.bbox)
        if y1 <= y0 or x1 <= x0:
            return False
        crop = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y1, x0:x1]), cv2.COLOR_RGB2GRAY)
        if box.rotation:
            crop = cv2.rotate(
                crop,
                (
                    cv2.ROTATE_90_CLOCKWISE
                    if box.rotation == 90
                    else cv2.ROTATE_90_COUNTERCLOCKWISE
                ),
            )
        text = max(
            (" ".join(w.text for w in ln) for ln in box.lines),
            key=len,
            default=box.text,
        )
        return tr.looks_bold(crop, text, line_h, len(box.lines))


def _is_shape(gray: np.ndarray, bbox) -> bool:
    """A 'word' whose box is mostly one solid fill is a shape (legend swatch, bar)."""
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    crop = gray[max(0, y0) : y1, max(0, x0) : x1]
    if crop.size < 16:
        return False
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]).astype(
        np.int16
    )
    bg = int(np.median(border))
    ink = np.abs(crop.astype(np.int16) - bg) > 40
    return bool(ink.mean() > 0.85)


def _overlaps(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])
