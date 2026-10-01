"""OCR engine interface and word-grouping helpers."""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from pt2en.model import OCRWord, Rect, union_rects


class OCREngine(abc.ABC):
    name = "base"

    @abc.abstractmethod
    def available(self) -> bool: ...

    @abc.abstractmethod
    def recognize(
        self, image: np.ndarray, *, psm: int = 3, languages: Optional[str] = None
    ) -> list[OCRWord]:
        """Recognise words in an RGB or grayscale image (pixel coordinates)."""

    def detect_rotation(self, image: np.ndarray) -> int:
        """Clockwise rotation (0/90/180/270) needed to make the text upright."""
        return 0


class NullOCR(OCREngine):
    name = "none"

    def available(self) -> bool:
        return False

    def recognize(self, image, *, psm=3, languages=None) -> list[OCRWord]:
        return []


def word_height(w: OCRWord) -> float:
    return w.bbox[3] - w.bbox[1]


def group_into_lines(words: list[OCRWord]) -> list[list[OCRWord]]:
    """Group words into text lines using geometry (robust for sparse layouts)."""
    remaining = sorted(words, key=lambda w: (w.bbox[0], w.bbox[1]))
    lines: list[list[OCRWord]] = []
    for w in remaining:
        h = word_height(w)
        placed = False
        for line in sorted(lines, key=lambda ln: abs(ln[-1].bbox[1] - w.bbox[1])):
            last = line[-1]
            lh = max(word_height(x) for x in line)
            overlap = min(last.bbox[3], w.bbox[3]) - max(last.bbox[1], w.bbox[1])
            gap = w.bbox[0] - last.bbox[2]
            if overlap > 0.5 * min(h, lh) and -0.3 * lh <= gap <= 1.2 * max(h, lh):
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w.bbox[0])
    return lines


def line_bbox(line: list[OCRWord]) -> Rect:
    return union_rects(w.bbox for w in line)


def group_lines_into_boxes(lines: list[list[OCRWord]]) -> list[list[list[OCRWord]]]:
    """Merge vertically adjacent, aligned lines into multi-line labels."""
    ordered = sorted(lines, key=lambda ln: (line_bbox(ln)[1], line_bbox(ln)[0]))
    boxes: list[list[list[OCRWord]]] = []
    for line in ordered:
        lb = line_bbox(line)
        lh = lb[3] - lb[1]
        target = None
        for box in boxes:
            last = line_bbox(box[-1])
            bh = last[3] - last[1]
            vgap = lb[1] - last[3]
            if not (-0.2 * bh <= vgap <= 0.6 * max(bh, lh)):
                continue
            if abs(lh - bh) > 0.35 * max(lh, bh):
                continue
            left_aligned = abs(lb[0] - last[0]) < 0.8 * bh
            centred = abs((lb[0] + lb[2]) / 2 - (last[0] + last[2]) / 2) < 0.8 * bh
            overlap = min(lb[2], last[2]) - max(lb[0], last[0])
            if (left_aligned or centred) and overlap > 0:
                target = box
                break
        if target is None:
            boxes.append([line])
        else:
            target.append(line)
    return boxes
