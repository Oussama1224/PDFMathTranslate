"""Raster text utilities: background estimation, text erasing and redrawing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

SANS_FILES = {
    (False, False): [
        "LiberationSans-Regular.ttf",
        "DejaVuSans.ttf",
        "FreeSans.ttf",
        "Arial.ttf",
    ],
    (True, False): [
        "LiberationSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "FreeSansBold.ttf",
        "Arial Bold.ttf",
    ],
}
SERIF_FILES = {
    (False, False): ["LiberationSerif-Regular.ttf", "DejaVuSerif.ttf", "FreeSerif.ttf"],
    (True, False): [
        "LiberationSerif-Bold.ttf",
        "DejaVuSerif-Bold.ttf",
        "FreeSerifBold.ttf",
    ],
}
FONT_DIRS = [
    "/usr/share/fonts/truetype/liberation2",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/freefont",
    "/Library/Fonts",
    "C:/Windows/Fonts",
]


@lru_cache(maxsize=16)
def _font_path(bold: bool, serif: bool) -> Optional[str]:
    table = SERIF_FILES if serif else SANS_FILES
    for name in table[(bold, False)]:
        for d in FONT_DIRS:
            p = Path(d) / name
            if p.exists():
                return str(p)
    return None


@lru_cache(maxsize=256)
def pil_font(px: int, bold: bool = False, serif: bool = False) -> ImageFont.ImageFont:
    path = _font_path(bold, serif)
    if path:
        return ImageFont.truetype(path, max(4, int(px)))
    return ImageFont.load_default()


def ink_bbox(text: str, font: ImageFont.ImageFont) -> tuple[int, int, int, int]:
    img = Image.new("L", (1, 1))
    return ImageDraw.Draw(img).textbbox((0, 0), text, font=font)


def text_size(text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    x0, y0, x1, y1 = ink_bbox(text, font)
    return x1 - x0, y1 - y0


def font_px_for_height(
    text: str, height: float, bold: bool = False, serif: bool = False
) -> int:
    """Font pixel size whose rendering of ``text`` has the given ink height."""
    lo, hi = 4, 400
    target = max(4.0, height)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        _, h = text_size(text, pil_font(mid, bold, serif))
        if h <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


@dataclass
class Background:
    color: np.ndarray  # RGB
    std: float


def ring_background(
    rgb: np.ndarray, box: tuple[int, int, int, int], width: int = 3
) -> Background:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    ox0, oy0 = max(0, x0 - width), max(0, y0 - width)
    ox1, oy1 = min(w, x1 + width), min(h, y1 + width)
    outer = rgb[oy0:oy1, ox0:ox1].reshape(-1, 3).astype(np.float32)
    mask = np.ones((oy1 - oy0, ox1 - ox0), bool)
    mask[max(0, y0 - oy0) : max(0, y1 - oy0), max(0, x0 - ox0) : max(0, x1 - ox0)] = (
        False
    )
    ring = outer[mask.reshape(-1)]
    if ring.size == 0:
        ring = outer
    color = np.median(ring, axis=0)
    std = float(np.mean(np.std(ring, axis=0))) if len(ring) > 1 else 0.0
    return Background(color=color.astype(np.uint8), std=std)


def text_mask(
    rgb: np.ndarray, box: tuple[int, int, int, int], bg: Background
) -> np.ndarray:
    """Pixels inside ``box`` that differ clearly from the background colour."""
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1].astype(np.int16)
    dist = np.abs(crop - bg.color.astype(np.int16)).sum(axis=2)
    if dist.size == 0:
        return np.zeros((y1 - y0, x1 - x0), bool)
    thresh = max(60.0, bg.std * 3.0)
    otsu_src = np.clip(dist, 0, 765).astype(np.uint16)
    if otsu_src.max() > 0:
        scaled = (otsu_src * (255.0 / max(1, otsu_src.max()))).astype(np.uint8)
        t, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = max(thresh * 0.6, t * max(1, otsu_src.max()) / 255.0)
    return dist > thresh


def erase_box(
    rgb: np.ndarray, box: tuple[int, int, int, int], height: float
) -> tuple[np.ndarray, Background]:
    """Remove text pixels inside ``box``; returns the text colour and background."""
    h, w = rgb.shape[:2]
    pad = max(2, int(round(height * 0.18)))
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(w, x1 + pad),
        min(h, y1 + pad),
    )
    bg = ring_background(rgb, (x0, y0, x1, y1))
    mask = text_mask(rgb, (x0, y0, x1, y1), bg)
    core = rgb[y0:y1, x0:x1][mask]
    if core.size:
        dist = np.abs(core.astype(np.int16) - bg.color.astype(np.int16)).sum(axis=1)
        strong = core[dist >= np.percentile(dist, 60)]
        color = np.median(strong if len(strong) else core, axis=0).astype(np.uint8)
    else:
        color = np.array([0, 0, 0], np.uint8)
    full = np.zeros((h, w), np.uint8)
    full[y0:y1, x0:x1][mask] = 255
    k = max(1, int(round(height * 0.08)))
    full = cv2.dilate(full, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    if bg.std < 18:
        rgb[full > 0] = bg.color
    else:
        region = (
            slice(max(0, y0 - 8), min(h, y1 + 8)),
            slice(max(0, x0 - 8), min(w, x1 + 8)),
        )
        patch = cv2.inpaint(
            np.ascontiguousarray(rgb[region]),
            np.ascontiguousarray(full[region]),
            3,
            cv2.INPAINT_TELEA,
        )
        rgb[region] = patch
    return color, bg


def free_extent(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    bg: Background,
    limit: int,
    obstacles: list[tuple[int, int, int, int]],
) -> tuple[int, int]:
    """How far (px) text may extend to the left/right of ``box`` over plain background."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    y0, y1 = max(0, y0), min(h, y1)
    band = rgb[y0:y1].astype(np.int16)
    tol = max(45.0, bg.std * 3)

    def clear(col: int) -> bool:
        if col < 0 or col >= w:
            return False
        for o in obstacles:
            if o[0] <= col <= o[2] and o[1] < y1 and o[3] > y0:
                return False
        diff = np.abs(band[:, col] - bg.color.astype(np.int16)).sum(axis=1)
        return bool(np.mean(diff > tol) < 0.02)

    left = 0
    while left < limit and clear(x0 - left - 1):
        left += 1
    right = 0
    while right < limit and clear(x1 + right):
        right += 1
    return left, right


def has_swatch_left(
    rgb: np.ndarray, box: tuple[int, int, int, int], bg: Background, height: float
) -> bool:
    """A legend swatch/icon immediately left of a label means left anchoring."""
    x0, y0, x1, y1 = box
    sx0 = max(0, int(x0 - height * 1.8))
    if sx0 >= x0:
        return False
    crop = rgb[max(0, y0) : y1, sx0:x0].astype(np.int16)
    if crop.size == 0:
        return False
    diff = np.abs(crop - bg.color.astype(np.int16)).sum(axis=2)
    return bool(np.mean(diff > 90) > 0.15)


def wrap_lines(
    text: str, font: ImageFont.ImageFont, width: float, max_lines: int
) -> Optional[list[str]]:
    words = text.split()
    if not words:
        return []
    lines = [words[0]]
    for word in words[1:]:
        candidate = lines[-1] + " " + word
        if text_size(candidate, font)[0] <= width:
            lines[-1] = candidate
        else:
            lines.append(word)
    if len(lines) > max_lines or any(text_size(ln, font)[0] > width for ln in lines):
        return None
    return lines


def draw_lines(
    rgb: np.ndarray,
    alpha: Optional[np.ndarray],
    lines: list[str],
    font: ImageFont.ImageFont,
    color: np.ndarray,
    center: tuple[float, float],
    line_height: float,
    align: str,
    anchor_x: float,
    rotation: int = 0,
) -> tuple[int, int, int, int]:
    """Draw text lines; returns the drawn bounding box (pixels)."""
    widths = [text_size(ln, font)[0] for ln in lines]
    block_w = max(widths) if widths else 0
    block_h = line_height * len(lines)
    pad = int(line_height)
    canvas = Image.new("L", (int(block_w + 2 * pad), int(block_h + 2 * pad)), 0)
    draw = ImageDraw.Draw(canvas)
    for i, (ln, lw) in enumerate(zip(lines, widths)):
        if align == "center":
            x = pad + (block_w - lw) / 2
        elif align == "right":
            x = pad + block_w - lw
        else:
            x = pad
        bx0, by0, _, _ = ink_bbox(ln, font)
        ascent = (
            font.getmetrics()[0] if hasattr(font, "getmetrics") else line_height * 0.8
        )
        y = pad + i * line_height + (line_height - ascent) / 2 - 0
        draw.text((x - bx0, y), ln, font=font, fill=255)
    if rotation:
        canvas = canvas.rotate(rotation, expand=True, resample=Image.BICUBIC)
    mask = np.array(canvas)
    ys, xs = np.nonzero(mask > 8)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    mx0, my0, mx1, my1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    glyphs = mask[my0:my1, mx0:mx1]
    gh, gw = glyphs.shape
    cx, cy = center
    if rotation:
        left = int(round(cx - gw / 2))
    elif align == "left":
        left = int(round(anchor_x))
    elif align == "right":
        left = int(round(anchor_x - gw))
    else:
        left = int(round(cx - gw / 2))
    top = int(round(cy - gh / 2))
    h, w = rgb.shape[:2]
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(w, left + gw), min(h, top + gh)
    if sx1 <= sx0 or sy1 <= sy0:
        return (0, 0, 0, 0)
    g = (
        glyphs[sy0 - top : sy1 - top, sx0 - left : sx1 - left].astype(np.float32)
        / 255.0
    )
    region = rgb[sy0:sy1, sx0:sx1].astype(np.float32)
    region = (
        region * (1 - g[..., None])
        + color.astype(np.float32)[None, None, :] * g[..., None]
    )
    rgb[sy0:sy1, sx0:sx1] = np.clip(region, 0, 255).astype(np.uint8)
    if alpha is not None:
        a = alpha[sy0:sy1, sx0:sx1].astype(np.float32)
        alpha[sy0:sy1, sx0:sx1] = np.maximum(a, g * 255).astype(np.uint8)
    return (sx0, sy0, sx1, sy1)


def ink_fraction(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    frac = float((bw > 0).mean())
    return 1.0 - frac if frac > 0.5 else frac


def looks_bold(
    gray_crop: np.ndarray, text: str, line_h: float, lines: int = 1, serif: bool = False
) -> bool:
    """Compare the ink density of a text crop with a regular rendering of the same text."""
    if gray_crop.size == 0 or not text.strip():
        return False
    observed = ink_fraction(gray_crop)
    font = pil_font(font_px_for_height(text, line_h, serif=serif), serif=serif)
    w, h = text_size(text, font)
    canvas = Image.new("L", (w + 4, h + 4), 255)
    bx0, by0, _, _ = ink_bbox(text, font)
    ImageDraw.Draw(canvas).text((2 - bx0, 2 - by0), text, font=font, fill=0)
    ref = np.array(canvas)
    ys, xs = np.nonzero(ref < 200)
    if len(xs) == 0:
        return False
    reference = ink_fraction(ref[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])
    if lines > 1:
        reference *= 0.8  # inter-line gaps dilute multi-line boxes
    return reference > 0 and observed > reference * 1.25
