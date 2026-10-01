"""Reading and in-place rewriting of PDF image XObjects.

Images are rewritten *in place* (same object number), so every page and form
that references an image picks up the translated version, and placement,
clipping and transforms stay exactly as in the source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pymupdf

log = logging.getLogger(__name__)


@dataclass
class ImageData:
    rgb: np.ndarray  # H x W x 3 uint8
    alpha: Optional[np.ndarray]  # H x W uint8 or None
    gray: bool
    jpeg: bool
    smask: int


def _pixmap_to_array(pix: pymupdf.Pixmap) -> np.ndarray:
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    return arr.copy()


def read_image(doc: pymupdf.Document, xref: int) -> Optional[ImageData]:
    try:
        if doc.xref_get_key(xref, "ImageMask")[1] == "true":
            return None  # stencil masks carry no colour information
        filt = doc.xref_get_key(xref, "Filter")[1]
        pix = pymupdf.Pixmap(doc, xref)
    except Exception as exc:
        log.debug("cannot read image %s: %s", xref, exc)
        return None
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    gray = pix.colorspace is not None and pix.colorspace.n == 1
    if pix.colorspace is None or pix.colorspace.n not in (1, 3):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        gray = False
    arr = _pixmap_to_array(pix)
    rgb = np.repeat(arr, 3, axis=2) if arr.shape[2] == 1 else arr[:, :, :3]

    alpha = None
    smask = 0
    try:
        key = doc.xref_get_key(xref, "SMask")
        if key[0] == "xref":
            smask = int(key[1].split()[0])
            apix = pymupdf.Pixmap(doc, smask)
            a = _pixmap_to_array(apix)[:, :, 0]
            if a.shape != rgb.shape[:2]:
                a = cv2.resize(
                    a, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
                )
            alpha = a
    except Exception as exc:
        log.debug("cannot read soft mask of %s: %s", xref, exc)
    return ImageData(rgb=rgb, alpha=alpha, gray=gray, jpeg="DCT" in filt, smask=smask)


def _set_image_keys(
    doc: pymupdf.Document, xref: int, w: int, h: int, colorspace: str
) -> None:
    doc.xref_set_key(xref, "Width", str(w))
    doc.xref_set_key(xref, "Height", str(h))
    doc.xref_set_key(xref, "ColorSpace", colorspace)
    doc.xref_set_key(xref, "BitsPerComponent", "8")
    for key in ("DecodeParms", "Decode", "Matte", "Intent"):
        doc.xref_set_key(xref, key, "null")


def write_image(doc: pymupdf.Document, xref: int, data: ImageData) -> None:
    rgb = data.rgb
    h, w = rgb.shape[:2]
    if data.gray:
        plane = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        colorspace = "/DeviceGray"
    else:
        plane = rgb
        colorspace = "/DeviceRGB"
    if data.jpeg:
        bgr = plane if data.gray else cv2.cvtColor(plane, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:  # pragma: no cover
            raise RuntimeError("JPEG encoding failed")
        doc.update_stream(xref, buf.tobytes(), compress=False)
        doc.xref_set_key(xref, "Filter", "/DCTDecode")
    else:
        doc.update_stream(xref, np.ascontiguousarray(plane).tobytes(), compress=True)
    _set_image_keys(doc, xref, w, h, colorspace)
    if data.smask and data.alpha is not None:
        alpha = data.alpha
        if alpha.shape != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        doc.update_stream(
            data.smask, np.ascontiguousarray(alpha).tobytes(), compress=True
        )
        _set_image_keys(doc, data.smask, w, h, "/DeviceGray")


def encode_png(data: ImageData) -> bytes:
    rgb = data.rgb
    if data.alpha is not None:
        rgba = np.dstack([rgb, data.alpha])
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    else:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:  # pragma: no cover
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
