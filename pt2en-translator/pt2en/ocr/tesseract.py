"""Tesseract OCR engine (via pytesseract)."""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional

import numpy as np

from pt2en.model import OCRWord
from pt2en.ocr.base import OCREngine

log = logging.getLogger(__name__)


class TesseractOCR(OCREngine):
    name = "tesseract"

    def __init__(self, languages: str = "por+eng", cmd: Optional[str] = None):
        import pytesseract

        self._pt = pytesseract
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        self._available: Optional[bool] = None
        self._lock = threading.Lock()
        self.languages = languages

    def available(self) -> bool:
        if self._available is None:
            try:
                self._pt.get_tesseract_version()
                installed = set(self._pt.get_languages(config=""))
                wanted = [lang for lang in self.languages.split("+") if lang]
                usable = [lang for lang in wanted if lang in installed]
                if not usable:
                    log.warning(
                        "No requested Tesseract languages installed (%s)",
                        self.languages,
                    )
                    usable = ["eng"] if "eng" in installed else []
                elif len(usable) < len(wanted):
                    log.warning(
                        "Tesseract languages missing: %s",
                        ", ".join(sorted(set(wanted) - set(usable))),
                    )
                self.languages = "+".join(usable)
                self._available = bool(usable)
            except Exception as exc:
                log.warning("Tesseract is not available: %s", exc)
                self._available = False
        return self._available

    def recognize(
        self, image: np.ndarray, *, psm: int = 3, languages: Optional[str] = None
    ) -> list[OCRWord]:
        if not self.available():
            return []
        config = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
        data = self._pt.image_to_data(
            image,
            lang=languages or self.languages,
            config=config,
            output_type=self._pt.Output.DICT,
        )
        words: list[OCRWord] = []
        for i, text in enumerate(data["text"]):
            text = (text or "").strip()
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if not text or conf < 0:
                continue
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            if w <= 0 or h <= 0:
                continue
            words.append(
                OCRWord(
                    text=text,
                    bbox=(float(x), float(y), float(x + w), float(y + h)),
                    confidence=conf,
                    block=int(data["block_num"][i]),
                    paragraph=int(data["par_num"][i]),
                    line=int(data["line_num"][i]),
                )
            )
        return words

    def detect_rotation(self, image: np.ndarray) -> int:
        if not self.available():
            return 0
        try:
            osd = self._pt.image_to_osd(image, config="--psm 0")
        except Exception:
            return 0
        m = re.search(r"Rotate:\s+(\d+)", osd)
        conf = re.search(r"Orientation confidence:\s+([\d.]+)", osd)
        if m and conf and float(conf.group(1)) >= 2.0:
            return int(m.group(1)) % 360
        return 0


def create_ocr(engine: str, languages: str, cmd: Optional[str] = None) -> OCREngine:
    from pt2en.ocr.base import NullOCR

    if engine == "tesseract":
        try:
            ocr = TesseractOCR(languages, cmd)
            if ocr.available():
                return ocr
        except ImportError:  # pragma: no cover
            log.warning("pytesseract not installed; OCR disabled")
    return NullOCR()
