"""Tesseract OCR engine (via pytesseract)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from pt2en.model import OCRWord
from pt2en.ocr.base import OCREngine

log = logging.getLogger(__name__)

# Standard install locations, used when tesseract is not on PATH (the Windows
# installer does not add it to PATH by default).
COMMON_LOCATIONS = [
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/usr/bin/tesseract",
]
# Folders checked on every Windows drive (C:\Program Files\Tesseract-OCR, D:\Tesseract, ...)
WINDOWS_FOLDERS = [
    r"Program Files\Tesseract-OCR",
    r"Program Files (x86)\Tesseract-OCR",
    "Tesseract-OCR",
    "Tesseract",
    r"Program Files\Tesseract",
]
_ESCAPED = {
    "\t": "\\t",
    "\n": "\\n",
    "\r": "\\r",
    "\b": "\\b",
    "\f": "\\f",
    "\a": "\\a",
    "\v": "\\v",
}


def _clean_cmd(cmd: str) -> str:
    """Undo quoting and escape mangling.

    A quoted ``"C:\\Tesseract\\tesseract.exe"`` in .env turns ``\\t`` into a tab.
    """
    cmd = cmd.strip().strip('"').strip("'").strip()
    for ch, repl in _ESCAPED.items():
        cmd = cmd.replace(ch, repl)
    return cmd


def _executable_in(path: Path) -> Optional[str]:
    """``path`` itself if it is a file, else tesseract(.exe) inside that folder."""
    try:
        if path.is_file():
            return str(path)
        if path.is_dir():
            for name in ("tesseract.exe", "tesseract"):
                if (path / name).is_file():
                    return str(path / name)
    except OSError:  # unreadable drive or network share
        pass
    return None


def _windows_candidates() -> list[Path]:
    if os.name != "nt":
        return []
    drives = [f"{letter}:\\" for letter in "CDEFGH"]
    if hasattr(os, "listdrives"):  # Python 3.12+
        try:
            drives = os.listdrives()
        except OSError:
            pass
    return [Path(d) / folder for d in drives for folder in WINDOWS_FOLDERS]


def find_tesseract(cmd: Optional[str] = None) -> tuple[Optional[str], str]:
    """Locate the tesseract executable; returns (path, problem description)."""
    if cmd:
        cleaned = _clean_cmd(cmd)
        found = _executable_in(Path(cleaned)) or shutil.which(cleaned)
        if found:
            return found, ""
        return None, (
            f"PT2EN_TESSERACT_CMD is set to '{cleaned}', but no tesseract "
            "executable exists there (use the full path to tesseract.exe or "
            "its folder)."
        )
    found = shutil.which("tesseract")
    if found:
        return found, ""
    for path in [Path(p) for p in COMMON_LOCATIONS if p] + _windows_candidates():
        found = _executable_in(path)
        if found:
            return found, ""
    return None, (
        "Tesseract was not found. Install it (Windows: UB Mannheim installer "
        "with the Portuguese language data; macOS: brew install tesseract "
        "tesseract-lang; Linux: apt install tesseract-ocr tesseract-ocr-por) "
        "or set PT2EN_TESSERACT_CMD to its full path."
    )


class TesseractOCR(OCREngine):
    name = "tesseract"

    def __init__(self, languages: str = "por+eng", cmd: Optional[str] = None):
        import pytesseract

        self._pt = pytesseract
        self.path, self.reason = find_tesseract(cmd)
        if self.path:
            pytesseract.pytesseract.tesseract_cmd = self.path
        self._available: Optional[bool] = None if self.path else False
        self._lock = threading.Lock()
        self.languages = languages

    def available(self) -> bool:
        if self._available is None:
            try:
                self._pt.get_tesseract_version()
                installed = set(self._pt.get_languages(config=""))
                wanted = [lang for lang in self.languages.split("+") if lang]
                usable = [lang for lang in wanted if lang in installed]
                missing = sorted(set(wanted) - set(usable))
                if not usable:
                    usable = ["eng"] if "eng" in installed else []
                if missing:
                    self.reason = (
                        "Tesseract language data missing: "
                        + ", ".join(missing)
                        + (
                            " (Portuguese text will be recognised poorly; "
                            "reinstall Tesseract with the Portuguese language)"
                            if "por" in missing
                            else ""
                        )
                    )
                    log.warning(self.reason)
                if not usable:
                    self.reason = (
                        f"Tesseract at {self.path} has no usable language data."
                    )
                self.languages = "+".join(usable)
                self._available = bool(usable)
            except Exception as exc:
                self.reason = f"Tesseract at {self.path} could not be run: {exc}"
                log.warning(self.reason)
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
        except ImportError:  # pragma: no cover
            return NullOCR("The pytesseract package is not installed.")
        if not ocr.available():
            log.warning("OCR unavailable: %s", ocr.reason)
        return ocr  # callers check available(); keeping it preserves the reason
    return NullOCR()
