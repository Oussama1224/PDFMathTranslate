"""Font selection for re-typeset text.

Preference order for a run of text:

1. the original embedded font of the run, when it contains real glyphs for
   every character of the English text (subset fonts often miss k/w/y);
2. a metric-compatible open font for well-known families (Carlito for
   Calibri, Caladea for Cambria, Liberation for Arial/Times/Courier);
3. a generic serif / sans / mono family matching the original style;
4. MuPDF's built-in Base-14 fonts (always available);
5. a broad-coverage Unicode font for any glyph still missing.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf

from pt2en.model import TextStyle

log = logging.getLogger(__name__)

DEFAULT_FONT_DIRS = [
    "/usr/share/fonts/truetype/crosextra",
    "/usr/share/fonts/truetype/liberation2",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/opentype/noto",
    "/usr/share/fonts/TTF",
    "/Library/Fonts",
    "C:/Windows/Fonts",
]

FAMILY_FILES = {
    "carlito": (
        "Carlito-Regular.ttf",
        "Carlito-Bold.ttf",
        "Carlito-Italic.ttf",
        "Carlito-BoldItalic.ttf",
    ),
    "caladea": (
        "Caladea-Regular.ttf",
        "Caladea-Bold.ttf",
        "Caladea-Italic.ttf",
        "Caladea-BoldItalic.ttf",
    ),
    "sans": (
        "LiberationSans-Regular.ttf",
        "LiberationSans-Bold.ttf",
        "LiberationSans-Italic.ttf",
        "LiberationSans-BoldItalic.ttf",
    ),
    "serif": (
        "LiberationSerif-Regular.ttf",
        "LiberationSerif-Bold.ttf",
        "LiberationSerif-Italic.ttf",
        "LiberationSerif-BoldItalic.ttf",
    ),
    "mono": (
        "LiberationMono-Regular.ttf",
        "LiberationMono-Bold.ttf",
        "LiberationMono-Italic.ttf",
        "LiberationMono-BoldItalic.ttf",
    ),
}
BASE14 = {
    "sans": ("helv", "hebo", "heit", "hebi"),
    "serif": ("tiro", "tibo", "tiit", "tibi"),
    "mono": ("cour", "cobo", "coit", "cobi"),
}
UNICODE_FALLBACK_FILES = ("DejaVuSans.ttf", "NotoSans-Regular.ttf", "FreeSans.ttf")

_FAMILY_RULES = [
    (re.compile(r"calibri|carlito", re.I), "carlito"),
    (re.compile(r"cambria(?!.*math)|caladea", re.I), "caladea"),
    (re.compile(r"courier|consolas|mono|cousine|cmtt|lmmono|menlo|code", re.I), "mono"),
    (
        re.compile(
            r"arial|helvetica|arimo|liberationsans|verdana|tahoma|segoe|sans|gothic"
            r"|roboto|lato|futura|frutiger|myriad|gill",
            re.I,
        ),
        "sans",
    ),
    (
        re.compile(
            r"times|tinos|liberationserif|georgia|garamond|book|palatino|cmr|lmroman"
            r"|nimbus ?rom|minion|charter|serif|cambria",
            re.I,
        ),
        "serif",
    ),
]


@dataclass(frozen=True)
class ResolvedFont:
    font: pymupdf.Font
    label: str
    original: bool = False


def _variant_index(style: TextStyle) -> int:
    return (1 if style.bold else 0) + (2 if style.italic else 0)


class FontResolver:
    def __init__(
        self,
        doc: Optional[pymupdf.Document] = None,
        *,
        extra_dirs: str = "",
        reuse_original: bool = True,
    ):
        dirs = [
            d for d in (extra_dirs or "").split(os.pathsep) if d
        ] + DEFAULT_FONT_DIRS
        self.dirs = [Path(d) for d in dirs if Path(d).is_dir()]
        self.doc = doc
        self.reuse_original = reuse_original and doc is not None
        self._lock = threading.RLock()
        self._file_cache: dict[str, Optional[pymupdf.Font]] = {}
        self._original_fonts: dict[str, Optional[pymupdf.Font]] = {}
        self._original_xrefs: Optional[dict[str, int]] = None
        self._ink_cache: dict[tuple[int, str], bool] = {}
        self._fallback: Optional[pymupdf.Font] = None

    # --------------------------------------------------------------- files
    def _find_file(self, name: str) -> Optional[Path]:
        for d in self.dirs:
            p = d / name
            if p.exists():
                return p
            for sub in d.rglob(name):
                return sub
        return None

    def _load_file(self, name: str) -> Optional[pymupdf.Font]:
        with self._lock:
            if name not in self._file_cache:
                path = self._find_file(name)
                font = None
                if path:
                    try:
                        font = pymupdf.Font(fontfile=str(path))
                    except Exception as exc:  # pragma: no cover
                        log.warning("cannot load font %s: %s", path, exc)
                self._file_cache[name] = font
            return self._file_cache[name]

    def _base14(self, family: str, idx: int) -> pymupdf.Font:
        key = "base14:" + BASE14[family][idx]
        with self._lock:
            if key not in self._file_cache:
                self._file_cache[key] = pymupdf.Font(BASE14[family][idx])
            return self._file_cache[key]  # type: ignore[return-value]

    def fallback_font(self) -> pymupdf.Font:
        if self._fallback is None:
            for name in UNICODE_FALLBACK_FILES:
                font = self._load_file(name)
                if font:
                    self._fallback = font
                    break
            else:
                self._fallback = pymupdf.Font("cjk")
        return self._fallback

    # ------------------------------------------------------------ original
    def _original_font(self, name: str) -> Optional[pymupdf.Font]:
        if not self.reuse_original or not name:
            return None
        with self._lock:
            if name in self._original_fonts:
                return self._original_fonts[name]
            if self._original_xrefs is None:
                self._original_xrefs = {}
                for page in self.doc:
                    for f in page.get_fonts(full=True):
                        xref, ext, ftype, basefont = f[0], f[1], f[2], f[3]
                        if ext in ("n/a", "") or ftype.lower() == "type3":
                            continue
                        clean = re.sub(r"^[A-Z]{6}\+", "", basefont)
                        self._original_xrefs.setdefault(clean, xref)
            font = None
            xref = self._original_xrefs.get(name)
            if xref:
                try:
                    _, ext, _, buffer = self.doc.extract_font(xref)
                    if buffer and ext not in ("n/a",):
                        font = pymupdf.Font(fontbuffer=buffer)
                except Exception as exc:
                    log.debug("cannot reuse font %s: %s", name, exc)
            self._original_fonts[name] = font
            return font

    def _has_ink(self, font: pymupdf.Font, ch: str) -> bool:
        if ch.isspace():
            return True
        key = (id(font), ch)
        with self._lock:
            if key in self._ink_cache:
                return self._ink_cache[key]
        ok = False
        try:
            if font.has_glyph(ord(ch)):
                tmp = pymupdf.open()
                page = tmp.new_page(width=60, height=60)
                tw = pymupdf.TextWriter(page.rect)
                tw.append((8, 44), ch, font=font, fontsize=40)
                tw.write_text(page)
                pix = page.get_pixmap(dpi=36, colorspace=pymupdf.csGRAY)
                ok = min(pix.samples) < 200 if pix.samples else False
                tmp.close()
        except Exception:
            ok = False
        with self._lock:
            self._ink_cache[key] = ok
        return ok

    def covers(self, font: pymupdf.Font, text: str) -> bool:
        return all(self._has_ink(font, ch) for ch in set(text))

    # -------------------------------------------------------------- public
    def family_for(self, style: TextStyle) -> str:
        name = style.font or ""
        if style.mono:
            return "mono"
        for pattern, family in _FAMILY_RULES:
            if pattern.search(name):
                return family
        return "serif" if style.serif else "sans"

    def substitute(self, style: TextStyle) -> ResolvedFont:
        family = self.family_for(style)
        idx = _variant_index(style)
        if family in FAMILY_FILES:
            font = self._load_file(FAMILY_FILES[family][idx]) or self._load_file(
                FAMILY_FILES[family][0]
            )
            if font:
                return ResolvedFont(font, FAMILY_FILES[family][idx])
        generic = (
            "serif"
            if family == "caladea"
            else "sans" if family == "carlito" else family
        )
        if generic in FAMILY_FILES:
            font = self._load_file(FAMILY_FILES[generic][idx])
            if font:
                return ResolvedFont(font, FAMILY_FILES[generic][idx])
        return ResolvedFont(
            self._base14(generic if generic in BASE14 else "serif", idx),
            BASE14.get(generic, BASE14["serif"])[idx],
        )

    def resolve(self, style: TextStyle, text: str) -> ResolvedFont:
        """Best font for a whole run of text in the given style."""
        original = self._original_font(style.font)
        if original is not None and self.covers(original, text):
            return ResolvedFont(original, style.font, original=True)
        return self.substitute(style)

    def split_by_coverage(
        self, font: ResolvedFont, text: str
    ) -> list[tuple[str, pymupdf.Font]]:
        """Split text into pieces, using the Unicode fallback for missing glyphs."""
        pieces: list[tuple[str, pymupdf.Font]] = []
        fallback = None
        for ch in text:
            use = font.font
            if not ch.isspace() and not font.font.has_glyph(ord(ch)):
                fallback = fallback or self.fallback_font()
                use = fallback
            if pieces and pieces[-1][1] is use:
                pieces[-1] = (pieces[-1][0] + ch, use)
            else:
                pieces.append((ch, use))
        return pieces
