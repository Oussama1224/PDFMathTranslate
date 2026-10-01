"""Mathematical-content detection.

Formulas are never sent to the translator. Every glyph is classified as
*text* or *math* using the font it is drawn with, its Unicode properties and
its neighbourhood; contiguous math glyphs form math runs which the
reconstruction stage copies verbatim (as vector content) from the source PDF.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

from pt2en.model import Glyph, Run, TextStyle

MATH_FONT_RE = re.compile(
    r"(CMMI|CMSY|CMEX|CMBSY|CMMIB|MSAM|MSBM|EUFM|EUFB|EUSM|EUSB|EUEX|EURM|EURB|RSFS"
    r"|STIX|XITS|Cambria\s*Math|CambriaMath|MT\s*Extra|MTExtra|Math|Symbol|Asana"
    r"|LMMath|LatinModernMath|TeXGyre\w*Math|txsy|txex|txmi|txsym|pxsy|pxmi|pxex"
    r"|esint|wasy|stmary|Euclid|MathematicalPi|Mathematica|Fourier-Math|zplm"
    r"|Wingdings|Webdings|ZapfDingbats|Dingbats|MS\s*Reference\s*Specialty)",
    re.IGNORECASE,
)

BULLET_CHARS = set("•●○◦▪▫■□►▶▸▹➢➤✓✔✗✘❖◆◇–—-*·⁃∙")

FUNCTION_NAMES = {
    "sin",
    "sen",
    "cos",
    "tan",
    "tg",
    "cot",
    "cotg",
    "sec",
    "cosec",
    "csc",
    "arcsin",
    "arcsen",
    "arccos",
    "arctan",
    "arctg",
    "sinh",
    "senh",
    "cosh",
    "tanh",
    "tgh",
    "log",
    "ln",
    "lg",
    "exp",
    "lim",
    "max",
    "min",
    "sup",
    "inf",
    "det",
    "dim",
    "ker",
    "arg",
    "mod",
    "gcd",
    "mdc",
    "mmc",
    "tr",
    "dx",
    "dy",
    "dt",
}

_GREEK_EXTRA = {0x03D1, 0x03D5, 0x03D6, 0x03F0, 0x03F1, 0x03F5, 0x03C2}
_MATH_RANGES = (
    (0x2200, 0x22FF),  # mathematical operators
    (0x2190, 0x21FF),  # arrows
    (0x27C0, 0x27EF),  # misc math symbols A
    (0x27F0, 0x27FF),  # supplemental arrows A
    (0x2980, 0x29FF),  # misc math symbols B
    (0x2A00, 0x2AFF),  # supplemental math operators
    (0x2070, 0x209F),  # superscripts and subscripts
    (0x1D400, 0x1D7FF),  # mathematical alphanumerics
    (0x2032, 0x2037),  # primes
    (0x2308, 0x230B),  # ceiling / floor
    (0xE000, 0xF8FF),  # private use (symbol fonts)
)
_LETTERLIKE_MATH = set("ℂℍℕℙℚℝℤℓ℘ℑℜℵℶ∂")
_SUPERSCRIPTS = set("²³¹⁰⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ᵢⱼₖₗₘₙₚₛₜₓᵃᵇᶜᵈᵉᶠᵍʰʲᵏˡᵐᵒᵖʳˢᵗᵘᵛʷˣʸᶻ")
_FORMULA_PUNCT = set("()[]{}.,;:/|'^_!*-−+=<>")


def is_math_font(font_name: str) -> bool:
    return bool(font_name) and bool(MATH_FONT_RE.search(font_name))


def unicode_is_math(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch[0])
    if ch in _LETTERLIKE_MATH or ch in _SUPERSCRIPTS:
        return True
    cat = unicodedata.category(ch[0])
    if cat == "Sm":
        return True
    if 0x0391 <= o <= 0x03C9 or o in _GREEK_EXTRA:
        return True
    return any(lo <= o <= hi for lo, hi in _MATH_RANGES)


def glyph_is_math(glyph: Glyph, style: TextStyle) -> bool:
    c = glyph.c
    if not c or c.isspace():
        return False
    if is_math_font(style.font):
        # Symbol/dingbat fonts also carry bullets: those are handled as list
        # prefixes by the block builder, but they still must never be re-typeset.
        return True
    return unicode_is_math(c)


def _is_operator_word(word: str) -> bool:
    stripped = word.strip()
    return bool(stripped) and all(unicode_is_math(ch) for ch in stripped)


def _is_numeric_word(word: str) -> bool:
    w = word.strip()
    return (
        bool(w) and not any(ch.isalpha() for ch in w) and any(ch.isdigit() for ch in w)
    )


_VARIABLE_RE = re.compile(
    r"^[(\[]?[A-Za-z\u03b1-\u03c9\u0391-\u03a9][\u0300-\u036f]*"
    r"[0-9₀-₉²³¹′'*ᵢⱼₖₙₘ]*[)\],;:.]?$"
)
_LETTER_PAIR_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]{2,}")


def _is_variable_token(word: str) -> bool:
    """A single letter possibly decorated: x, x̄, xᵢ, x', a1, (x ..."""
    return bool(_VARIABLE_RE.match(word.strip()))


def _is_formula_filler(word: str) -> bool:
    """Digits/punctuation/single-letter operands without natural-language letters."""
    w = word.strip()
    if not w or _LETTER_PAIR_RE.search(w):
        return False
    if not any(ch.isdigit() or ch in _FORMULA_PUNCT for ch in w):
        return False
    return all(
        ch.isdigit()
        or ch in _FORMULA_PUNCT
        or ch.isalpha()
        or unicodedata.category(ch) == "Mn"
        for ch in w
    )


def classify_line_glyphs(
    glyphs: Sequence[Glyph], styles: Sequence[TextStyle]
) -> list[bool]:
    """Return a math flag per glyph, using context to absorb operands."""
    n = len(glyphs)
    flags = [glyph_is_math(g, s) for g, s in zip(glyphs, styles)]
    if n == 0:
        return flags

    # Split into words (maximal non-space sequences) to reason about context.
    words: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if glyphs[i].c.isspace():
            i += 1
            continue
        j = i
        while j < n and not glyphs[j].c.isspace():
            j += 1
        words.append((i, j))
        i = j

    def text_of(w: tuple[int, int]) -> str:
        return "".join(g.c for g in glyphs[w[0] : w[1]])

    word_math = []
    for a, b in words:
        has_math = any(flags[a:b])
        if has_math:
            # "α-hélice" style compounds: mostly natural language -> text.
            text_letters = sum(
                1 for k in range(a, b) if not flags[k] and glyphs[k].c.isalpha()
            )
            if text_letters >= 3 and text_letters > (b - a) * 0.6:
                for k in range(a, b):
                    flags[k] = False
                has_math = False
        word_math.append(has_math)

    # Context propagation across neighbouring words.
    changed = True
    while changed:
        changed = False
        for idx, (a, b) in enumerate(words):
            if word_math[idx]:
                continue
            word = text_of((a, b))
            left = word_math[idx - 1] if idx > 0 else False
            right = word_math[idx + 1] if idx + 1 < len(words) else False
            left_word = text_of(words[idx - 1]) if idx > 0 else ""
            right_word = text_of(words[idx + 1]) if idx + 1 < len(words) else ""
            left_op = left and _is_operator_word(left_word[-1:] or "")
            right_op = right and _is_operator_word(right_word[:1] or "")
            make_math = False
            if _is_formula_filler(word) and ((left and right) or left_op or right_op):
                make_math = True
            elif _is_variable_token(word) and (left_op or right_op):
                make_math = True
            elif word.lower().rstrip("(") in FUNCTION_NAMES and (left or right):
                make_math = True
            if make_math:
                word_math[idx] = True
                changed = True

    result = [False] * n
    for (a, b), is_math in zip(words, word_math):
        if is_math:
            for k in range(a, b):
                result[k] = True
    # Spaces between two math words belong to the formula.
    for k in range(n):
        if glyphs[k].c.isspace():
            left = next(
                (result[p] for p in range(k - 1, -1, -1) if not glyphs[p].c.isspace()),
                False,
            )
            right = next(
                (result[p] for p in range(k + 1, n) if not glyphs[p].c.isspace()), False
            )
            result[k] = left and right
    return result


def build_runs(glyphs: Sequence[Glyph], styles: Sequence[TextStyle]) -> list[Run]:
    """Group glyphs into runs of equal style and equal math flag."""
    flags = classify_line_glyphs(glyphs, styles)
    runs: list[Run] = []
    for glyph, style, flag in zip(glyphs, styles, flags):
        if (
            runs
            and runs[-1].is_math == flag
            and (flag or runs[-1].style.visual_key() == style.visual_key())
        ):
            runs[-1].glyphs.append(glyph)
            continue
        runs.append(Run(glyphs=[glyph], style=style, is_math=flag))
    return runs


def math_ratio(runs: Iterable[Run]) -> float:
    total = 0
    math = 0
    for run in runs:
        count = sum(1 for g in run.glyphs if not g.c.isspace())
        total += count
        if run.is_math:
            math += count
    return (math / total) if total else 0.0


_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}")


def natural_words(runs: Iterable[Run]) -> list[str]:
    words: list[str] = []
    for run in runs:
        if run.is_math:
            continue
        words.extend(
            w for w in _WORD_RE.findall(run.text) if w.lower() not in FUNCTION_NAMES
        )
    return words


def is_formula_line(runs: Sequence[Run]) -> bool:
    """A line made of mathematics only (possibly with numbering / punctuation)."""
    ratio = math_ratio(runs)
    if ratio == 0:
        return False
    words = natural_words(runs)
    if not words:
        return ratio >= 0.3
    return ratio >= 0.75 and len(words) <= 1 and len(words[0]) <= 4
