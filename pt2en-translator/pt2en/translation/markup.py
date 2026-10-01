"""Inline markup used to exchange text with the translator.

A block is serialised as text with lightweight tags so the translator can keep
inline formatting and formulas attached to the right words:

* ``<b>…</b>``, ``<i>…</i>``, ``<u>…</u>``, ``<sup>…</sup>``, ``<sub>…</sub>``
  add a property to the paragraph's base style;
* ``<s1>…</s1>`` (``s2``, …) apply an arbitrary style from the block's style
  table (colour, font family, size changes);
* ``<m1/>`` marks an inline formula that is copied verbatim from the source;
* ``<x1/>`` marks a protected literal (URL, e-mail, code identifier).

Text is XML-escaped (``&amp; &lt; &gt;``).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Optional, Union

from pt2en.model import TextBlock, TextStyle

SIMPLE_TAGS = ("b", "i", "u", "sup", "sub")
_TAG_RE = re.compile(r"<(/?)([a-z]+\d*)\s*(/?)>")
_PLACEHOLDER_RE = re.compile(r"<([mx]\d+)\s*/>")
_PROTECT_RE = re.compile(
    r"(https?://[^\s<>\"]+[^\s<>\".,;:!?)]|www\.[^\s<>\"]+[^\s<>\".,;:!?)]"
    r"|[\w.+-]+@[\w-]+\.[\w.-]*\w|\b[A-Za-z]\w*_\w+\b)"
)
_STYLE_WORDS_RE = re.compile(
    r"[-_,]?(bold|black|heavy|semibold|demibold|demi|medium|light|italic|oblique|regular|roman"
    r"|book|mt|ps|bd|it|bi)\b",
    re.I,
)


def font_family(name: str) -> str:
    base = name.split("+")[-1]
    base = _STYLE_WORDS_RE.sub("", base.replace("-", " "))
    return re.sub(r"\s+", "", base).lower()


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def unescape(text: str) -> str:
    return html.unescape(text)


@dataclass
class TextToken:
    text: str
    style: TextStyle


@dataclass
class MathToken:
    region_id: str


Token = Union[TextToken, MathToken]


class MarkupError(ValueError):
    pass


# --------------------------------------------------------------------- encode
def _tag_for(
    style: TextStyle, base: TextStyle, rel_baseline: float, styles: dict
) -> Optional[str]:
    size_close = abs(style.size - base.size) <= 0.12 * base.size
    if style.superscript or (
        rel_baseline < -0.18 * base.size and style.size < 0.9 * base.size
    ):
        return "sup"
    if rel_baseline > 0.1 * base.size and style.size < 0.9 * base.size:
        return "sub"
    same_family = font_family(style.font) == font_family(base.font) or not style.font
    others_same = (
        style.mono == base.mono
        and style.color == base.color
        and style.strike == base.strike
        and size_close
        and same_family
    )
    if (
        others_same
        and style.bold == base.bold
        and style.italic == base.italic
        and style.underline == base.underline
    ):
        return None
    if others_same:
        added = []
        if style.bold and not base.bold:
            added.append("b")
        if style.italic and not base.italic:
            added.append("i")
        if style.underline and not base.underline:
            added.append("u")
        removed = (
            (base.bold and not style.bold)
            or (base.italic and not style.italic)
            or (base.underline and not style.underline)
        )
        if added and not removed:
            return ">".join(added)
    for key, existing in styles.items():
        if (
            existing.visual_key() == style.visual_key()
            and abs(existing.size - style.size) < 0.3
        ):
            return key
    key = f"s{len(styles) + 1}"
    styles[key] = style
    return key


def encode_block(block: TextBlock) -> str:
    """Build ``block.source_markup`` / ``block.styles`` / ``block.protected``."""
    base = block.style
    styles: dict[str, TextStyle] = {}
    pieces: list[tuple[str, Optional[str]]] = []  # (text or placeholder, tag)
    emitted: set[str] = set()

    for li, line in enumerate(block.lines):
        if li > 0 and pieces:
            text, tag = pieces[-1]
            first_char = next(
                (r.text.lstrip()[:1] for r in line.runs if r.text.strip()), ""
            )
            if (
                tag != "__math__"
                and text.rstrip().endswith("-")
                and first_char.islower()
            ):
                stripped = text.rstrip()
                if len(stripped) >= 2 and stripped[-2].isalpha():
                    pieces[-1] = (stripped[:-1], tag)  # de-hyphenate "infor-" + "mação"
                else:
                    pieces.append((" ", None))
            else:
                pieces.append((" ", None))
        for run in line.runs:
            if run.is_math:
                rid = run.region_id
                if rid and rid not in emitted:
                    emitted.add(rid)
                    pieces.append((rid, "__math__"))
                continue
            text = run.text
            if not text:
                continue
            rel = _relative_baseline(run, line.baseline)
            tag = _tag_for(run.style, base, rel, styles) if text.strip() else None
            pieces.append((text, tag))

    protected: dict[str, str] = {}
    out: list[str] = []
    i = 0
    while i < len(pieces):
        text, tag = pieces[i]
        if tag == "__math__":
            out.append(f"<{text}/>")
            i += 1
            continue
        # Merge consecutive pieces with the same tag (spaces join anything).
        buf = text
        j = i + 1
        while (
            j < len(pieces)
            and pieces[j][1] != "__math__"
            and (pieces[j][1] == tag or (not pieces[j][0].strip() and tag is None))
        ):
            buf += pieces[j][0]
            j += 1
        encoded = _protect(buf, protected)
        if tag:
            opens = "".join(f"<{t}>" for t in tag.split(">"))
            closes = "".join(f"</{t}>" for t in reversed(tag.split(">")))
            lead = encoded[: len(encoded) - len(encoded.lstrip())]
            trail = encoded[len(encoded.rstrip()) :]
            out.append(f"{lead}{opens}{encoded.strip()}{closes}{trail}")
        else:
            out.append(encoded)
        i = j
    markup = re.sub(r"[ \t ]+", " ", "".join(out)).strip()
    block.source_markup = markup
    block.styles = styles
    block.protected = protected
    block.source_text = plain_text(
        markup, protected, {k: v.text for k, v in block.math.items()}
    )
    return markup


def _relative_baseline(run, line_baseline: float) -> float:
    ys = sorted(g.origin[1] for g in run.glyphs if not g.c.isspace()) or [line_baseline]
    return ys[len(ys) // 2] - line_baseline


def _protect(text: str, protected: dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        value = m.group(0)
        for key, existing in protected.items():
            if existing == value:
                return f"\x00{key}\x00"
        key = f"x{len(protected) + 1}"
        protected[key] = value
        return f"\x00{key}\x00"

    marked = _PROTECT_RE.sub(repl, text)
    parts = marked.split("\x00")
    out = []
    for k, part in enumerate(parts):
        out.append(f"<{part}/>" if k % 2 == 1 else escape(part))
    return "".join(out)


# --------------------------------------------------------------------- decode
def decode(
    markup: str,
    base: TextStyle,
    styles: dict[str, TextStyle],
    protected: dict[str, str],
    math_ids: set[str],
) -> list[Token]:
    tokens: list[Token] = []
    stack: list[tuple[str, TextStyle]] = [("", base)]
    pos = 0
    for m in _TAG_RE.finditer(markup):
        if m.start() > pos:
            _emit_text(tokens, unescape(markup[pos : m.start()]), stack[-1][1])
        closing, name, selfclose = m.group(1), m.group(2), m.group(3)
        pos = m.end()
        if selfclose or name[0] in "mx" and name[1:].isdigit():
            if name in math_ids:
                tokens.append(MathToken(name))
            elif name in protected:
                _emit_text(tokens, protected[name], stack[-1][1])
            else:
                raise MarkupError(f"unknown placeholder <{name}/>")
            continue
        if closing:
            if not any(tag == name for tag, _ in stack[1:]):
                raise MarkupError(f"unbalanced closing tag </{name}>")
            while stack and stack[-1][0] != name:
                stack.pop()
            stack.pop()
            continue
        current = stack[-1][1]
        if name == "b":
            new = current.copy(bold=True)
        elif name == "i":
            new = current.copy(italic=True)
        elif name == "u":
            new = current.copy(underline=True)
        elif name == "sup":
            new = current.copy(superscript=True, subscript=False)
        elif name == "sub":
            new = current.copy(subscript=True, superscript=False)
        elif name in styles:
            new = styles[name]
        else:
            raise MarkupError(f"unknown tag <{name}>")
        stack.append((name, new))
    if pos < len(markup):
        _emit_text(tokens, unescape(markup[pos:]), stack[-1][1])
    if len(stack) > 1:
        raise MarkupError("unclosed tag(s): " + ", ".join(t for t, _ in stack[1:]))
    return tokens


def _emit_text(tokens: list[Token], text: str, style: TextStyle) -> None:
    if not text:
        return
    if tokens and isinstance(tokens[-1], TextToken) and tokens[-1].style is style:
        tokens[-1].text += text
    else:
        tokens.append(TextToken(text, style))


def placeholders(markup: str) -> list[str]:
    return _PLACEHOLDER_RE.findall(markup)


def strip_tags(markup: str) -> str:
    """Remove style tags but keep placeholders."""
    return re.sub(r"</?(?:b|i|u|sup|sub|s\d+)\s*>", "", markup)


def plain_text(
    markup: str, protected: dict[str, str], math_text: dict[str, str]
) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key in protected:
            return protected[key]
        return math_text.get(key, "")

    text = _PLACEHOLDER_RE.sub(repl, markup)
    text = re.sub(r"</?[a-z]+\d*\s*>", "", text)
    return unescape(text)


def validate_structure(
    source: str, target: str, styles: dict, protected: dict, math_ids: set[str]
) -> list[str]:
    """Return a list of problems with a translated markup string."""
    problems = []
    src_ph = sorted(placeholders(source))
    tgt_ph = sorted(placeholders(target))
    missing = [p for p in src_ph if p not in tgt_ph]
    extra = [p for p in tgt_ph if p not in src_ph]
    if missing:
        problems.append(
            "missing placeholders: " + ", ".join(f"<{p}/>" for p in missing)
        )
    if extra:
        problems.append(
            "unexpected placeholders: " + ", ".join(f"<{p}/>" for p in extra)
        )
    dupes = {p for p in tgt_ph if tgt_ph.count(p) > 1}
    if dupes:
        problems.append("duplicated placeholders: " + ", ".join(sorted(dupes)))
    try:
        decode(target, TextStyle(), styles, protected, math_ids)
    except MarkupError as exc:
        problems.append(f"invalid markup: {exc}")
    return problems
