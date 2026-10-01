"""Typesetting of translated text inside the space of the original block.

The engine reflows English text (with inline styles and formula boxes) into
the block's frame, preserving alignment, indentation, line spacing and the
first baseline. If the text does not fit, it first uses free space below the
block, then tightens line spacing, then reduces the font size down to a
configurable floor; anything beyond is reported so QA can flag it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import pymupdf

from pt2en.model import TextStyle
from pt2en.reconstruction.fonts import FontResolver
from pt2en.translation.markup import MathToken, TextToken, Token

SUP_SCALE = 0.68
SUP_RISE = 0.36
SUB_DROP = 0.18


@dataclass
class Frame:
    x0: float  # left edge of the text column
    x1: float  # right edge
    top: float  # top of the original block
    bottom: float  # maximum bottom (region limit)
    first_baseline: float
    first_indent: float = 0.0
    rest_indent: float = 0.0
    align: str = "left"
    line_pitch: float = 0.0  # original baseline-to-baseline distance (0 = unknown)
    size: float = 10.0  # base font size
    single_line: bool = False
    anchor_x: Optional[float] = None  # for single-line centre/right anchoring


@dataclass
class MathBox:
    region_id: str
    width: float
    ascent: float
    descent: float


@dataclass
class Fragment:
    text: str
    style: TextStyle
    font: pymupdf.Font
    size: float
    rise: float  # positive = raised (superscript)
    width: float


Piece = Union[Fragment, MathBox]


@dataclass
class Word:
    pieces: list[Piece]
    space_after: bool = False

    @property
    def width(self) -> float:
        return sum(p.width for p in self.pieces)


@dataclass
class PlacedFragment:
    x: float
    baseline: float
    fragment: Fragment


@dataclass
class PlacedMath:
    x: float
    baseline: float
    box: MathBox

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return (
            self.x,
            self.baseline - self.box.ascent,
            self.x + self.box.width,
            self.baseline + self.box.descent,
        )


@dataclass
class TypesetResult:
    fragments: list[PlacedFragment] = field(default_factory=list)
    maths: list[PlacedMath] = field(default_factory=list)
    scale: float = 1.0
    bottom: float = 0.0
    lines: int = 0
    overflow: bool = False  # text exceeds the frame (after all fitting steps)
    too_wide: bool = False  # a single word wider than the column
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)


class Typesetter:
    def __init__(self, fonts: FontResolver):
        self.fonts = fonts

    # ------------------------------------------------------------ words
    def _words(
        self,
        tokens: list[Token],
        scale: float,
        maths: dict[str, MathBox],
        base_size: float,
    ) -> list[Word]:
        words: list[Word] = [Word(pieces=[])]
        for tok in tokens:
            if isinstance(tok, MathToken):
                box = maths.get(tok.region_id)
                if box is None:
                    continue
                scaled = MathBox(
                    box.region_id,
                    box.width * scale,
                    box.ascent * scale,
                    box.descent * scale,
                )
                words[-1].pieces.append(scaled)
                continue
            assert isinstance(tok, TextToken)
            style = tok.style
            size = style.size * scale
            rise = 0.0
            if style.superscript:
                rise = base_size * scale * SUP_RISE
                size = size * SUP_SCALE if style.size >= base_size * 0.85 else size
            elif style.subscript:
                rise = -base_size * scale * SUB_DROP
                size = size * SUP_SCALE if style.size >= base_size * 0.85 else size
            parts = tok.text.replace("\n", " ").split(" ")
            for k, part in enumerate(parts):
                if k > 0:
                    if words[-1].pieces:
                        words[-1].space_after = True
                        words.append(Word(pieces=[]))
                if not part:
                    continue
                resolved = self.fonts.resolve(style, part)
                for text, font in self.fonts.split_by_coverage(resolved, part):
                    width = font.text_length(text, size)
                    words[-1].pieces.append(
                        Fragment(text, style, font, size, rise, width)
                    )
        return [w for w in words if w.pieces]

    def _space_width(self, word: Word, scale: float, base: TextStyle) -> float:
        frag = next((p for p in reversed(word.pieces) if isinstance(p, Fragment)), None)
        if frag is not None:
            return frag.font.text_length(" ", frag.size)
        font = self.fonts.resolve(base, " ").font
        return font.text_length(" ", base.size * scale)

    # ----------------------------------------------------------- layout
    def layout(
        self,
        tokens: list[Token],
        frame: Frame,
        base: TextStyle,
        maths: dict[str, MathBox],
        scale: float,
        pitch_factor: float = 1.0,
        allow_wrap: bool = True,
    ) -> TypesetResult:
        words = self._words(tokens, scale, maths, base.size)
        res = TypesetResult(scale=scale)
        size = frame.size * scale
        pitch = (
            frame.line_pitch * scale if frame.line_pitch else size * 1.2
        ) * pitch_factor
        pitch = max(pitch, size * 1.0)
        width_first = frame.x1 - frame.x0 - frame.first_indent
        width_rest = frame.x1 - frame.x0 - frame.rest_indent

        # Greedy line breaking.
        lines: list[list[Word]] = [[]]
        line_w = 0.0
        for word in words:
            avail = width_first if len(lines) == 1 else width_rest
            sp = self._space_width(lines[-1][-1], scale, base) if lines[-1] else 0.0
            if lines[-1] and allow_wrap and line_w + sp + word.width > avail + 0.01:
                lines.append([word])
                line_w = word.width
            else:
                line_w += sp + word.width
                lines[-1].append(word)
            if word.width > (width_first if len(lines) == 1 else width_rest) + 0.01:
                res.too_wide = True
        lines = [ln for ln in lines if ln]
        res.lines = len(lines)

        # Vertical metrics.
        ascent = size * 0.78
        descent = size * 0.22
        first_shift = (frame.first_baseline - frame.top) * scale
        baseline = frame.top + max(first_shift, ascent * 0.9)
        min_x = float("inf")
        max_x = float("-inf")
        prev_descent = descent
        for li, line in enumerate(lines):
            line_ascent = max(
                [ascent]
                + [p.ascent for w in line for p in w.pieces if isinstance(p, MathBox)]
            )
            line_descent = max(
                [descent]
                + [p.descent for w in line for p in w.pieces if isinstance(p, MathBox)]
            )
            if li == 0:
                baseline = max(baseline, frame.top + line_ascent * 0.95)
            else:
                baseline += max(pitch, prev_descent + line_ascent + size * 0.08)
            prev_descent = line_descent
            indent = frame.first_indent if li == 0 else frame.rest_indent
            avail = frame.x1 - frame.x0 - indent
            spaces = [
                self._space_width(w, scale, base) if w.space_after else 0.0
                for w in line[:-1]
            ]
            natural = sum(w.width for w in line) + sum(spaces)
            extra = avail - natural
            x = frame.x0 + indent
            gap_bonus = 0.0
            align = frame.align
            if frame.single_line and frame.anchor_x is not None and len(lines) == 1:
                if align == "center":
                    x = frame.anchor_x - natural / 2
                elif align == "right":
                    x = frame.anchor_x - natural
            elif align == "center":
                x += extra / 2
            elif align == "right":
                x += extra
            elif (
                align == "justify"
                and li < len(lines) - 1
                and extra > 0
                and len(line) > 1
            ):
                if extra < avail * 0.35:
                    gap_bonus = extra / max(1, sum(1 for s in spaces if s > 0))
            min_x = min(min_x, x)
            for wi, word in enumerate(line):
                for piece in word.pieces:
                    if isinstance(piece, MathBox):
                        res.maths.append(PlacedMath(x, baseline, piece))
                    else:
                        res.fragments.append(
                            PlacedFragment(x, baseline - piece.rise, piece)
                        )
                    x += piece.width
                if wi < len(line) - 1:
                    x += spaces[wi] + (gap_bonus if spaces[wi] > 0 else 0.0)
            max_x = max(max_x, x)
        res.bottom = baseline + prev_descent if lines else frame.top
        res.overflow = res.bottom > frame.bottom + 0.5 or res.too_wide
        if lines:
            res.bbox = (min_x, frame.top, max_x, res.bottom)
        return res

    def fit(
        self,
        tokens: list[Token],
        frame: Frame,
        base: TextStyle,
        maths: dict[str, MathBox],
        min_scale: float = 0.72,
        hard_min_scale: float = 0.5,
    ) -> TypesetResult:
        """Find the largest scale at which the text fits the frame."""
        scales = []
        s = 1.0
        while s >= min_scale - 1e-6:
            scales.append(round(s, 3))
            s -= 0.03
        hard = []
        s = min_scale - 0.04
        while s >= hard_min_scale - 1e-6:
            hard.append(round(s, 3))
            s -= 0.04
        if frame.single_line:
            for scale in scales:
                res = self.layout(tokens, frame, base, maths, scale, allow_wrap=False)
                if not res.overflow and self._within_width(res, frame):
                    return res
        last = None
        for scale in scales:
            for pitch_factor in (1.0, 0.9):
                res = self.layout(tokens, frame, base, maths, scale, pitch_factor)
                last = res
                if not res.overflow:
                    return res
        for scale in hard:
            res = self.layout(tokens, frame, base, maths, scale, 0.9)
            last = res
            if not res.overflow:
                return res
        assert last is not None
        return last

    @staticmethod
    def _within_width(res: TypesetResult, frame: Frame) -> bool:
        return res.bbox[0] >= frame.x0 - 0.5 and res.bbox[2] <= frame.x1 + 0.5
