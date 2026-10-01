from pt2en.layout.math import (
    build_runs,
    classify_line_glyphs,
    is_formula_line,
    is_math_font,
)
from pt2en.model import Glyph, TextStyle

TEXT = TextStyle(font="Times-Roman", size=11)
ITALIC = TextStyle(font="Times-Italic", size=11, italic=True)
SYMBOL = TextStyle(font="OpenSymbol", size=11)
CMMI = TextStyle(font="CMMI10", size=11)


def glyphs(spec):
    """spec: list of (text, style) -> glyph list + style list with fake geometry."""
    out, styles, x = [], [], 0.0
    for text, style in spec:
        for ch in text:
            out.append(Glyph(ch, (x, 0, x + 5, 12), (x, 10), style.size))
            styles.append(style)
            x += 5
    return out, styles


def flags(spec):
    g, s = glyphs(spec)
    return "".join("M" if f else "t" for f in classify_line_glyphs(g, s))


def test_math_fonts_detected():
    assert is_math_font("CMMI10")
    assert is_math_font("CambriaMath")
    assert is_math_font("OpenSymbol")
    assert not is_math_font("Calibri")
    assert not is_math_font("CMR10")  # roman text font: decided by context


def test_inline_formula_absorbs_operands():
    # "dada por x = (1/n) Σ xi"
    f = flags(
        [
            ("dada por ", TEXT),
            ("x", ITALIC),
            (" ", TEXT),
            ("=", SYMBOL),
            (" (1/", TEXT),
            ("n", ITALIC),
            (") ", TEXT),
            ("Σ", SYMBOL),
            ("x", ITALIC),
            ("i", ITALIC),
        ]
    )
    assert f.startswith("tttttttt")  # "dada por" stays text
    assert f[9:].replace(" ", "M").count("t") == 0  # the whole formula is math


def test_portuguese_single_letter_words_stay_text():
    # "a e o" are Portuguese words, not variables, when no operator is adjacent.
    f = flags([("a casa e o jardim", TEXT)])
    assert "M" not in f


def test_numbers_in_prose_are_text():
    f = flags([("Exercício 2.1 com 3 alunos", TEXT)])
    assert "M" not in f


def test_latex_math_font_glyphs_are_math():
    f = flags([("onde ", TEXT), ("x", CMMI), (" é real", TEXT)])
    assert f[5] == "M"
    assert f[:5] == "ttttt"


def test_formula_line_detection():
    g, s = glyphs(
        [
            ("s", ITALIC),
            ("2", TEXT),
            (" ", TEXT),
            ("=", SYMBOL),
            (" Σ(x", TEXT),
            ("−", SYMBOL),
            ("y)", TEXT),
        ]
    )
    assert is_formula_line(build_runs(g, s))
    g, s = glyphs([("A média é calculada assim", TEXT)])
    assert not is_formula_line(build_runs(g, s))


def test_compound_word_with_greek_letter_is_text():
    f = flags([("α-hélice", TEXT)])
    assert "M" not in f
