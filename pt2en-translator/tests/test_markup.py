from pt2en.model import ElementKind, Glyph, Line, MathRegion, Run, TextBlock, TextStyle
from pt2en.translation import markup as mk

BASE = TextStyle(font="Calibri", size=11)
BOLD = TextStyle(font="Calibri-Bold", size=11, bold=True)
RED = TextStyle(font="Calibri", size=11, color=0xFF0000)


def run(text, style, math=False, region=None, x=0.0, base=10.0):
    gl = [
        Glyph(
            c,
            (x + i * 5, base - 9, x + i * 5 + 5, base + 3),
            (x + i * 5, base),
            style.size,
        )
        for i, c in enumerate(text)
    ]
    return Run(glyphs=gl, style=style, is_math=math, region_id=region)


def block(*lines_runs):
    lines = [
        Line(runs=list(r), bbox=(0, 0, 100, 12), baseline=10.0, size=11)
        for r in lines_runs
    ]
    b = TextBlock(
        id="b",
        page=0,
        kind=ElementKind.PARAGRAPH,
        bbox=(0, 0, 100, 30),
        lines=lines,
        style=BASE,
    )
    return b


def test_encode_styles_and_math():
    b = block(
        [
            run("A média de ", BASE),
            run("x", BASE, math=True, region="m1"),
            run(" é ", BASE),
            run("importante", BOLD),
        ]
    )
    b.math["m1"] = MathRegion("m1", 0, (0, 0, 5, 12), "x", 10.0)
    markup = mk.encode_block(b)
    assert markup == "A média de <m1/> é <b>importante</b>"
    tokens = mk.decode(markup, BASE, b.styles, b.protected, {"m1"})
    assert isinstance(tokens[1], mk.MathToken)
    assert tokens[-1].style.bold


def test_generic_style_tag_for_colour():
    b = block([run("Atenção: ", RED), run("leia com cuidado", BASE)])
    markup = mk.encode_block(b)
    assert markup.startswith("<s1>Atenção:</s1>")
    tokens = mk.decode("<s1>Warning:</s1> read carefully", BASE, b.styles, {}, set())
    assert tokens[0].style.color == 0xFF0000


def test_dehyphenation_across_lines():
    b = block([run("infor-", BASE)], [run("mação útil", BASE)])
    assert mk.encode_block(b) == "informação útil"


def test_protected_urls_and_escaping():
    b = block([run("Ver https://exemplo.pt/a e x<y & z", BASE)])
    markup = mk.encode_block(b)
    assert "<x1/>" in markup and "&lt;" in markup and "&amp;" in markup
    assert b.protected["x1"] == "https://exemplo.pt/a"
    assert (
        mk.plain_text(markup, b.protected, {}) == "Ver https://exemplo.pt/a e x<y & z"
    )


def test_validate_structure_detects_problems():
    src = "Valor <m1/> e <b>texto</b>"
    assert (
        mk.validate_structure(src, "Value <m1/> and <b>text</b>", {}, {}, {"m1"}) == []
    )
    assert any(
        "missing" in p
        for p in mk.validate_structure(src, "Value and <b>text</b>", {}, {}, {"m1"})
    )
    assert any(
        "invalid" in p
        for p in mk.validate_structure(src, "Value <m1/> and <b>text", {}, {}, {"m1"})
    )
    assert any(
        "duplicated" in p
        for p in mk.validate_structure(src, "<m1/> <m1/> <b>t</b>", {}, {}, {"m1"})
    )


def test_strip_tags_keeps_placeholders():
    assert mk.strip_tags("<b>A</b> <m2/> <s1>B</s1>") == "A <m2/> B"
