import pymupdf

from pt2en.ingestion.loader import load_pdf
from pt2en.layout.analyzer import analyze_document
from pt2en.model import ElementKind, PageStrategy, TextStyle
from pt2en.reconstruction.fonts import FontResolver
from pt2en.reconstruction.layout_engine import Frame, MathBox, Typesetter
from pt2en.translation.markup import MathToken, TextToken

STYLE = TextStyle(font="Times-Roman", size=11, serif=True)


def frame(**kw):
    base = dict(
        x0=50, x1=250, top=100, bottom=160, first_baseline=110, size=11, line_pitch=13
    )
    base.update(kw)
    return Frame(**base)


def test_typesetter_wraps_and_fits():
    ts = Typesetter(FontResolver(None))
    tokens = [TextToken("The quick brown fox jumps over the lazy dog " * 3, STYLE)]
    res = ts.fit(tokens, frame(), STYLE, {})
    assert res.lines >= 2
    assert not res.overflow
    assert res.bbox[0] >= 50 - 0.5 and res.bbox[2] <= 250 + 0.5
    assert res.fragments[0].baseline == 110


def test_typesetter_shrinks_when_needed():
    ts = Typesetter(FontResolver(None))
    tokens = [TextToken("word " * 80, STYLE)]
    res = ts.fit(tokens, frame(bottom=140), STYLE, {})
    assert res.scale < 1.0


def test_typesetter_places_math_boxes():
    ts = Typesetter(FontResolver(None))
    tokens = [
        TextToken("given by ", STYLE),
        MathToken("m1"),
        TextToken(" where", STYLE),
    ]
    res = ts.fit(tokens, frame(), STYLE, {"m1": MathBox("m1", 30, 9, 3)})
    assert len(res.maths) == 1
    math = res.maths[0]
    words = [f for f in res.fragments if f.fragment.text == "where"]
    assert words and words[0].x >= math.x + 30


def test_centered_single_line_anchor():
    ts = Typesetter(FontResolver(None))
    f = frame(
        x0=0, x1=300, single_line=True, align="center", anchor_x=150, line_pitch=0
    )
    res = ts.fit([TextToken("Title", STYLE)], f, STYLE, {})
    left, right = res.bbox[0], res.bbox[2]
    assert abs((left + right) / 2 - 150) < 1.0


def test_sample_analysis(sample_pdf):
    loaded = load_pdf(sample_pdf)
    model = analyze_document(loaded.doc)
    assert model.page_count == 3
    assert model.pages[2].strategy == PageStrategy.SCANNED
    kinds = {b.kind for b in model.blocks()}
    for kind in (
        ElementKind.HEADING,
        ElementKind.PARAGRAPH,
        ElementKind.LIST_ITEM,
        ElementKind.TABLE_CELL,
        ElementKind.FORMULA,
        ElementKind.LABEL,
        ElementKind.FOOTNOTE,
        ElementKind.HEADER,
        ElementKind.PAGE_NUMBER,
    ):
        assert kind in kinds, kind
    blocks = {b.text.strip(): b for b in model.blocks()}
    heading = next(b for b in model.blocks() if b.prefix_text == "2.1")
    assert heading.kind == ElementKind.HEADING
    inline = next(b for b in model.blocks() if b.math)
    assert "observações" in inline.text
    assert any(b.rotation for b in model.blocks()), "rotated axis title expected"
    assert "15,5" in blocks and not blocks["15,5"].translatable


def test_loader_rejects_non_pdf():
    import pytest

    from pt2en.errors import InvalidDocumentError

    with pytest.raises(InvalidDocumentError):
        load_pdf(b"hello world")
    doc = pymupdf.open()
    doc.new_page()
    assert load_pdf(doc.tobytes()).page_count == 1
