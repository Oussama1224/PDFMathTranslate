import shutil
from pathlib import Path

import pymupdf
import pytest

from pt2en.model import TextStyle
from pt2en.reconstruction import fonts
from pt2en.reconstruction.fonts import FontResolver

SYSTEM_FONT_DIRS = list(fonts.DEFAULT_FONT_DIRS)


def system_font(name: str) -> Path:
    for d in SYSTEM_FONT_DIRS:
        found = next(Path(d).rglob(name), None) if Path(d).is_dir() else None
        if found:
            return found
    pytest.skip(f"{name} not installed")


@pytest.fixture()
def windows_fonts(tmp_path, monkeypatch):
    """A font folder laid out like C:\\Windows\\Fonts (no Carlito/Liberation)."""
    for src, dst in [
        ("Carlito-Regular.ttf", "calibri.ttf"),
        ("LiberationSans-Bold.ttf", "arialbd.ttf"),
        ("LiberationSerif-Regular.ttf", "times.ttf"),
    ]:
        shutil.copy(system_font(src), tmp_path / dst)
    monkeypatch.setattr(fonts, "DEFAULT_FONT_DIRS", [])
    return tmp_path


def test_windows_system_fonts_are_used(windows_fonts):
    resolver = FontResolver(None, extra_dirs=str(windows_fonts))
    assert resolver.substitute(TextStyle(font="Calibri")).label == "calibri.ttf"
    # Calibri Bold missing: same family's regular beats a different family
    assert resolver.substitute(TextStyle(font="Calibri", bold=True)).label == (
        "calibri.ttf"
    )
    assert resolver.substitute(TextStyle(font="Arial", bold=True)).label == (
        "arialbd.ttf"
    )
    assert resolver.substitute(TextStyle(font="TimesNewRoman")).label == "times.ttf"
    # Cambria missing: generic serif (Times New Roman)
    assert resolver.substitute(TextStyle(font="Cambria")).label == "times.ttf"
    assert resolver.family_file("carlito") == windows_fonts / "calibri.ttf"
    assert resolver.family_file("mono") is None
    # Nothing at all for mono: Base-14 Courier
    assert resolver.substitute(TextStyle(font="Consolas", mono=True)).label == "cour"


def test_font_collection_face_is_embedded(windows_fonts, tmp_path):
    ttlib = pytest.importorskip("fontTools.ttLib")
    collection = ttlib.TTCollection()
    collection.fonts = [
        ttlib.TTFont(system_font("Caladea-Regular.ttf")),
        ttlib.TTFont(system_font("LiberationSans-Regular.ttf")),
    ]
    collection.save(windows_fonts / "cambria.ttc")
    resolver = FontResolver(None, extra_dirs=str(windows_fonts))
    resolved = resolver.substitute(TextStyle(font="Cambria"))
    assert resolved.label == "cambria.ttc"

    doc = pymupdf.open()
    page = doc.new_page()
    tw = pymupdf.TextWriter(page.rect)
    tw.append((72, 100), "Learning outcomes", font=resolved.font, fontsize=12)
    tw.write_text(page)
    doc.subset_fonts()
    out = tmp_path / "ttc.pdf"
    doc.save(out, garbage=3, deflate=True)
    doc = pymupdf.open(out)
    (xref, ext, *_), *_ = doc[0].get_fonts(full=True)
    assert ext == "ttf"  # single face extracted, not the whole collection
    assert doc[0].get_text().strip() == "Learning outcomes"
