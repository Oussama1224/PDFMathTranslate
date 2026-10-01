"""End-to-end pipeline test on the generated Portuguese sample (offline demo translator)."""

import pymupdf
import pytest

from pt2en.pipeline.options import JobOptions
from pt2en.pipeline.orchestrator import TranslationPipeline
from pt2en.pipeline.progress import Stage

from .conftest import HAS_TESSERACT, requires_ocr


@pytest.fixture(scope="module")
def result(sample_pdf, tmp_path_factory):
    from pt2en.config import Settings

    settings = Settings(
        _env_file=None, translator="demo", data_dir=tmp_path_factory.mktemp("data")
    )
    events = []
    pipeline = TranslationPipeline(
        settings, JobOptions.from_settings(settings), progress=events.append
    )
    res = pipeline.run(sample_pdf)
    res.events = events
    return res


def test_structure_preserved(result, sample_pdf):
    src = pymupdf.open(stream=sample_pdf, filetype="pdf")
    out = pymupdf.open(stream=result.output_pdf, filetype="pdf")
    assert out.page_count == src.page_count
    for a, b in zip(src, out):
        assert a.rect == b.rect
        assert len(b.get_image_info()) >= len(a.get_image_info())


def test_text_translated_and_numbers_kept(result):
    out = pymupdf.open(stream=result.output_pdf, filetype="pdf")
    text = out[0].get_text()
    assert "Chapter 2" in text and "Descriptive statistics" in text
    assert "Page 1 of 3" in text
    assert "15,5" in text and "(2.1)" in text  # numbers and equation numbers unchanged
    assert "Capítulo" not in text


def test_formulas_unchanged(result):
    checks = {c["id"]: c for c in result.report["checks"]}
    assert checks["formulas"]["status"] == "pass"
    assert checks["page_structure"]["status"] == "pass"
    assert checks["images"]["status"] == "pass"
    assert checks["bounds"]["status"] == "pass"
    stats = result.report["stats"]
    assert stats["display_formulas"] >= 2 and stats["inline_formulas"] >= 1


def test_qa_flags_untranslated_portuguese(result):
    # The demo dictionary does not know every word; QA must notice.
    cats = {i["category"] for i in result.report["issues"]}
    assert "untranslated" in cats
    assert result.report["summary"]["score"] < 100


def test_progress_covers_all_stages(result):
    stages = [e.stage for e in result.events]
    for stage in (
        Stage.ANALYZING,
        Stage.TRANSLATING,
        Stage.REBUILDING,
        Stage.VALIDATING,
        Stage.FINALIZING,
    ):
        assert stage.value in stages
    assert result.events[-1].overall == 1.0
    overall = [e.overall for e in result.events]
    assert overall == sorted(overall)


def test_review_pdf_has_summary_and_annotations(result):
    review = pymupdf.open(stream=result.review_pdf, filetype="pdf")
    out = pymupdf.open(stream=result.output_pdf, filetype="pdf")
    assert review.page_count == out.page_count + 1
    assert "Translation quality report" in review[0].get_text()
    assert sum(len(list(p.annots())) for p in review) >= 1


@requires_ocr
def test_chart_image_text_translated(result):
    stats = result.report["stats"]
    assert stats["images_translated"] == 1
    assert stats["image_labels_translated"] >= 5


@requires_ocr
def test_scanned_page_rebuilt_with_english_text(result):
    out = pymupdf.open(stream=result.output_pdf, filetype="pdf")
    text = out[2].get_text()
    assert "exercises" in text.lower()
    assert len(out[2].get_image_info()) == 1  # cleaned scan kept as background


@pytest.mark.skipif(HAS_TESSERACT, reason="only meaningful without OCR")
def test_pipeline_works_without_ocr(result):
    assert (
        result.report["summary"]["errors"] >= 1
    )  # scanned page reported, not silently skipped
