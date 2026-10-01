import shutil

import pytest

from pt2en.config import Settings
from pt2en.samples import make_sample_pdf


def tesseract_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract

        return "por" in pytesseract.get_languages(config="")
    except Exception:
        return False


HAS_TESSERACT = tesseract_available()
requires_ocr = pytest.mark.skipif(
    not HAS_TESSERACT, reason="Tesseract with Portuguese data not installed"
)


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    return make_sample_pdf()


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        translator="demo",
        data_dir=tmp_path / "data",
        translation_concurrency=2,
        anthropic_api_key=None,
        openai_api_key=None,
        deepl_api_key=None,
    )
