import stat
from pathlib import Path

import numpy as np
import pytest

from pt2en.ocr import tesseract
from pt2en.ocr.tesseract import TesseractOCR, _clean_cmd, create_ocr, find_tesseract


def fake_exe(folder: Path, name: str = "tesseract.exe") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / name
    exe.write_text("")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_clean_cmd_undoes_quotes_and_escape_mangling():
    # A quoted "D:\Tesseract\tesseract.exe" in .env arrives with \t as a tab.
    assert _clean_cmd('"D:\\Tesseract\tesseract.exe"') == r"D:\Tesseract\tesseract.exe"
    assert _clean_cmd("  /usr/bin/tesseract ") == "/usr/bin/tesseract"


def test_configured_executable_or_folder(tmp_path):
    exe = fake_exe(tmp_path / "Tesseract")
    assert find_tesseract(str(exe)) == (str(exe), "")
    # A folder is accepted too (PT2EN_TESSERACT_CMD=D:/Tesseract)
    assert find_tesseract(str(tmp_path / "Tesseract")) == (str(exe), "")


def test_configured_path_missing_is_explained(tmp_path):
    path, reason = find_tesseract(str(tmp_path / "nope" / "tesseract.exe"))
    assert path is None
    assert "PT2EN_TESSERACT_CMD" in reason and "nope" in reason


def test_windows_drive_folders_are_searched(tmp_path, monkeypatch):
    exe = fake_exe(tmp_path / "D" / "Tesseract")
    monkeypatch.setattr(tesseract.shutil, "which", lambda name: None)
    monkeypatch.setattr(tesseract, "COMMON_LOCATIONS", [])
    monkeypatch.setattr(
        tesseract,
        "_windows_candidates",
        lambda: [
            tmp_path / d / f for d in "CD" for f in ("Tesseract-OCR", "Tesseract")
        ],
    )
    assert find_tesseract() == (str(exe), "")


def test_not_found_reason_reaches_engine(monkeypatch):
    monkeypatch.setattr(tesseract.shutil, "which", lambda name: None)
    monkeypatch.setattr(tesseract, "COMMON_LOCATIONS", [])
    monkeypatch.setattr(tesseract, "_windows_candidates", lambda: [])
    ocr = create_ocr("tesseract", "por+eng")
    assert isinstance(ocr, TesseractOCR)
    assert not ocr.available()
    assert "Tesseract was not found" in ocr.reason
    assert ocr.recognize(np.zeros((4, 4), "uint8")) == []


def test_disabled_engine_reason():
    ocr = create_ocr("none", "por+eng")
    assert not ocr.available()
    assert "PT2EN_OCR_ENGINE" in ocr.reason


def test_real_tesseract_reports_path():
    ocr = create_ocr("tesseract", "por+eng")
    if not ocr.available():
        pytest.skip(ocr.reason)
    assert ocr.path and Path(ocr.path).name.startswith("tesseract")
