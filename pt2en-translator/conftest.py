"""Lets the repository-level ``pytest .`` (pdf2zh environment) skip this app's
tests when the app's own dependencies are not installed there."""

import importlib.util

_REQUIRED = (
    "pymupdf",
    "pydantic_settings",
    "fastapi",
    "pytesseract",
    "cv2",
    "anthropic",
    "multipart",
)

collect_ignore_glob = (
    ["tests/*"] if any(importlib.util.find_spec(m) is None for m in _REQUIRED) else []
)
