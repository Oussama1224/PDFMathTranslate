"""Application settings, loaded from environment variables (and an optional .env).

Every setting can be overridden with an environment variable prefixed with
``PT2EN_`` (e.g. ``PT2EN_TRANSLATOR=openai``). Provider credentials use their
conventional names (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``DEEPL_API_KEY``).
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TranslationStyle(str, Enum):
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    LITERAL = "literal"


class EnglishVariant(str, Enum):
    UK = "en-GB"
    US = "en-US"


class QAReviewMode(str, Enum):
    OFF = "off"
    REPORT = "report"
    FIX = "fix"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PT2EN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ------------------------------------------------------------------ server
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: Path = Path("./data")
    max_upload_mb: int = 100
    max_pages: int = 500
    max_concurrent_jobs: int = 2
    job_ttl_hours: int = 72
    access_token: Optional[str] = None
    cors_origins: str = ""
    log_level: str = "INFO"

    # ------------------------------------------------------------- translation
    translator: str = "anthropic"
    style: TranslationStyle = TranslationStyle.ACADEMIC
    english_variant: EnglishVariant = EnglishVariant.UK
    preserve_terminology: bool = True
    localize_numbers: bool = False
    translation_concurrency: int = 4
    batch_max_chars: int = 5000
    batch_max_segments: int = 40
    max_segment_retries: int = 2
    extract_glossary: bool = True
    cache_path: Optional[Path] = None

    anthropic_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "PT2EN_ANTHROPIC_API_KEY"),
    )
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "high"
    # "default" enables server-side refusal fallbacks; "none" disables them
    # (useful behind gateways that do not support the beta parameter).
    anthropic_fallbacks: str = "default"
    anthropic_max_tokens: int = 32000

    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "PT2EN_OPENAI_API_KEY"),
    )
    openai_model: str = "gpt-4o"
    openai_base_url: Optional[str] = None

    deepl_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DEEPL_API_KEY", "PT2EN_DEEPL_API_KEY"),
    )

    # --------------------------------------------------------------------- OCR
    ocr_engine: str = "tesseract"
    tesseract_cmd: Optional[str] = None
    ocr_languages: str = "por+eng"
    ocr_min_confidence: float = 55.0
    ocr_dpi: int = 300
    translate_images: bool = True
    min_image_side_px: int = 48
    min_image_area_pt: float = 900.0

    # ------------------------------------------------------------------ layout
    layout_detector: str = "heuristic"  # heuristic | doclayout
    doclayout_model_path: Optional[Path] = None
    font_dirs: str = ""
    reuse_original_fonts: bool = True
    min_font_scale: float = 0.72
    hard_min_font_scale: float = 0.5
    scan_font_family: str = "serif"

    # ---------------------------------------------------------------------- QA
    qa_review: QAReviewMode = QAReviewMode.FIX
    qa_ocr_images: bool = True
    qa_max_repair_rounds: int = 1

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def resolved_cache_path(self) -> Path:
        return self.cache_path or (self.data_dir / "translation_memory.sqlite3")

    def provider_available(self, name: str) -> bool:
        if name == "anthropic":
            return bool(self.anthropic_api_key)
        if name == "openai":
            return bool(self.openai_api_key) or bool(self.openai_base_url)
        if name == "deepl":
            return bool(self.deepl_api_key)
        if name == "argos":
            try:
                import argostranslate  # noqa: F401

                return True
            except ImportError:
                return False
        return name in ("demo", "mock")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
