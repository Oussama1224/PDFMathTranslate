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

# .env is read from the project folder (pt2en-translator/, for a source checkout)
# and from the working directory; the working directory wins.
PROJECT_DIR = Path(__file__).resolve().parent.parent
ENV_FILES = (PROJECT_DIR / ".env", Path(".env"))
# Names an editor or browser may give the file by accident.
MISNAMED_ENV_FILES = (".env.txt", "env", "env.txt", "_env")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


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
        env_file=ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        # blank lines such as "OPENAI_API_KEY=" mean "not set", so an alias
        # (NVIDIA_API_KEY) or the default still applies
        env_ignore_empty=True,
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
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_effort: str = "high"
    # "default" enables server-side refusal fallbacks; "none" disables them
    # (useful behind gateways that do not support the beta parameter).
    anthropic_fallbacks: str = "default"
    anthropic_max_tokens: int = 32000

    # Any OpenAI-compatible chat-completions endpoint. An NVIDIA key (nvapi-...)
    # selects NVIDIA's hosted API automatically.
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_API_KEY", "PT2EN_OPENAI_API_KEY", "NVIDIA_API_KEY"
        ),
    )
    # unset = gpt-4o, or on NVIDIA the best chat model the key can use
    openai_model: Optional[str] = None
    openai_base_url: Optional[str] = None
    # Output-token limit per request; unset = endpoint default (16384 for NVIDIA,
    # whose own default is too small for a translation batch).
    openai_max_tokens: Optional[int] = None

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

    @property
    def openai_endpoint(self) -> Optional[str]:
        if self.openai_base_url:
            return self.openai_base_url
        if (self.openai_api_key or "").startswith("nvapi-"):
            return NVIDIA_BASE_URL
        return None

    @property
    def openai_is_nvidia(self) -> bool:
        return "nvidia.com" in (self.openai_endpoint or "")

    @property
    def openai_model_name(self) -> str:
        if self.openai_model:
            return self.openai_model
        return "auto" if self.openai_is_nvidia else "gpt-4o"

    def provider_problem(self, name: str) -> str:
        """Why a provider cannot be used ("" when it can)."""
        if name == "anthropic":
            return "" if self.anthropic_api_key else "ANTHROPIC_API_KEY is not set."
        if name == "openai":
            if not (self.openai_api_key or self.openai_base_url):
                return "No API key: set NVIDIA_API_KEY or OPENAI_API_KEY."
            return _missing_package("openai", "openai")
        if name == "deepl":
            if not self.deepl_api_key:
                return "DEEPL_API_KEY is not set."
            return _missing_package("deepl", "deepl")
        if name == "argos":
            return _missing_package("argostranslate", "argos")
        if name in ("demo", "mock"):
            return ""
        return f"Unknown translation provider '{name}'."

    def provider_available(self, name: str) -> bool:
        return not self.provider_problem(name)


def _missing_package(module: str, extra: str) -> str:
    try:
        __import__(module)
        return ""
    except ImportError:
        return f"The {module} package is not installed: " f'pip install -e ".[{extra}]"'


def env_files_found() -> list[Path]:
    """The .env files that were read, in load order."""
    found: list[Path] = []
    for path in ENV_FILES:
        path = path.resolve()
        if path.is_file() and path not in found:
            found.append(path)
    return found


def settings_warning(settings: "Settings") -> str:
    """A configuration problem worth showing to the user, or ""."""
    if not env_files_found():
        folders = sorted({str(p.resolve().parent) for p in ENV_FILES})
        message = "No .env file found (looked in " + " and ".join(folders) + ")."
        for folder in folders:
            for name in MISNAMED_ENV_FILES:
                if (Path(folder) / name).is_file():
                    message += f" Found {Path(folder) / name}: rename it to .env."
        return message
    problem = settings.provider_problem(settings.translator)
    if problem:
        return f"PT2EN_TRANSLATOR={settings.translator}, but: {problem}"
    return ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
