"""Per-job options chosen by the user (defaults come from Settings)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from pt2en.config import EnglishVariant, QAReviewMode, Settings, TranslationStyle


class JobOptions(BaseModel):
    provider: Optional[str] = None
    source_language: str = "pt-PT"
    target_language: str = "en"
    style: TranslationStyle = TranslationStyle.ACADEMIC
    english_variant: EnglishVariant = EnglishVariant.UK
    preserve_terminology: bool = True
    localize_numbers: bool = False
    glossary_text: str = Field(default="", max_length=200_000)
    extract_glossary: bool = True
    translate_images: bool = True
    ocr_enabled: bool = True
    qa_review: QAReviewMode = QAReviewMode.FIX

    @classmethod
    def from_settings(cls, settings: Settings, **overrides) -> "JobOptions":
        base = cls(
            provider=settings.translator,
            style=settings.style,
            english_variant=settings.english_variant,
            preserve_terminology=settings.preserve_terminology,
            localize_numbers=settings.localize_numbers,
            extract_glossary=settings.extract_glossary,
            translate_images=settings.translate_images,
            ocr_enabled=settings.ocr_engine != "none",
            qa_review=settings.qa_review,
        )
        return base.model_copy(
            update={k: v for k, v in overrides.items() if v is not None}
        )

    def resolved_provider(self, settings: Settings) -> str:
        return self.provider or settings.translator
