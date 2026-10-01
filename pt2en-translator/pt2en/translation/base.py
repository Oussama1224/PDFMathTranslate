"""Translator interface shared by every provider."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

from pt2en.config import EnglishVariant, TranslationStyle
from pt2en.translation.glossary import GlossaryEntry


@dataclass
class Segment:
    id: str
    text: str  # markup (see pt2en.translation.markup)
    kind: str = "paragraph"
    section: str = ""
    note: str = ""  # per-segment instruction (e.g. feedback on a failed attempt)
    max_ratio: Optional[float] = None  # length hint for labels in figures
    preceding: list[str] = field(
        default_factory=list
    )  # source context before the segment


@dataclass
class TranslationContext:
    document_title: str = ""
    style: TranslationStyle = TranslationStyle.ACADEMIC
    variant: EnglishVariant = EnglishVariant.UK
    preserve_terminology: bool = True
    localize_numbers: bool = False
    glossary: list[GlossaryEntry] = field(default_factory=list)


@dataclass
class ReviewFinding:
    id: str
    severity: str  # minor | major | critical
    category: str  # mistranslation | omission | addition | terminology | grammar | untranslated
    explanation: str
    suggestion: Optional[str] = None


class TranslationError(Exception):
    """A provider failed to translate a batch."""


class TranslationRefused(TranslationError):
    """The provider declined to translate the content."""


class Translator(abc.ABC):
    name: str = "base"
    #: whether the provider can keep inline markup tags/placeholders
    supports_markup: bool = True
    #: whether it can do terminology extraction and quality review
    is_llm: bool = False

    @abc.abstractmethod
    def translate_batch(
        self, segments: list[Segment], ctx: TranslationContext
    ) -> dict[str, str]:
        """Translate segments; return {segment id: translated markup}."""

    def extract_terminology(
        self, candidates: list[tuple[str, int, str]], ctx: TranslationContext
    ) -> list[GlossaryEntry]:
        return []

    def review(
        self, pairs: list[tuple[str, str, str]], ctx: TranslationContext
    ) -> list[ReviewFinding]:
        """Quality-review (id, source, translation) triples."""
        return []

    def describe(self) -> str:
        return self.name
