"""DeepL provider (supports XML tag handling, so inline markup is preserved)."""

from __future__ import annotations

import logging

from pt2en.config import EnglishVariant, Settings
from pt2en.errors import ProviderConfigurationError
from pt2en.translation.base import (
    Segment,
    TranslationContext,
    TranslationError,
    Translator,
)

log = logging.getLogger(__name__)


class DeepLTranslator(Translator):
    name = "deepl"
    supports_markup = True
    is_llm = False

    def __init__(self, settings: Settings):
        try:
            import deepl
        except ImportError as exc:
            raise ProviderConfigurationError(
                "Install the optional dependency: pip install 'pt2en-translator[deepl]'"
            ) from exc
        if not settings.deepl_api_key:
            raise ProviderConfigurationError("DEEPL_API_KEY is not set.")
        self._deepl = deepl
        self.client = deepl.Translator(settings.deepl_api_key)
        self._glossaries: dict[str, object] = {}

    def describe(self) -> str:
        return "DeepL"

    def _glossary_for(self, ctx: TranslationContext):
        entries = {
            e.pt: (e.pt if e.keep else e.en.split("/")[0].strip())
            for e in ctx.glossary
            if e.strict
        }
        if not entries:
            return None
        key = repr(sorted(entries.items()))
        if key not in self._glossaries:
            try:
                self._glossaries[key] = self.client.create_glossary(
                    "pt2en-job", source_lang="PT", target_lang="EN", entries=entries
                )
            except Exception as exc:  # pragma: no cover - depends on account features
                log.warning("DeepL glossary creation failed: %s", exc)
                self._glossaries[key] = None
        return self._glossaries[key]

    def translate_batch(
        self, segments: list[Segment], ctx: TranslationContext
    ) -> dict[str, str]:
        target = "EN-GB" if ctx.variant == EnglishVariant.UK else "EN-US"
        texts = [s.text for s in segments]
        try:
            results = self.client.translate_text(
                texts,
                source_lang="PT",
                target_lang=target,
                tag_handling="xml",
                glossary=self._glossary_for(ctx),
                context=next((s.section for s in segments if s.section), None) or None,
                preserve_formatting=True,
            )
        except self._deepl.AuthorizationException as exc:
            raise ProviderConfigurationError("The DeepL API key was rejected.") from exc
        except self._deepl.DeepLException as exc:
            raise TranslationError(str(exc)) from exc
        if not isinstance(results, list):
            results = [results]
        return {s.id: r.text for s, r in zip(segments, results)}
