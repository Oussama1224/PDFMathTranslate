"""OpenAI-compatible chat-completions provider (OpenAI, Azure, OpenRouter, vLLM, Ollama...)."""

from __future__ import annotations

import json
import logging
from typing import Any

from pt2en.config import Settings
from pt2en.errors import ProviderConfigurationError
from pt2en.translation.base import (
    ReviewFinding,
    Segment,
    TranslationContext,
    TranslationError,
    Translator,
)
from pt2en.translation.glossary import GlossaryEntry
from pt2en.translation.prompts import (
    REVIEW_SCHEMA,
    REVIEW_SYSTEM,
    TERMINOLOGY_SCHEMA,
    TRANSLATION_SCHEMA,
    build_batch_message,
    build_review_prompt,
    build_system_prompt,
    build_terminology_prompt,
)

log = logging.getLogger(__name__)


class OpenAICompatibleTranslator(Translator):
    name = "openai"
    supports_markup = True
    is_llm = True

    def __init__(self, settings: Settings):
        try:
            import openai
        except ImportError as exc:
            raise ProviderConfigurationError(
                "Install the optional dependency: pip install 'pt2en-translator[openai]'"
            ) from exc
        if not (settings.openai_api_key or settings.openai_base_url):
            raise ProviderConfigurationError(
                "OPENAI_API_KEY (or PT2EN_OPENAI_BASE_URL) is not set."
            )
        self._openai = openai
        self.client = openai.OpenAI(
            api_key=settings.openai_api_key or "not-needed",
            base_url=settings.openai_base_url or None,
            max_retries=4,
            timeout=600,
        )
        self.model = settings.openai_model
        self._schema_mode = True

    def describe(self) -> str:
        return f"OpenAI-compatible ({self.model})"

    def _call(self, system: str, user: str, schema: dict, name: str) -> Any:
        openai = self._openai
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        fmt: dict = (
            {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            }
            if self._schema_mode
            else {"type": "json_object"}
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=fmt,
                temperature=0.1,
            )
        except openai.BadRequestError as exc:
            if self._schema_mode:
                log.info(
                    "json_schema not supported by endpoint; falling back to json_object"
                )
                self._schema_mode = False
                return self._call(
                    system + "\nRespond with JSON only.", user, schema, name
                )
            raise TranslationError(str(exc)) from exc
        except openai.AuthenticationError as exc:
            raise ProviderConfigurationError(
                "The OpenAI API key was rejected."
            ) from exc
        except openai.APIError as exc:
            raise TranslationError(str(exc)) from exc
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise TranslationError("Response truncated.")
        try:
            return json.loads(choice.message.content or "{}")
        except json.JSONDecodeError as exc:
            raise TranslationError("Invalid JSON returned by the model.") from exc

    def translate_batch(
        self, segments: list[Segment], ctx: TranslationContext
    ) -> dict[str, str]:
        section = next((s.section for s in segments if s.section), "")
        data = self._call(
            build_system_prompt(ctx),
            build_batch_message(segments, ctx, section, segments[0].preceding),
            TRANSLATION_SCHEMA,
            "translations",
        )
        return {str(i["id"]): str(i["text"]) for i in data.get("translations", [])}

    def extract_terminology(
        self, candidates, ctx: TranslationContext
    ) -> list[GlossaryEntry]:
        if not candidates:
            return []
        data = self._call(
            "You are a terminologist for European Portuguese to English translation.",
            build_terminology_prompt(candidates, ctx),
            TERMINOLOGY_SCHEMA,
            "terms",
        )
        out = []
        for t in data.get("terms", []):
            pt, en = str(t.get("pt", "")).strip(), str(t.get("en", "")).strip()
            if pt and en:
                keep = bool(t.get("keep"))
                out.append(
                    GlossaryEntry(
                        pt=pt,
                        en=pt if keep else en,
                        note=str(t.get("note", "")),
                        source="document",
                        strict=True,
                        keep=keep,
                    )
                )
        return out

    def review(self, pairs, ctx: TranslationContext) -> list[ReviewFinding]:
        if not pairs:
            return []
        data = self._call(
            REVIEW_SYSTEM, build_review_prompt(pairs), REVIEW_SCHEMA, "review"
        )
        return [
            ReviewFinding(
                id=str(f.get("id")),
                severity=str(f.get("severity", "minor")),
                category=str(f.get("category", "mistranslation")),
                explanation=str(f.get("explanation", "")),
                suggestion=f.get("suggestion") or None,
            )
            for f in data.get("findings", [])
        ]
