"""Claude (Anthropic API) translation provider — the default engine."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from pt2en.config import Settings
from pt2en.errors import ProviderConfigurationError
from pt2en.translation.base import (
    ReviewFinding,
    Segment,
    TranslationContext,
    TranslationError,
    TranslationRefused,
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
    format_glossary,
)

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicTranslator(Translator):
    name = "anthropic"
    supports_markup = True
    is_llm = True

    def __init__(self, settings: Settings):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - hard dependency
            raise ProviderConfigurationError(
                "The 'anthropic' package is not installed."
            ) from exc
        if not settings.anthropic_api_key:
            raise ProviderConfigurationError(
                "ANTHROPIC_API_KEY is not set. Configure it or choose another translation provider."
            )
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key, max_retries=4, timeout=900.0
        )
        self.model = settings.anthropic_model
        self.effort = settings.anthropic_effort
        self.max_tokens = settings.anthropic_max_tokens
        self.use_fallbacks = settings.anthropic_fallbacks.lower() == "default"
        self._lock = threading.Lock()
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "requests": 0,
        }

    def describe(self) -> str:
        return f"Claude ({self.model})"

    # ---------------------------------------------------------------- calls
    def _call(
        self, system: str, user: str, schema: dict, max_tokens: int | None = None
    ) -> Any:
        anthropic = self._anthropic
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        }
        try:
            if self.use_fallbacks:
                with self.client.beta.messages.stream(
                    betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                ) as stream:
                    message = stream.get_final_message()
            else:
                with self.client.messages.stream(**kwargs) as stream:
                    message = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise ProviderConfigurationError(
                "The Anthropic API key was rejected."
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            raise ProviderConfigurationError(
                f"The Anthropic API key has no access to model {self.model}."
            ) from exc
        except anthropic.NotFoundError as exc:
            raise ProviderConfigurationError(
                f"Unknown Anthropic model '{self.model}'."
            ) from exc
        except anthropic.BadRequestError as exc:
            if self.use_fallbacks and "fallback" in str(exc).lower():
                log.warning(
                    "Server-side fallbacks not accepted by this endpoint; disabling them"
                )
                self.use_fallbacks = False
                return self._call(system, user, schema, max_tokens)
            raise TranslationError(
                f"Request rejected by the Anthropic API: {exc}"
            ) from exc
        except anthropic.RateLimitError as exc:
            raise TranslationError("Anthropic rate limit reached; will retry.") from exc
        except anthropic.APIStatusError as exc:
            raise TranslationError(f"Anthropic API error {exc.status_code}.") from exc
        except anthropic.APIConnectionError as exc:
            raise TranslationError("Could not reach the Anthropic API.") from exc

        self._record_usage(message)
        if message.stop_reason == "refusal":
            raise TranslationRefused("The model declined to translate this content.")
        if message.stop_reason == "max_tokens":
            raise TranslationError("Response truncated (max_tokens reached).")
        text = next((b.text for b in message.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise TranslationError("The model returned invalid JSON.") from exc

    def _record_usage(self, message: Any) -> None:
        usage = getattr(message, "usage", None)
        if usage is None:
            return
        with self._lock:
            self.usage["requests"] += 1
            for key in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
                self.usage[key] += int(getattr(usage, key, 0) or 0)

    # ------------------------------------------------------------ features
    def translate_batch(
        self, segments: list[Segment], ctx: TranslationContext
    ) -> dict[str, str]:
        system = build_system_prompt(ctx)
        section = next((s.section for s in segments if s.section), "")
        user = build_batch_message(segments, ctx, section, segments[0].preceding)
        data = self._call(system, user, TRANSLATION_SCHEMA)
        return {
            str(item["id"]): str(item["text"]) for item in data.get("translations", [])
        }

    def extract_terminology(
        self, candidates, ctx: TranslationContext
    ) -> list[GlossaryEntry]:
        if not candidates:
            return []
        system = (
            "You are a terminologist preparing bilingual glossaries for translating European "
            "Portuguese academic material into English."
        )
        data = self._call(
            system, build_terminology_prompt(candidates, ctx), TERMINOLOGY_SCHEMA
        )
        out = []
        for item in data.get("terms", []):
            pt, en = str(item.get("pt", "")).strip(), str(item.get("en", "")).strip()
            if not pt or not en:
                continue
            keep = bool(item.get("keep"))
            out.append(
                GlossaryEntry(
                    pt=pt,
                    en=pt if keep else en,
                    note=str(item.get("note", "")),
                    source="document",
                    strict=True,
                    keep=keep,
                )
            )
        return out

    def review(self, pairs, ctx: TranslationContext) -> list[ReviewFinding]:
        if not pairs:
            return []
        system = REVIEW_SYSTEM
        glossary = format_glossary([e for e in ctx.glossary if e.strict])
        if glossary:
            system += "\n\nMandatory glossary:\n" + glossary
        data = self._call(system, build_review_prompt(pairs), REVIEW_SCHEMA)
        return [
            ReviewFinding(
                id=str(f.get("id")),
                severity=str(f.get("severity", "minor")),
                category=str(f.get("category", "mistranslation")),
                explanation=str(f.get("explanation", "")),
                suggestion=(
                    (str(f.get("suggestion")) or None) if f.get("suggestion") else None
                ),
            )
            for f in data.get("findings", [])
        ]
