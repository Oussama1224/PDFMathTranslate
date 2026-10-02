"""OpenAI-compatible chat-completions provider.

Works with OpenAI, NVIDIA (build.nvidia.com), Azure, OpenRouter, vLLM, Ollama...
"""

from __future__ import annotations

import json
import logging
import re
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


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_TOO_LONG = re.compile(
    r"context length|context window|maximum context|too long|too many tokens", re.I
)
# Response formats, best first; an endpoint that rejects one gets the next.
FORMATS = ("json_schema", "json_object", "prompt")


def parse_json_reply(content: str) -> Any:
    """JSON from a chat reply, tolerating reasoning blocks, code fences and chatter."""
    text = (content or "").strip()
    if "</think>" in text:  # reasoning models (the opening tag is sometimes omitted)
        text = text.rsplit("</think>", 1)[1].strip()
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


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
                "OPENAI_API_KEY / NVIDIA_API_KEY (or PT2EN_OPENAI_BASE_URL) is not set."
            )
        self._openai = openai
        self.nvidia = settings.openai_is_nvidia
        self.client = openai.OpenAI(
            api_key=settings.openai_api_key or "not-needed",
            base_url=settings.openai_endpoint,
            max_retries=4,
            timeout=600,
        )
        self.model = settings.openai_model_name
        self.max_tokens = settings.openai_max_tokens or (4096 if self.nvidia else None)
        self._format = FORMATS[0]

    def describe(self) -> str:
        return f"{'NVIDIA' if self.nvidia else 'OpenAI-compatible'} ({self.model})"

    def _call(self, system: str, user: str, schema: dict, name: str) -> Any:
        openai = self._openai
        fmt = self._format
        prompt = system
        if fmt != "json_schema":
            prompt += (
                "\n\nRespond with one JSON object only (no prose, no code fences) "
                "matching this JSON Schema:\n" + json.dumps(schema)
            )
        kwargs: dict = {}
        if fmt == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            }
        elif fmt == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                **kwargs,
            )
        except openai.BadRequestError as exc:
            message = str(exc)
            if _TOO_LONG.search(message):
                raise TranslationError(f"Response truncated: {message}") from exc
            if self.max_tokens and "max_tokens" in message.lower():
                log.info("endpoint rejected max_tokens; using its default")
                self.max_tokens = None
            elif fmt != FORMATS[-1]:
                self._format = FORMATS[FORMATS.index(fmt) + 1]
                log.info(
                    "response format %s rejected (%s); trying %s",
                    fmt,
                    message,
                    self._format,
                )
            else:
                raise TranslationError(message) from exc
            return self._call(system, user, schema, name)
        except openai.AuthenticationError as exc:
            raise ProviderConfigurationError(
                "The API key was rejected by "
                + ("NVIDIA." if self.nvidia else "the OpenAI-compatible endpoint.")
            ) from exc
        except (openai.NotFoundError, openai.PermissionDeniedError) as exc:
            raise ProviderConfigurationError(
                f"Model '{self.model}' is not available with this key/endpoint "
                f"(set PT2EN_OPENAI_MODEL): {exc}"
            ) from exc
        except openai.APIError as exc:
            raise TranslationError(str(exc)) from exc
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise TranslationError("Response truncated.")
        try:
            return parse_json_reply(choice.message.content or "")
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
