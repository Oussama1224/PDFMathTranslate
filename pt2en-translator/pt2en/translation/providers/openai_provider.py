"""OpenAI-compatible chat-completions provider.

Works with OpenAI, NVIDIA (build.nvidia.com), Azure, OpenRouter, vLLM, Ollama...
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Optional

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
_MAX_TOKENS = re.compile(
    r"max_tokens|max_completion_tokens|max_new_tokens|(output|completion) tokens", re.I
)
# Response formats, best first; an endpoint that rejects one gets the next.
FORMATS = ("json_schema", "json_object", "prompt")


# Model IDs that are not general chat/instruction models.
_NOT_CHAT = re.compile(
    r"embed|rerank|retriev|guard|safety|reward|pii|parse|ocr|vision|(^|[-_/])vl([-_]|$)"
    r"|clip|coder|code|whisper|asr|tts|audio|riva|translat|cosmos|image|diffusion"
    r"|flux|sdxl|bge|gliner|kosmos|paligemma|fuyu|neva|vila|llava|deplot|math|chatqa"
    r"|detect|segment|grounding|usd|pose|kumo|relational|animate|wan2|resolution"
    r"|ising|parakeet|canary|chatterbox|laguna|(^|[-_])base($|[-_])",
    re.I,
)
# General chat models by translation strength (largest current families first);
# within a family, full models beat flash/mini variants and newer beats older.
_PREFERRED = [
    re.compile(p, re.I)
    for p in (
        r"moonshotai/kimi",
        r"z-ai/glm",
        r"nvidia/.*nemotron.*ultra",
        r"deepseek-ai/deepseek-v\d",
        r"qwen/qwen3",
        r"mistralai/mistral-(large|medium)",
        r"meta/llama-4",
        r"meta/llama-3\.[13]-(70|405)b",
        r"nvidia/.*nemotron.*super",
        r"openai/gpt-oss",
        r"minimaxai/minimax",
        r"google/gemma-[34]",
        r"mistralai/",
        r"meta/llama",
        r"nvidia/.*nemotron",
    )
]
_SMALL = re.compile(
    r"nano|mini|small|tiny|lite|flash|lightning|\b(\d|1[0-4])b\b|\be[24]b\b", re.I
)
_REASONING = re.compile(r"(^|[-_/])r1($|[-_])|reason|think|qwq", re.I)
# Consecutive unavailable models tried before giving up in auto mode.
MAX_MODEL_SWITCHES = 8


def rank_chat_models(ids: list[str]) -> list[str]:
    """Chat models best suited to translation first (newest first within a family)."""

    def key(model: str):
        family = next(
            (i for i, p in enumerate(_PREFERRED) if p.search(model)), len(_PREFERRED)
        )
        return (
            family,
            bool(_SMALL.search(model)),
            bool(_REASONING.search(model)),
            "instruct" not in model.lower(),
            [-ord(c) for c in model],
        )

    return sorted({m for m in ids if not _NOT_CHAT.search(m)}, key=key)


def probe_model(provider: "OpenAICompatibleTranslator", model: str) -> str:
    """One tiny request to see whether a model answers: "ok" or the reason."""
    openai = provider._openai
    try:
        provider.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with: ok"}],
            max_tokens=5,
        )
        return "ok"
    except openai.APIStatusError as exc:
        label = "unavailable" if exc.status_code in (403, 404, 410) else "error"
        return f"{label} ({exc.status_code}: {_error_detail(exc)[:80]})"
    except openai.APIError as exc:
        return f"error ({str(exc)[:80]})"


def _error_detail(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        detail = body.get("detail") or (
            error.get("message") if isinstance(error, dict) else error
        )
        if detail:
            return str(detail)
    return str(exc)


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
        # "auto": pick from the models the key can use (NVIDIA retires models often)
        self.auto_model = settings.openai_model_name == "auto"
        self.model = None if self.auto_model else settings.openai_model_name
        self._candidates: list[str] = []
        self._switches = 0
        self._model_lock = threading.Lock()
        # NVIDIA's own default is too small for a batch, and large models often
        # think before answering, so leave room; halved if a model rejects it.
        self.max_tokens = settings.openai_max_tokens or (16384 if self.nvidia else None)
        self._format = FORMATS[0]
        # optional request parameters, each dropped if the endpoint rejects it
        self.optional: dict[str, Any] = {"temperature": settings.openai_temperature}
        if settings.openai_reasoning_effort:
            self.optional["reasoning_effort"] = settings.openai_reasoning_effort
        self.stream = settings.openai_stream

    def describe(self) -> str:
        name = self.model or "auto"
        return f"{'NVIDIA' if self.nvidia else 'OpenAI-compatible'} ({name})"

    # ---------------------------------------------------------------- models
    def available_models(self) -> list[str]:
        openai = self._openai
        try:
            return [m.id for m in self.client.models.list()]
        except openai.AuthenticationError as exc:
            raise ProviderConfigurationError(self._key_rejected()) from exc
        except openai.APIError as exc:
            raise TranslationError(f"Could not list models: {exc}") from exc

    def _resolve_model(self) -> str:
        with self._model_lock:
            if self.model is None:
                self._candidates = rank_chat_models(self.available_models())
                if not self._candidates:
                    raise ProviderConfigurationError(
                        "No chat models are available with this API key; "
                        "set PT2EN_OPENAI_MODEL in .env."
                    )
                self.model = self._candidates[0]
                log.info("model chosen automatically: %s", self.model)
            return self.model

    def _model_unavailable(self, model: str, exc: Exception) -> None:
        """A model was retired or is not accessible: switch (auto) or explain."""
        detail = _error_detail(exc)
        if not self.auto_model:
            raise ProviderConfigurationError(
                f"Model '{model}' is not available: {detail} "
                "Run `pt2en models` to list the models your key can use, then set "
                "PT2EN_OPENAI_MODEL in .env (or leave it empty to choose automatically)."
            ) from exc
        with self._model_lock:
            if model in self._candidates:
                self._candidates.remove(model)
            if self.model != model:
                return  # another request already switched
            self._switches += 1
            if not self._candidates or self._switches > MAX_MODEL_SWITCHES:
                raise ProviderConfigurationError(
                    f"No working chat model found (last: '{model}': {detail}). "
                    "Run `pt2en models --check` and set PT2EN_OPENAI_MODEL in .env."
                ) from exc
            self.model = self._candidates[0]
            log.warning(
                "model %s unavailable (%s); trying %s", model, detail, self.model
            )

    def _key_rejected(self) -> str:
        return "The API key was rejected by " + (
            "NVIDIA." if self.nvidia else "the OpenAI-compatible endpoint."
        )

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
        kwargs.update(self.optional)
        if self.stream:
            kwargs["stream"] = True
        model = self._resolve_model()
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
            content, finish_reason = self._read(resp)
        except openai.BadRequestError as exc:
            message = str(exc)
            rejected = next((k for k in self.optional if k in message), None)
            if _TOO_LONG.search(message):
                raise TranslationError(f"Response truncated: {message}") from exc
            if self.max_tokens and _MAX_TOKENS.search(message):
                self.max_tokens = (
                    self.max_tokens // 2 if self.max_tokens > 2048 else None
                )
                log.info("endpoint rejected the output limit; now %s", self.max_tokens)
            elif rejected:
                log.info("endpoint rejected %s; sending it no more", rejected)
                del self.optional[rejected]
            elif self.stream and re.search(r"\bstream", message, re.I):
                log.info("endpoint rejected streaming; using plain requests")
                self.stream = False
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
            raise ProviderConfigurationError(self._key_rejected()) from exc
        except openai.APIStatusError as exc:
            # 404 unknown, 410 end of life, 403 not enabled for this account
            if exc.status_code not in (403, 404, 410):
                raise TranslationError(str(exc)) from exc
            self._model_unavailable(model, exc)
            return self._call(system, user, schema, name)
        except openai.APIError as exc:
            raise TranslationError(str(exc)) from exc
        if finish_reason == "length":
            raise TranslationError("Response truncated.")
        try:
            return parse_json_reply(content)
        except json.JSONDecodeError as exc:
            raise TranslationError("Invalid JSON returned by the model.") from exc

    def _read(self, resp: Any) -> tuple[str, Optional[str]]:
        """(answer text, finish reason) from a plain or streamed completion.

        Reasoning arrives separately (delta.reasoning_content) and is ignored.
        """
        if hasattr(resp, "choices"):
            choice = resp.choices[0]
            return choice.message.content or "", choice.finish_reason
        parts: list[str] = []
        finish_reason = None
        try:
            for chunk in resp:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta is not None and choice.delta.content:
                    parts.append(choice.delta.content)
                finish_reason = choice.finish_reason or finish_reason
        except self._openai.APIError:
            raise
        except Exception as exc:  # connection dropped mid-stream
            raise TranslationError(f"Reply stream interrupted: {exc}") from exc
        return "".join(parts), finish_reason

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
