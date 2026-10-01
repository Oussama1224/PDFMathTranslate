"""Provider registry: maps provider names to implementations."""

from __future__ import annotations

from typing import Callable

from pt2en.config import Settings
from pt2en.errors import ProviderConfigurationError
from pt2en.translation.base import Translator


def _anthropic(settings: Settings) -> Translator:
    from pt2en.translation.providers.anthropic_provider import AnthropicTranslator

    return AnthropicTranslator(settings)


def _openai(settings: Settings) -> Translator:
    from pt2en.translation.providers.openai_provider import OpenAICompatibleTranslator

    return OpenAICompatibleTranslator(settings)


def _deepl(settings: Settings) -> Translator:
    from pt2en.translation.providers.deepl_provider import DeepLTranslator

    return DeepLTranslator(settings)


def _argos(settings: Settings) -> Translator:
    from pt2en.translation.providers.argos_provider import ArgosTranslator

    return ArgosTranslator(settings)


def _demo(settings: Settings) -> Translator:
    from pt2en.translation.providers.demo_provider import DemoTranslator

    return DemoTranslator(settings)


PROVIDERS: dict[str, tuple[str, Callable[[Settings], Translator]]] = {
    "anthropic": ("Claude (Anthropic) — recommended", _anthropic),
    "openai": ("OpenAI-compatible API", _openai),
    "deepl": ("DeepL", _deepl),
    "argos": ("Argos Translate (offline)", _argos),
    "demo": ("Offline demo dictionary (testing only)", _demo),
}
ALIASES = {"claude": "anthropic", "mock": "demo"}


def create_translator(name: str, settings: Settings) -> Translator:
    key = ALIASES.get(name, name)
    if key not in PROVIDERS:
        raise ProviderConfigurationError(f"Unknown translation provider '{name}'.")
    return PROVIDERS[key][1](settings)


def list_providers(settings: Settings) -> list[dict]:
    default = ALIASES.get(settings.translator, settings.translator)
    return [
        {
            "name": key,
            "label": label,
            "available": settings.provider_available(key),
            "default": key == default,
        }
        for key, (label, _) in PROVIDERS.items()
    ]
