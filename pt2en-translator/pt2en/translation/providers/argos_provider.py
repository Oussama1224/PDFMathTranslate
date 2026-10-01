"""Offline neural MT with Argos Translate (no inline markup support)."""

from __future__ import annotations

from pt2en.config import Settings
from pt2en.errors import ProviderConfigurationError
from pt2en.translation.base import (
    Segment,
    TranslationContext,
    TranslationError,
    Translator,
)


class ArgosTranslator(Translator):
    name = "argos"
    supports_markup = False
    is_llm = False

    def __init__(self, settings: Settings):
        try:
            import argostranslate.package
            import argostranslate.translate
        except ImportError as exc:
            raise ProviderConfigurationError(
                "Install the optional dependency: pip install 'pt2en-translator[argos]'"
            ) from exc
        self._translate = argostranslate.translate
        installed = {
            (p.from_code, p.to_code)
            for p in argostranslate.package.get_installed_packages()
        }
        if ("pt", "en") not in installed:
            try:
                argostranslate.package.update_package_index()
                pkg = next(
                    p
                    for p in argostranslate.package.get_available_packages()
                    if p.from_code == "pt" and p.to_code == "en"
                )
                argostranslate.package.install_from_path(pkg.download())
            except Exception as exc:  # pragma: no cover - network dependent
                raise ProviderConfigurationError(
                    "The Argos pt->en model is not installed and could not be downloaded."
                ) from exc

    def describe(self) -> str:
        return "Argos Translate (offline)"

    def translate_batch(
        self, segments: list[Segment], ctx: TranslationContext
    ) -> dict[str, str]:
        out = {}
        for s in segments:
            try:
                out[s.id] = (
                    self._translate.translate(s.text, "pt", "en")
                    if s.text.strip()
                    else s.text
                )
            except Exception as exc:  # pragma: no cover
                raise TranslationError(str(exc)) from exc
        return out
