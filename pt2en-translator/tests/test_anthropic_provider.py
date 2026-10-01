"""Contract test for the Claude provider with a mocked Anthropic client (no network)."""

import json
from types import SimpleNamespace

import pytest

from pt2en.translation.base import Segment, TranslationContext, TranslationRefused
from pt2en.translation.glossary import GlossaryEntry
from pt2en.translation.providers.anthropic_provider import (
    FALLBACK_BETA,
    AnthropicTranslator,
)


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


def message(payload, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=20, cache_read_input_tokens=80
        ),
    )


@pytest.fixture()
def provider(settings):
    settings.anthropic_api_key = "sk-test"
    return AnthropicTranslator(settings)


def test_request_shape_and_parsing(provider, monkeypatch):
    captured = {}

    def fake_stream(**kwargs):
        captured.update(kwargs)
        return FakeStream(
            message({"translations": [{"id": "u0", "text": "The <b>mean</b> <m1/>."}]})
        )

    monkeypatch.setattr(provider.client.beta.messages, "stream", fake_stream)
    ctx = TranslationContext(
        document_title="Estatística",
        glossary=[GlossaryEntry("média", "mean", strict=True)],
    )
    out = provider.translate_batch(
        [Segment("u0", "A <b>média</b> <m1/>.", section="2.1 Introdução")], ctx
    )
    assert out == {"u0": "The <b>mean</b> <m1/>."}
    assert captured["model"] == "claude-opus-5-5"
    assert captured["betas"] == [FALLBACK_BETA] and captured["fallbacks"] == "default"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert captured["output_config"]["effort"] == "high"
    system = captured["system"][0]
    assert system["cache_control"] == {"type": "ephemeral"}
    assert "média => mean [MUST]" in system["text"]
    assert "European Portuguese" in system["text"]
    user = captured["messages"][0]["content"]
    assert "2.1 Introdução" in user and '"u0"' in user
    assert provider.usage["cache_read_input_tokens"] == 80


def test_refusal_raises(provider, monkeypatch):
    monkeypatch.setattr(
        provider.client.beta.messages,
        "stream",
        lambda **kw: FakeStream(message({}, stop_reason="refusal")),
    )
    with pytest.raises(TranslationRefused):
        provider.translate_batch([Segment("u0", "Olá")], TranslationContext())


def test_fallbacks_can_be_disabled(settings, monkeypatch):
    settings.anthropic_api_key = "sk-test"
    settings.anthropic_fallbacks = "none"
    p = AnthropicTranslator(settings)
    called = {}

    def fake_stream(**kwargs):
        called.update(kwargs)
        return FakeStream(message({"translations": [{"id": "u0", "text": "Hello"}]}))

    monkeypatch.setattr(p.client.messages, "stream", fake_stream)
    assert p.translate_batch([Segment("u0", "Olá")], TranslationContext()) == {
        "u0": "Hello"
    }
    assert "fallbacks" not in called and "betas" not in called


def test_missing_key_is_a_configuration_error(settings):
    from pt2en.errors import ProviderConfigurationError

    with pytest.raises(ProviderConfigurationError):
        AnthropicTranslator(settings)
