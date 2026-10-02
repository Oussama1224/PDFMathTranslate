"""Contract tests for the OpenAI-compatible / NVIDIA provider (mocked client, no network)."""

import json
from types import SimpleNamespace

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

from pt2en.config import NVIDIA_BASE_URL  # noqa: E402
from pt2en.errors import ProviderConfigurationError  # noqa: E402
from pt2en.translation.base import (  # noqa: E402
    Segment,
    TranslationContext,
    TranslationError,
)
from pt2en.translation.providers.openai_provider import (  # noqa: E402
    OpenAICompatibleTranslator,
    parse_json_reply,
)

REPLY = {"translations": [{"id": "u0", "text": "The <b>mean</b> <m1/>."}]}


def completion(content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ]
    )


def api_error(cls, status, message):
    request = httpx.Request("POST", NVIDIA_BASE_URL + "/chat/completions")
    return cls(message, response=httpx.Response(status, request=request), body=None)


@pytest.fixture()
def nvidia(settings):
    settings.openai_api_key = "nvapi-test"
    return OpenAICompatibleTranslator(settings)


def translate(provider):
    return provider.translate_batch(
        [Segment("u0", "A <b>média</b> <m1/>.")], TranslationContext()
    )


@pytest.mark.parametrize(
    "content",
    [
        json.dumps(REPLY),
        "```json\n" + json.dumps(REPLY) + "\n```",
        "<think>The user wants English.</think>\n" + json.dumps(REPLY),
        "Reasoning without opening tag</think>```\n" + json.dumps(REPLY) + "```",
        "Here is the translation: " + json.dumps(REPLY) + " Hope this helps.",
    ],
)
def test_parse_json_reply_variants(content):
    assert parse_json_reply(content) == REPLY


def test_nvidia_key_selects_endpoint_model_and_limit(nvidia, monkeypatch):
    assert str(nvidia.client.base_url).rstrip("/") == NVIDIA_BASE_URL
    assert nvidia.describe() == "NVIDIA (meta/llama-3.3-70b-instruct)"
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return completion(json.dumps(REPLY))

    monkeypatch.setattr(nvidia.client.chat.completions, "create", create)
    assert translate(nvidia) == {"u0": "The <b>mean</b> <m1/>."}
    assert captured["model"] == "meta/llama-3.3-70b-instruct"
    assert captured["max_tokens"] == 4096
    assert captured["response_format"]["type"] == "json_schema"


def test_plain_openai_keeps_endpoint_defaults(settings):
    settings.openai_api_key = "sk-test"
    p = OpenAICompatibleTranslator(settings)
    assert "api.openai.com" in str(p.client.base_url)
    assert p.model == "gpt-4o" and p.max_tokens is None


def test_rejected_formats_fall_back_and_describe_schema(nvidia, monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if "response_format" in kwargs:
            raise api_error(
                openai.BadRequestError, 400, "response_format is not supported"
            )
        return completion("```json\n" + json.dumps(REPLY) + "\n```")

    monkeypatch.setattr(nvidia.client.chat.completions, "create", create)
    assert translate(nvidia) == {"u0": "The <b>mean</b> <m1/>."}
    assert [c.get("response_format", {}).get("type") for c in calls] == [
        "json_schema",
        "json_object",
        None,
    ]
    system = calls[-1]["messages"][0]["content"]
    assert system.count("JSON Schema") == 1 and '"translations"' in system
    # the working format is remembered
    calls.clear()
    translate(nvidia)
    assert len(calls) == 1


def test_context_overflow_asks_engine_to_split(nvidia, monkeypatch):
    def create(**kwargs):
        raise api_error(
            openai.BadRequestError, 400, "This model's maximum context length is 8192"
        )

    monkeypatch.setattr(nvidia.client.chat.completions, "create", create)
    with pytest.raises(TranslationError, match="truncated"):
        translate(nvidia)
    assert nvidia._format == "json_schema"


def test_truncated_reply(nvidia, monkeypatch):
    monkeypatch.setattr(
        nvidia.client.chat.completions,
        "create",
        lambda **kw: completion('{"translations": [', finish_reason="length"),
    )
    with pytest.raises(TranslationError, match="truncated"):
        translate(nvidia)


@pytest.mark.parametrize(
    "cls,status,expected",
    [
        (openai.AuthenticationError, 401, "rejected by NVIDIA"),
        (openai.NotFoundError, 404, "PT2EN_OPENAI_MODEL"),
    ],
)
def test_configuration_errors_fail_fast(nvidia, monkeypatch, cls, status, expected):
    def create(**kwargs):
        raise api_error(cls, status, "nope")

    monkeypatch.setattr(nvidia.client.chat.completions, "create", create)
    with pytest.raises(ProviderConfigurationError, match=expected):
        translate(nvidia)


def test_nvidia_alias_and_blank_lines(tmp_path, monkeypatch):
    from pt2en.config import Settings

    for var in ("OPENAI_API_KEY", "PT2EN_OPENAI_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "OPENAI_API_KEY=\nPT2EN_OPENAI_MODEL=\nPT2EN_OPENAI_BASE_URL=\n"
        "NVIDIA_API_KEY=nvapi-abc\n"
    )
    s = Settings(_env_file=env)
    assert s.openai_api_key == "nvapi-abc"
    assert s.openai_endpoint == NVIDIA_BASE_URL
    assert s.openai_model_name == "meta/llama-3.3-70b-instruct"
    assert s.provider_available("openai")


def test_settings_warning_names_missing_env_file(tmp_path, monkeypatch):
    from pt2en import config as config_module

    (tmp_path / ".env.txt").write_text("NVIDIA_API_KEY=nvapi-abc\n")
    monkeypatch.setattr(
        config_module, "ENV_FILES", (tmp_path / ".env", tmp_path / "sub" / ".env")
    )
    s = config_module.Settings(_env_file=None, translator="openai")
    s.openai_api_key = None
    warning = config_module.settings_warning(s)
    assert "No .env file found" in warning and "rename it to .env" in warning
