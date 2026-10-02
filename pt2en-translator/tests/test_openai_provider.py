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
    rank_chat_models,
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


def api_error(cls, status, message, body=None):
    request = httpx.Request("POST", NVIDIA_BASE_URL + "/chat/completions")
    return cls(message, response=httpx.Response(status, request=request), body=body)


CATALOG = [
    "nvidia/nv-embedqa-e5-v5",
    "meta/llama-3.3-70b-instruct",
    "deepseek-ai/deepseek-v4-pro",
    "qwen/qwen3.5-397b-a17b",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "qwen/qwen2.5-coder-32b-instruct",
    "meta/llama-3.2-11b-vision-instruct",
    "deepseek-ai/deepseek-r1",
]
RANKED = [
    "deepseek-ai/deepseek-v4-pro",
    "qwen/qwen3.5-397b-a17b",
    "meta/llama-3.3-70b-instruct",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "deepseek-ai/deepseek-r1",
]


@pytest.fixture()
def nvidia(settings, monkeypatch):
    settings.openai_api_key = "nvapi-test"
    provider = OpenAICompatibleTranslator(settings)
    monkeypatch.setattr(
        provider.client.models,
        "list",
        lambda: [SimpleNamespace(id=m) for m in CATALOG],
    )
    return provider


def gone(model):
    return api_error(
        openai.APIStatusError,
        410,
        f"Error code: 410 - {model} is gone",
        body={"detail": f"The model '{model}' has reached its end of life."},
    )


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


def test_rank_chat_models_prefers_large_general_models():
    assert rank_chat_models(CATALOG) == RANKED


def test_rank_current_nvidia_catalog():
    # chat-relevant models from build.nvidia.com's free endpoints (Oct 2026)
    catalog = [
        "nvidia/body-pose-3d",
        "deepseek-ai/deepseek-v4.1-flash",
        "nvidia/kumo-relational",
        "z-ai/glm-5.3",
        "z-ai/glm-5.3-flash",
        "nvidia/nemotron-parse-2.0",
        "nvidia/parakeet-tdt-0.6b",
        "moonshotai/kimi-k3",
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "meta/muse-glimmer-30b",
        "nvidia/riva-translate-4b-instruct-v2",
        "nvidia/nemotron-3-embed-1b",
        "poolside/laguna-xs-2.1",
        "google/diffusiongemma-26b-a4b-it",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3.5-content-safety",
        "nvidia/cosmos3-nano",
    ]
    assert rank_chat_models(catalog)[:5] == [
        "moonshotai/kimi-k3",
        "z-ai/glm-5.3",
        "z-ai/glm-5.3-flash",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "deepseek-ai/deepseek-v4.1-flash",
    ]


def test_rejected_output_limit_is_halved(nvidia, monkeypatch):
    limits = []

    def create(**kwargs):
        limits.append(kwargs.get("max_tokens"))
        if kwargs.get("max_tokens", 0) > 8192:
            raise api_error(
                openai.BadRequestError, 400, "max_tokens must be <= 8192 for this model"
            )
        return completion(json.dumps(REPLY))

    monkeypatch.setattr(nvidia.client.chat.completions, "create", create)
    translate(nvidia)
    assert limits == [16384, 8192]


def test_nvidia_key_selects_endpoint_auto_model_and_limit(nvidia, monkeypatch):
    assert str(nvidia.client.base_url).rstrip("/") == NVIDIA_BASE_URL
    assert nvidia.describe() == "NVIDIA (auto)"
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return completion(json.dumps(REPLY))

    monkeypatch.setattr(nvidia.client.chat.completions, "create", create)
    assert translate(nvidia) == {"u0": "The <b>mean</b> <m1/>."}
    assert captured["model"] == RANKED[0]
    assert nvidia.describe() == f"NVIDIA ({RANKED[0]})"
    assert captured["max_tokens"] == 16384
    assert captured["response_format"]["type"] == "json_schema"


def test_retired_model_switches_to_next_candidate(nvidia, monkeypatch):
    tried = []

    def create(**kwargs):
        tried.append(kwargs["model"])
        if kwargs["model"] in RANKED[:2]:
            raise gone(kwargs["model"])
        return completion(json.dumps(REPLY))

    monkeypatch.setattr(nvidia.client.chat.completions, "create", create)
    assert translate(nvidia) == {"u0": "The <b>mean</b> <m1/>."}
    assert tried == RANKED[:3]
    tried.clear()
    translate(nvidia)  # the working model is remembered
    assert tried == [RANKED[2]]


def test_no_working_model_is_a_configuration_error(nvidia, monkeypatch):
    def create(**kwargs):
        raise gone(kwargs["model"])

    monkeypatch.setattr(nvidia.client.chat.completions, "create", create)
    with pytest.raises(ProviderConfigurationError, match="pt2en models --check"):
        translate(nvidia)


def test_retired_configured_model_is_explained(settings, monkeypatch):
    settings.openai_api_key = "nvapi-test"
    settings.openai_model = "meta/llama-3.3-70b-instruct"
    provider = OpenAICompatibleTranslator(settings)
    calls = []

    def create(**kwargs):
        calls.append(kwargs["model"])
        raise gone(kwargs["model"])

    monkeypatch.setattr(provider.client.chat.completions, "create", create)
    with pytest.raises(ProviderConfigurationError, match="end of life") as info:
        translate(provider)
    assert "pt2en models" in str(info.value) and calls == [settings.openai_model]


def test_models_command_lists_and_checks(monkeypatch, capsys):
    from pt2en import cli
    from pt2en.translation.providers import openai_provider

    monkeypatch.setattr(
        openai_provider.OpenAICompatibleTranslator,
        "available_models",
        lambda self: CATALOG,
    )
    monkeypatch.setattr(
        openai_provider,
        "probe_model",
        lambda provider, model: "ok" if "llama" in model else "unavailable (410)",
    )
    from pt2en.config import Settings

    s = Settings(_env_file=None, openai_api_key="nvapi-test")
    code = cli.cmd_models(SimpleNamespace(check=3), s)
    out = capsys.readouterr().out
    assert code == 0
    lines = [line.strip() for line in out.splitlines() if "/" in line]
    assert lines[0] == f"{RANKED[0]}  unavailable (410)"
    assert lines[2] == f"{RANKED[2]}  ok"
    assert lines[3] == RANKED[3]  # beyond --check 3: not tested
    assert "PT2EN_OPENAI_MODEL" in out


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
    assert s.openai_model_name == "auto"
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
