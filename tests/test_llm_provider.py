"""Tests for the LLM provider abstraction's retry/fallback logic (llm_provider.py).

Uses fake provider callers (no real network calls) so these run fast and
fully offline, and monkeypatches time.sleep so retry backoff doesn't slow
the test suite down.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm_provider
from llm_provider import LLMError, call_llm, is_llm_configured

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Question: hi"}]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_provider.time, "sleep", lambda _seconds: None)


@pytest.fixture
def fake_providers(monkeypatch):
    """Two fake providers ('alpha', 'beta') wired into PROVIDER_CALLERS, with
    a shared call log and per-provider failure control."""
    calls = []
    should_fail = {"alpha": False, "beta": False}

    def make_caller(name):
        def caller(messages, model, api_key):
            calls.append(name)
            if should_fail[name]:
                raise RuntimeError(f"{name} is down")
            return f"answer from {name}"
        return caller

    monkeypatch.setattr(
        llm_provider,
        "PROVIDER_CALLERS",
        {
            "alpha": (make_caller("alpha"), "ALPHA_API_KEY", "alpha-model"),
            "beta": (make_caller("beta"), "BETA_API_KEY", "beta-model"),
        },
    )
    monkeypatch.delenv("ALPHA_API_KEY", raising=False)
    monkeypatch.delenv("BETA_API_KEY", raising=False)
    return calls, should_fail


def test_no_provider_configured_uses_extractive_fallback(fake_providers, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "alpha")
    result = call_llm(MESSAGES)
    assert "No LLM API key configured" in result or "could not find" in result.lower()


def test_calls_primary_provider_when_configured(fake_providers, monkeypatch):
    calls, _ = fake_providers
    monkeypatch.setenv("LLM_PROVIDER", "alpha")
    monkeypatch.setenv("ALPHA_API_KEY", "key-a")
    result = call_llm(MESSAGES)
    assert result == "answer from alpha"
    assert calls == ["alpha"]


def test_retries_on_transient_failure_then_succeeds(fake_providers, monkeypatch):
    calls, should_fail = fake_providers
    monkeypatch.setenv("LLM_PROVIDER", "alpha")
    monkeypatch.setenv("ALPHA_API_KEY", "key-a")

    # Fail the first N-1 attempts, then succeed - simulate a transient blip.
    attempts = {"count": 0}
    caller, key_env_var, model = llm_provider.PROVIDER_CALLERS["alpha"]

    def flaky_caller(messages, model, api_key):
        attempts["count"] += 1
        if attempts["count"] < llm_provider.MAX_RETRIES + 1:
            raise RuntimeError("transient")
        return "recovered"

    llm_provider.PROVIDER_CALLERS["alpha"] = (flaky_caller, key_env_var, model)
    result = call_llm(MESSAGES)
    assert result == "recovered"
    assert attempts["count"] == llm_provider.MAX_RETRIES + 1


def test_falls_back_to_secondary_provider_when_primary_exhausted(fake_providers, monkeypatch):
    calls, should_fail = fake_providers
    monkeypatch.setenv("LLM_PROVIDER", "alpha")
    monkeypatch.setenv("ALPHA_API_KEY", "key-a")
    monkeypatch.setenv("BETA_API_KEY", "key-b")
    should_fail["alpha"] = True

    result = call_llm(MESSAGES)
    assert result == "answer from beta"
    # alpha attempted MAX_RETRIES+1 times before falling back to beta once.
    assert calls.count("alpha") == llm_provider.MAX_RETRIES + 1
    assert calls.count("beta") == 1


def test_raises_llm_error_when_all_configured_providers_fail(fake_providers, monkeypatch):
    calls, should_fail = fake_providers
    monkeypatch.setenv("LLM_PROVIDER", "alpha")
    monkeypatch.setenv("ALPHA_API_KEY", "key-a")
    monkeypatch.setenv("BETA_API_KEY", "key-b")
    should_fail["alpha"] = True
    should_fail["beta"] = True

    with pytest.raises(LLMError):
        call_llm(MESSAGES)


def test_is_llm_configured_true_if_any_provider_has_key(fake_providers, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "alpha")
    monkeypatch.delenv("ALPHA_API_KEY", raising=False)
    monkeypatch.setenv("BETA_API_KEY", "key-b")
    assert is_llm_configured() is True


def test_is_llm_configured_false_if_none_configured(fake_providers, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "alpha")
    assert is_llm_configured() is False


# --- Tests exercising _call_groq/_call_openai/_call_gemini directly -------
# The tests above stub out PROVIDER_CALLERS entirely, so they never touch
# the actual LangChain-vs-raw-SDK branching inside these functions. These
# tests stub the SDK modules themselves (via sys.modules) instead, so the
# real branching logic in llm_provider.py runs for real.


def _install_fake_module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _FakeLCResponse:
    def __init__(self, content):
        self.content = content


def test_call_groq_uses_langchain_when_available(monkeypatch):
    calls = []

    class FakeChatGroq:
        def __init__(self, model, api_key, temperature, max_tokens):
            calls.append(("init", model, api_key))

        def invoke(self, lc_messages):
            calls.append(("invoke", len(lc_messages)))
            return _FakeLCResponse("  langchain groq answer  ")

    _install_fake_module(monkeypatch, "langchain_groq", ChatGroq=FakeChatGroq)

    result = llm_provider._call_groq(MESSAGES, "test-model", "test-key")
    assert result == "langchain groq answer"
    assert calls[0] == ("init", "test-model", "test-key")


def test_call_groq_falls_back_to_raw_sdk_when_langchain_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_groq", None)  # forces ImportError
    calls = []

    class FakeMessage:
        content = "raw groq sdk answer"

    class FakeCompletions:
        def create(self, model, messages, temperature, max_tokens):
            calls.append((model, messages))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=FakeMessage())])

    class FakeGroqClient:
        def __init__(self, api_key):
            calls.append(("init", api_key))
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    _install_fake_module(monkeypatch, "groq", Groq=FakeGroqClient)

    result = llm_provider._call_groq(MESSAGES, "test-model", "test-key")
    assert result == "raw groq sdk answer"
    assert calls[0] == ("init", "test-key")


def test_call_openai_uses_langchain_when_available(monkeypatch):
    calls = []

    class FakeChatOpenAI:
        def __init__(self, model, api_key, temperature, max_tokens):
            calls.append(("init", model, api_key))

        def invoke(self, lc_messages):
            return _FakeLCResponse("langchain openai answer")

    _install_fake_module(monkeypatch, "langchain_openai", ChatOpenAI=FakeChatOpenAI)

    result = llm_provider._call_openai(MESSAGES, "test-model", "test-key")
    assert result == "langchain openai answer"
    assert calls[0] == ("init", "test-model", "test-key")


def test_call_openai_falls_back_to_raw_sdk_when_langchain_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_openai", None)
    calls = []

    class FakeMessage:
        content = "raw openai sdk answer"

    class FakeCompletions:
        def create(self, model, messages, temperature, max_tokens):
            calls.append((model, messages))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=FakeMessage())])

    class FakeOpenAIClient:
        def __init__(self, api_key):
            calls.append(("init", api_key))
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    _install_fake_module(monkeypatch, "openai", OpenAI=FakeOpenAIClient)

    result = llm_provider._call_openai(MESSAGES, "test-model", "test-key")
    assert result == "raw openai sdk answer"
    assert calls[0] == ("init", "test-key")


def test_call_gemini_uses_langchain_when_available(monkeypatch):
    calls = []

    class FakeChatGoogleGenerativeAI:
        def __init__(self, model, google_api_key, temperature, max_output_tokens):
            calls.append(("init", model, google_api_key, max_output_tokens))

        def invoke(self, lc_messages):
            return _FakeLCResponse("langchain gemini answer")

    _install_fake_module(
        monkeypatch, "langchain_google_genai", ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI
    )

    result = llm_provider._call_gemini(MESSAGES, "test-model", "test-key")
    assert result == "langchain gemini answer"
    assert calls[0] == ("init", "test-model", "test-key", 800)


def test_to_langchain_messages_maps_roles_correctly():
    from langchain_core.messages import HumanMessage, SystemMessage

    converted = llm_provider._to_langchain_messages(MESSAGES)
    assert isinstance(converted[0], SystemMessage)
    assert isinstance(converted[1], HumanMessage)
    assert converted[1].content == MESSAGES[1]["content"]
