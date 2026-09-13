"""Tests for the LLM provider abstraction's retry/fallback logic (llm_provider.py).

Uses fake provider callers (no real network calls) so these run fast and
fully offline, and monkeypatches time.sleep so retry backoff doesn't slow
the test suite down.
"""

from __future__ import annotations

import sys
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
