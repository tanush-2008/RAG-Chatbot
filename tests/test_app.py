"""Tests for the Streamlit UI (app.py) using Streamlit's own AppTest framework.

Runs the real app.py script in a simulated harness - a genuine exercise of
the UI logic (empty state, badges, login gate, chat flow), not just a
"does it import" smoke check. Each test points VECTOR_STORE_DIR and
AUTH_USERS_FILE at isolated tmp_path locations so nothing here ever touches
the real shared vector store or a real users.json.

The embedding model is a process-level singleton (see vector_store.py), so
only the first AppTest run in this file pays its cold-load cost; later
tests in the same process reuse it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import auth

APP_TEST_TIMEOUT = 90  # generous: covers a cold embedding-model load


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Point the app at an isolated, empty vector store and disable login
    (no users.json) by default for this test."""
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "vector_store"))
    monkeypatch.delenv("AUTH_USERS_FILE", raising=False)
    return tmp_path


def _run_app(timeout: int = APP_TEST_TIMEOUT) -> AppTest:
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"))
    at.run(timeout=timeout)
    return at


def test_app_runs_without_exception(isolated_env):
    at = _run_app()
    assert not at.exception


def test_empty_state_shown_with_no_documents(isolated_env):
    at = _run_app()
    markdown_text = " ".join(str(m.value) for m in at.markdown)
    assert "Upload a document to get started" in markdown_text
    assert "0 docs" in markdown_text
    assert "0 chunks" in markdown_text


def test_status_badges_reflect_provider_and_ocr(isolated_env, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    at = _run_app()
    markdown_text = " ".join(str(m.value) for m in at.markdown)
    assert "Groq" in markdown_text
    assert "OCR" in markdown_text  # "OCR ready" or "OCR off" depending on the host


def test_chat_input_before_upload_prompts_to_upload(isolated_env):
    at = _run_app()
    assert len(at.chat_input) == 1
    at.chat_input[0].set_value("hello").run(timeout=APP_TEST_TIMEOUT)
    assert not at.exception
    markdown_text = " ".join(str(m.value) for m in at.markdown)
    assert "Please upload and process at least one PDF" in markdown_text


def test_login_gate_appears_when_users_configured(tmp_path, monkeypatch):
    users_file = tmp_path / "users.json"
    monkeypatch.setattr(auth, "USERS_FILE", users_file)
    auth.add_user("tester", "testpass123")
    monkeypatch.setenv("AUTH_USERS_FILE", str(users_file))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "vector_store"))

    at = _run_app()
    assert not at.exception
    labels = {ti.label for ti in at.text_input}
    assert {"Username", "Password"} <= labels
    # The main app content must NOT be reachable before authenticating.
    # (Note: "badge-row" as a bare substring also appears in the injected
    # <style> block's CSS selectors, so check for the actual rendered
    # element, not just the class name appearing anywhere in the page.)
    markdown_text = " ".join(str(m.value) for m in at.markdown)
    assert 'class="login-card"' in markdown_text
    assert 'class="badge-row"' not in markdown_text


def test_wrong_password_rejected_and_stays_on_login_gate(tmp_path, monkeypatch):
    users_file = tmp_path / "users.json"
    monkeypatch.setattr(auth, "USERS_FILE", users_file)
    auth.add_user("tester", "testpass123")
    monkeypatch.setenv("AUTH_USERS_FILE", str(users_file))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "vector_store"))

    at = _run_app()
    at.text_input[0].set_value("tester")
    at.text_input[1].set_value("wrong-password")
    submit = next(b for b in at.button if "Log in" in (b.label or ""))
    submit.click().run(timeout=APP_TEST_TIMEOUT)

    assert not at.exception
    assert any("Incorrect username or password" in str(e.value) for e in at.error)
    labels = {ti.label for ti in at.text_input}
    assert {"Username", "Password"} <= labels  # still on the login gate


def test_correct_login_reveals_main_app(tmp_path, monkeypatch):
    users_file = tmp_path / "users.json"
    monkeypatch.setattr(auth, "USERS_FILE", users_file)
    auth.add_user("tester", "testpass123")
    monkeypatch.setenv("AUTH_USERS_FILE", str(users_file))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "vector_store"))

    at = _run_app()
    at.text_input[0].set_value("tester")
    at.text_input[1].set_value("testpass123")
    submit = next(b for b in at.button if "Log in" in (b.label or ""))
    submit.click().run(timeout=APP_TEST_TIMEOUT)

    assert not at.exception
    markdown_text = " ".join(str(m.value) for m in at.markdown)
    assert 'class="badge-row"' in markdown_text  # main app content is now reachable
    assert "Upload a document to get started" in markdown_text
