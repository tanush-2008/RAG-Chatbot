"""Tests for the optional login / document-isolation feature (auth.py).

Uses a temporary users file so these tests never touch a real users.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import auth


@pytest.fixture(autouse=True)
def isolated_users_file(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    yield


def test_login_not_required_when_no_users_file():
    assert auth.login_required() is False


def test_add_user_then_login_required():
    auth.add_user("alice", "correct horse battery")
    assert auth.login_required() is True


def test_authenticate_correct_and_wrong_password():
    auth.add_user("alice", "correct horse battery")
    assert auth.authenticate("alice", "correct horse battery") is True
    assert auth.authenticate("alice", "wrong password") is False


def test_authenticate_unknown_user():
    assert auth.authenticate("nobody", "whatever") is False


def test_add_user_rejects_short_password():
    with pytest.raises(ValueError):
        auth.add_user("alice", "short")


def test_add_user_rejects_invalid_username():
    with pytest.raises(ValueError):
        auth.add_user("a b/c", "longenoughpassword")


def test_remove_user():
    auth.add_user("alice", "correct horse battery")
    assert auth.remove_user("alice") is True
    assert auth.authenticate("alice", "correct horse battery") is False
    assert auth.login_required() is False


def test_remove_nonexistent_user_returns_false():
    assert auth.remove_user("nobody") is False


def test_passwords_are_hashed_not_stored_plaintext():
    auth.add_user("alice", "correct horse battery")
    users = auth.load_users()
    assert "correct horse battery" not in str(users)
    assert "password_hash" in users["alice"]
    assert "salt" in users["alice"]
