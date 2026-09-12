"""Optional advanced feature: lightweight user login and document isolation.

Design goals:
- Zero-config by default: if no `users.json` exists, the app runs exactly as
  before (single-user, no login prompt) - this stays a valid, ungated way to
  run the assignment.
- Opt-in: creating `users.json` (via tools/manage_users.py, never by hand -
  it stores salted password hashes, not plaintext) turns on a login gate.
- Each authenticated username gets its own vector store subdirectory, so
  documents uploaded by one user are never visible to another (Module 1's
  "document access control" requirement) - see app.py's VECTOR_STORE_DIR.

This is intentionally simple (stdlib-only PBKDF2 hashing, a JSON file, no
sessions/cookies beyond Streamlit's own session state) - adequate for a
single-instance course project, not a substitute for a real identity
provider in a multi-server production deployment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from pathlib import Path

USERS_FILE = Path(os.getenv("AUTH_USERS_FILE", "users.json"))
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")
PBKDF2_ITERATIONS = 260_000


def login_required() -> bool:
    """Login is only enforced once an operator has created users.json."""
    return USERS_FILE.exists() and bool(load_users())


def load_users() -> dict[str, dict[str, str]]:
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict[str, dict[str, str]]) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    )
    return digest.hex(), salt


def add_user(username: str, password: str) -> None:
    if not _USERNAME_RE.match(username):
        raise ValueError(
            "Username must be 3-32 characters: letters, digits, underscore, dot, or hyphen."
        )
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    users = load_users()
    password_hash, salt = hash_password(password)
    users[username] = {"password_hash": password_hash, "salt": salt}
    _save_users(users)


def remove_user(username: str) -> bool:
    users = load_users()
    if username in users:
        del users[username]
        _save_users(users)
        return True
    return False


def authenticate(username: str, password: str) -> bool:
    users = load_users()
    record = users.get(username)
    if not record:
        return False
    expected_hash, salt = record["password_hash"], record["salt"]
    computed_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(computed_hash, expected_hash)
