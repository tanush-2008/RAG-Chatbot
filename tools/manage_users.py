"""CLI for managing login accounts (see auth.py for the login-gate design).

Usage:
    python tools/manage_users.py add <username> <password>
    python tools/manage_users.py remove <username>
    python tools/manage_users.py list

Creating the first user turns on the app's login gate (see auth.login_required).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import auth


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    command = sys.argv[1]

    if command == "add":
        if len(sys.argv) != 4:
            print("Usage: python tools/manage_users.py add <username> <password>")
            raise SystemExit(1)
        _, _, username, password = sys.argv
        try:
            auth.add_user(username, password)
        except ValueError as exc:
            print(f"Error: {exc}")
            raise SystemExit(1)
        print(f"User '{username}' added/updated in {auth.USERS_FILE}.")

    elif command == "remove":
        if len(sys.argv) != 3:
            print("Usage: python tools/manage_users.py remove <username>")
            raise SystemExit(1)
        username = sys.argv[2]
        if auth.remove_user(username):
            print(f"User '{username}' removed.")
        else:
            print(f"User '{username}' not found.")
            raise SystemExit(1)

    elif command == "list":
        users = auth.load_users()
        if not users:
            print("No users configured - the app is running without a login gate.")
        else:
            for username in users:
                print(username)

    else:
        print(__doc__)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
