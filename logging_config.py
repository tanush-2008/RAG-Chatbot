"""Shared logging setup for the backend modules (not the Streamlit UI, which
has its own on-screen feedback via st.status/st.spinner/st.error).

Usage: `from logging_config import get_logger; log = get_logger(__name__)`
"""

from __future__ import annotations

import logging
import os

_configured = False


def _configure_once() -> None:
    global _configured
    if _configured:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_once()
    return logging.getLogger(name)
