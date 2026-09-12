"""Tests for the feedback-logging module (feedback.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import feedback


@pytest.fixture(autouse=True)
def isolated_feedback_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "FEEDBACK_DIR", tmp_path / "feedback")
    monkeypatch.setattr(feedback, "FEEDBACK_LOG", tmp_path / "feedback" / "feedback_log.jsonl")
    yield


def test_read_feedback_empty_when_no_log():
    assert feedback.read_feedback() == []


def test_log_and_read_feedback_roundtrip():
    feedback.log_feedback("Q1", "A1", ["doc.pdf p.1"], "up")
    records = feedback.read_feedback()
    assert len(records) == 1
    assert records[0]["question"] == "Q1"
    assert records[0]["rating"] == "up"


def test_feedback_summary_counts():
    feedback.log_feedback("Q1", "A1", [], "up")
    feedback.log_feedback("Q2", "A2", [], "up")
    feedback.log_feedback("Q3", "A3", [], "down")
    summary = feedback.feedback_summary()
    assert summary == {"total": 3, "up": 2, "down": 1, "helpful_rate": pytest.approx(2 / 3)}


def test_feedback_summary_empty():
    summary = feedback.feedback_summary()
    assert summary == {"total": 0, "up": 0, "down": 0, "helpful_rate": None}
