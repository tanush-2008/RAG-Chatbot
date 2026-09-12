"""Optional advanced feature: feedback buttons for useful/incorrect answers.

Logs each 👍/👎 vote as one JSON line to `feedback/feedback_log.jsonl` (kept
out of git - it's user-generated data, not source code) so the ratings can
later be reviewed or folded into an evaluation report.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

FEEDBACK_DIR = Path("feedback")
FEEDBACK_LOG = FEEDBACK_DIR / "feedback_log.jsonl"


def log_feedback(
    question: str,
    answer: str,
    sources: list[str],
    rating: str,
    comment: str = "",
) -> None:
    """Append one feedback record. `rating` is 'up' or 'down'."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "question": question,
        "answer": answer,
        "sources": sources,
        "rating": rating,
        "comment": comment,
    }
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_feedback() -> list[dict]:
    """Read all logged feedback records, oldest first."""
    if not FEEDBACK_LOG.exists():
        return []
    records = []
    with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def feedback_summary() -> dict:
    """Aggregate counts for a quick at-a-glance quality signal."""
    records = read_feedback()
    up = sum(1 for r in records if r["rating"] == "up")
    down = sum(1 for r in records if r["rating"] == "down")
    total = up + down
    return {
        "total": total,
        "up": up,
        "down": down,
        "helpful_rate": (up / total) if total else None,
    }
