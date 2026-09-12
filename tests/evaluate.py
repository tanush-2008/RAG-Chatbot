"""Optional advanced feature: automated evaluation against a prepared QA dataset.

Runs every question in tests/test_questions.csv through the real pipeline
(sample PDFs in documents/), checks retrieval against the expected source and
refusal behavior against the expected answerability, and writes a filled-in
results CSV plus a summary - covering the brief's Testing & Evaluation
section (retrieval accuracy, groundedness, refusal quality, response time).

Usage:
    python tests/evaluate.py

Output:
    tests/evaluation_results.csv
"""

from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from rag_pipeline import RAGPipeline

DOCS_DIR = Path(__file__).resolve().parent.parent / "documents"
QUESTIONS_CSV = Path(__file__).resolve().parent / "test_questions.csv"
RESULTS_CSV = Path(__file__).resolve().parent / "evaluation_results.csv"

_EXPECTED_SOURCE_RE = re.compile(r"([^,]+\.pdf)\s*,\s*page\s*(\d+)", re.IGNORECASE)


def _load_sample_files() -> list[tuple[str, bytes, int]]:
    files = []
    for path in sorted(DOCS_DIR.glob("*.pdf")):
        data = path.read_bytes()
        files.append((path.name, data, len(data)))
    return files


def _parse_expected(expected_source: str) -> tuple[str, int] | None:
    match = _EXPECTED_SOURCE_RE.search(expected_source)
    if not match:
        return None
    return match.group(1).strip(), int(match.group(2))


def run_evaluation() -> list[dict]:
    pipeline = RAGPipeline()
    pipeline.process_documents(_load_sample_files())

    with open(QUESTIONS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    for row in rows:
        question = row["question"]
        expected_answerable = row["expected_answerable"].strip().lower() == "yes"
        expected = _parse_expected(row["expected_source"])

        start = time.perf_counter()
        answer = pipeline.ask(question)
        elapsed = time.perf_counter() - start

        retrieved = "; ".join(f"{s.document}, page {s.page}" for s in answer.sources) or "Not available"

        if expected_answerable:
            correct = answer.grounded and expected is not None and any(
                s.document == expected[0] and s.page == expected[1] for s in answer.sources
            )
        else:
            correct = not answer.grounded

        results.append(
            {
                "question": question,
                "expected_source": row["expected_source"],
                "expected_answerable": row["expected_answerable"],
                "retrieved_source": retrieved,
                "grounded": answer.grounded,
                "correct": correct,
                "response_time_seconds": round(elapsed, 3),
            }
        )

    return results


def write_results(results: list[dict]) -> None:
    fieldnames = [
        "question",
        "expected_source",
        "expected_answerable",
        "retrieved_source",
        "grounded",
        "correct",
        "response_time_seconds",
    ]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict]) -> None:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    answerable = [r for r in results if r["expected_answerable"].strip().lower() == "yes"]
    unanswerable = [r for r in results if r["expected_answerable"].strip().lower() != "yes"]
    retrieval_accuracy = (
        sum(1 for r in answerable if r["correct"]) / len(answerable) if answerable else None
    )
    refusal_accuracy = (
        sum(1 for r in unanswerable if r["correct"]) / len(unanswerable) if unanswerable else None
    )
    avg_time = sum(r["response_time_seconds"] for r in results) / total if total else 0

    print(f"\n{'=' * 60}")
    print(f"Evaluation results ({total} questions)")
    print(f"{'=' * 60}")
    print(f"Overall accuracy:    {correct}/{total} ({correct / total:.0%})")
    if retrieval_accuracy is not None:
        print(f"Retrieval accuracy:  {retrieval_accuracy:.0%} ({len(answerable)} answerable questions)")
    if refusal_accuracy is not None:
        print(f"Refusal accuracy:    {refusal_accuracy:.0%} ({len(unanswerable)} unanswerable questions)")
    print(f"Avg response time:   {avg_time:.2f}s")
    print(f"\nFull results written to {RESULTS_CSV}")

    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n{len(failures)} question(s) did not match expectations:")
        for r in failures:
            print(f"  - {r['question']!r}: expected {r['expected_source']!r}, got {r['retrieved_source']!r}")


if __name__ == "__main__":
    results = run_evaluation()
    write_results(results)
    print_summary(results)
