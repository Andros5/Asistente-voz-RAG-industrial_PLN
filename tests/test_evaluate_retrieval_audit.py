import json
from pathlib import Path

from scripts.evaluate_retrieval import _chunk_reference, _summarize_records, _write_audit_report


def test_summarize_records_tracks_failures_and_partial_successes():
    records = [
        {
            "example_index": 1,
            "success": True,
            "fully_recalled": True,
            "latency_s": 0.10,
            "metrics": {"recall@5": 1.0, "hit@5": 1.0, "mrr": 1.0, "map": 1.0},
        },
        {
            "example_index": 2,
            "success": True,
            "fully_recalled": False,
            "latency_s": 0.20,
            "metrics": {"recall@5": 0.5, "hit@5": 1.0, "mrr": 0.5, "map": 0.5},
        },
        {
            "example_index": 3,
            "success": False,
            "fully_recalled": False,
            "latency_s": 0.30,
            "metrics": {"recall@5": 0.0, "hit@5": 0.0, "mrr": 0.0, "map": 0.0},
        },
    ]

    summary = _summarize_records(records, top_k=5)

    assert summary["examples"] == 3
    assert summary["successes"] == 2
    assert summary["failures"] == 1
    assert summary["partial_successes"] == 1
    assert summary["full_recall_successes"] == 1
    assert summary["failed_example_indices"] == [3]
    assert summary["recall@5"] == 0.5
    assert summary["hit@5"] == 2 / 3


def test_chunk_reference_keeps_auditable_metadata():
    chunk = {
        "chunk_id": "C000001",
        "source": "manual.md",
        "section": "Alarm section",
        "section_path": ["A", "B"],
        "start_line": 10,
        "end_line": 20,
        "word_count": 30,
        "content_sha256": "abc123",
        "content": " ".join(["word"] * 200),
    }

    reference = _chunk_reference(chunk)

    assert reference["chunk_id"] == "C000001"
    assert reference["section"] == "Alarm section"
    assert reference["start_line"] == 10
    assert reference["content_sha256"] == "abc123"
    assert reference["content_preview"].endswith("[...]")


def test_write_audit_report_creates_parent_directory(tmp_path: Path):
    output = tmp_path / "nested" / "audit.json"

    _write_audit_report({"schema_version": 1, "examples": []}, output)

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "examples": [],
    }
