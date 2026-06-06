import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sample_eval_references_existing_processed_chunks():
    chunks_path = ROOT / "data" / "processed" / "chunks_strategy2_2500c_150o.jsonl"
    eval_paths = [
        ROOT / "data" / "eval" / "eval_queries.jsonl",
        ROOT / "data" / "eval" / "eval_queries.sample.jsonl",
    ]

    chunk_ids = set()
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            chunk_ids.add(json.loads(line)["chunk_id"])

    for eval_path in eval_paths:
        with eval_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                example = json.loads(line)
                assert example["query"]
                assert set(example["relevant_chunk_ids"]).issubset(chunk_ids)
