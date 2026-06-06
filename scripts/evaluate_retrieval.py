from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_system.config import load_settings
from rag_system.metrics import average_precision, hit_rate_at_k, recall_at_k, reciprocal_rank
from rag_system.pipeline import RAGPipeline


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _load_dataset(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "query" not in item:
                raise ValueError(f"Falta 'query' en linea {line_no}")
            relevant = item.get("relevant_chunk_ids") or item.get("relevant_ids")
            if not relevant:
                raise ValueError(f"Falta 'relevant_chunk_ids' en linea {line_no}")
            item["relevant_chunk_ids"] = list(relevant)
            examples.append(item)
    return examples


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _load_chunk_lookup(path: Path) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                raise ValueError(f"Falta 'chunk_id' en {path}, linea {line_no}")
            chunks[str(chunk_id)] = item
    return chunks


def _default_audit_path(dataset_path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
    return dataset_path.parent / "reports" / f"retrieval_eval_{timestamp}.json"


def _preview(text: str, max_chars: int = 500) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rsplit(' ', 1)[0].rstrip()} [...]"


def _chunk_reference(chunk: dict[str, Any] | None) -> dict[str, Any] | None:
    if chunk is None:
        return None
    return {
        "chunk_id": chunk.get("chunk_id"),
        "source": chunk.get("source"),
        "section": chunk.get("section"),
        "section_path": chunk.get("section_path", []),
        "start_line": chunk.get("start_line"),
        "end_line": chunk.get("end_line"),
        "word_count": chunk.get("word_count"),
        "content_sha256": chunk.get("content_sha256"),
        "content_preview": _preview(str(chunk.get("content", ""))),
    }


def _retrieved_chunk_reference(result, relevant_ids: set[str]) -> dict[str, Any]:
    chunk = result.chunk
    return {
        "rank": result.rank,
        "chunk_id": result.chunk_id,
        "is_relevant": result.chunk_id in relevant_ids,
        "source": chunk.source,
        "section": chunk.section,
        "section_path": chunk.section_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "word_count": chunk.word_count,
        "content_sha256": chunk.content_sha256,
        "content_preview": _preview(chunk.text),
        "score": result.score,
        "bm25_score": result.bm25_score,
        "vector_distance": result.vector_distance,
        "rrf_score": result.rrf_score,
        "bm25_rank": result.bm25_rank,
        "vector_rank": result.vector_rank,
    }


def _first_relevant_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> int | None:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return rank
    return None


def _summarize_records(records: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    recalls = [record["metrics"][f"recall@{top_k}"] for record in records]
    hits = [record["metrics"][f"hit@{top_k}"] for record in records]
    mrrs = [record["metrics"]["mrr"] for record in records]
    maps = [record["metrics"]["map"] for record in records]
    latencies = [record["latency_s"] for record in records]
    failed = [record for record in records if not record["success"]]
    partial = [record for record in records if record["success"] and not record["fully_recalled"]]
    return {
        "examples": len(records),
        "successes": len(records) - len(failed),
        "failures": len(failed),
        "partial_successes": len(partial),
        "full_recall_successes": sum(1 for record in records if record["fully_recalled"]),
        f"recall@{top_k}": _mean(recalls),
        f"hit@{top_k}": _mean(hits),
        "mrr": _mean(mrrs),
        "map": _mean(maps),
        "latency_mean_s": _mean(latencies),
        "latency_p95_s": _percentile(latencies, 0.95),
        "failed_example_indices": [record["example_index"] for record in failed],
    }


def _summarize_by_difficulty(records: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["difficulty"], []).append(record)
    return {
        difficulty: _summarize_records(difficulty_records, top_k)
        for difficulty, difficulty_records in sorted(grouped.items())
    }


def _write_audit_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Evalua BM25/vector/hybrid con un JSONL de consultas.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--modes", default="bm25,hybrid")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument("--chunks", type=Path, default=settings.chunks_path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--no-audit", action="store_true")
    args = parser.parse_args()

    settings = replace(settings, collection_name=args.collection, top_k=args.top_k)
    examples = _load_dataset(args.dataset)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    chunk_lookup = {} if args.no_audit else _load_chunk_lookup(args.chunks)
    audit_output = args.audit_output or _default_audit_path(args.dataset)

    pipeline = RAGPipeline(settings)
    try:
        print(f"[Eval] ejemplos={len(examples)} top_k={args.top_k} coleccion={settings.collection_name}")
        audit_examples: list[dict[str, Any]] = []
        mode_records: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}

        for index, example in enumerate(examples, start=1):
            relevant_ids = set(example["relevant_chunk_ids"])
            audit_example = {
                "example_index": index,
                "query": example["query"],
                "difficulty": example.get("difficulty", "unknown"),
                "notes": example.get("notes", ""),
                "relevant_chunk_ids": example["relevant_chunk_ids"],
                "relevant_chunks": [
                    _chunk_reference(chunk_lookup.get(chunk_id))
                    for chunk_id in example["relevant_chunk_ids"]
                ],
                "modes": {},
            }

            for mode in modes:
                start = time.perf_counter()
                results = pipeline.retrieve(example["query"], top_k=args.top_k, mode=mode)
                latency = time.perf_counter() - start
                retrieved_ids = [result.chunk_id for result in results]
                recall = recall_at_k(retrieved_ids, relevant_ids, args.top_k)
                hit = hit_rate_at_k(retrieved_ids, relevant_ids, args.top_k)
                mrr = reciprocal_rank(retrieved_ids, relevant_ids)
                ap = average_precision(retrieved_ids, relevant_ids)
                missing_relevant_ids = [
                    chunk_id
                    for chunk_id in example["relevant_chunk_ids"]
                    if chunk_id not in set(retrieved_ids[: args.top_k])
                ]
                mode_record = {
                    "example_index": index,
                    "query": example["query"],
                    "difficulty": example.get("difficulty", "unknown"),
                    "notes": example.get("notes", ""),
                    "success": hit > 0.0,
                    "fully_recalled": recall == 1.0,
                    "first_relevant_rank": _first_relevant_rank(retrieved_ids, relevant_ids),
                    "missing_relevant_chunk_ids": missing_relevant_ids,
                    "retrieved_chunk_ids": retrieved_ids,
                    "retrieved_chunks": [
                        _retrieved_chunk_reference(result, relevant_ids)
                        for result in results
                    ],
                    "metrics": {
                        f"recall@{args.top_k}": recall,
                        f"hit@{args.top_k}": hit,
                        "mrr": mrr,
                        "map": ap,
                    },
                    "latency_s": latency,
                }
                audit_example["modes"][mode] = mode_record
                mode_records[mode].append(mode_record)

            audit_examples.append(audit_example)

        summaries = {
            mode: {
                **_summarize_records(records, args.top_k),
                "by_difficulty": _summarize_by_difficulty(records, args.top_k),
            }
            for mode, records in mode_records.items()
        }

        for mode in modes:
            summary = summaries[mode]
            print(
                f"{mode:>6} | "
                f"Recall@{args.top_k}: {summary[f'recall@{args.top_k}']:.3f} | "
                f"Hit@{args.top_k}: {summary[f'hit@{args.top_k}']:.3f} | "
                f"MRR: {summary['mrr']:.3f} | "
                f"MAP: {summary['map']:.3f} | "
                f"latencia_media: {summary['latency_mean_s']:.3f}s | "
                f"fallos: {summary['failures']}"
            )

        if not args.no_audit:
            report = {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "scope": "T3 retrieval evaluation",
                "dataset_path": str(args.dataset.resolve()),
                "chunks_path": str(args.chunks.resolve()),
                "collection": settings.collection_name,
                "top_k": args.top_k,
                "modes": modes,
                "settings": {
                    "embedding_model": settings.embedding_model,
                    "hybrid_candidates": settings.hybrid_candidates,
                    "rrf_k": settings.rrf_k,
                },
                "summaries": summaries,
                "examples": audit_examples,
            }
            _write_audit_report(report, audit_output)
            print(f"[Eval] reporte_auditoria={audit_output.resolve()}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
