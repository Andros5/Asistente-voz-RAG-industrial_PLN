from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

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


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Evalua BM25/vector/hybrid con un JSONL de consultas.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--modes", default="bm25,hybrid")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--collection", default=settings.collection_name)
    args = parser.parse_args()

    settings = replace(settings, collection_name=args.collection, top_k=args.top_k)
    examples = _load_dataset(args.dataset)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    pipeline = RAGPipeline(settings)
    try:
        print(f"[Eval] ejemplos={len(examples)} top_k={args.top_k} coleccion={settings.collection_name}")
        for mode in modes:
            recalls = []
            hits = []
            mrrs = []
            maps = []
            latencies = []
            for example in examples:
                start = time.perf_counter()
                results = pipeline.retrieve(example["query"], top_k=args.top_k, mode=mode)
                latencies.append(time.perf_counter() - start)
                retrieved_ids = [result.chunk_id for result in results]
                relevant_ids = set(example["relevant_chunk_ids"])
                recalls.append(recall_at_k(retrieved_ids, relevant_ids, args.top_k))
                hits.append(hit_rate_at_k(retrieved_ids, relevant_ids, args.top_k))
                mrrs.append(reciprocal_rank(retrieved_ids, relevant_ids))
                maps.append(average_precision(retrieved_ids, relevant_ids))

            print(
                f"{mode:>6} | "
                f"Recall@{args.top_k}: {_mean(recalls):.3f} | "
                f"Hit@{args.top_k}: {_mean(hits):.3f} | "
                f"MRR: {_mean(mrrs):.3f} | "
                f"MAP: {_mean(maps):.3f} | "
                f"latencia_media: {_mean(latencies):.3f}s"
            )
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
