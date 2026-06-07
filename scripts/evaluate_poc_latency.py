from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_system import DEFAULT_LLM_MODEL_ID
from poc_system import PoCRunner


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PHASE_KEYS = [
    "t2a_s",
    "t2b_s",
    "t3_s",
    "t3_embedding_s",
    "t3_retrieval_s",
    "t3_bm25_s",
    "t3_vector_s",
    "t3_fusion_s",
    "t4a_s",
    "t4b_s",
    "total_s",
]


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _load_queries(path: Path | None, inline_queries: list[str] | None) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    if inline_queries:
        for index, query in enumerate(inline_queries, start=1):
            queries.append({"id": f"inline-{index}", "query_es": query})

    if path is not None:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    item = json.loads(line)
                    query = item.get("query_es") or item.get("query")
                    if not query:
                        raise ValueError(f"Falta 'query_es' o 'query' en {path}, linea {line_no}")
                    item["query_es"] = str(query)
                    item.setdefault("id", f"file-{line_no}")
                    queries.append(item)
                else:
                    queries.append({"id": f"file-{line_no}", "query_es": line})

    if not queries:
        raise ValueError("Indica al menos --query o --queries-file")
    return queries


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_s": 0.0, "p50_s": 0.0, "p95_s": 0.0, "min_s": 0.0, "max_s": 0.0}
    return {
        "mean_s": _mean(values),
        "p50_s": _percentile(values, 0.50),
        "p95_s": _percentile(values, 0.95),
        "min_s": min(values),
        "max_s": max(values),
    }


def _default_output_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
    return Path("./data/eval/reports") / f"poc_latency_{timestamp}.json"


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "examples": len(records),
        "phases": {
            phase: _stats([record["timings"][phase] for record in records])
            for phase in PHASE_KEYS
        },
    }


def _record_from_result(index: int, item: dict[str, Any], result, omit_outputs: bool) -> dict[str, Any]:
    record = {
        "example_index": index,
        "id": item.get("id"),
        "query_es": item["query_es"],
        "notes": item.get("notes", ""),
        "timings": result.timings.as_dict(),
    }
    if not omit_outputs:
        record["outputs"] = {
            "normalized_query_es": result.normalized_query_es,
            "normalized_query_en": result.normalized_query_en,
            "retrieved_evidence": result.retrieved_evidence,
            "answer_en": result.answer_en,
            "final_answer_es": result.final_answer_es,
        }
    return record


def _write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    _load_dotenv_if_available()
    parser = argparse.ArgumentParser(description="Audita latencias del ciclo completo T2a -> T2b -> T3 -> T4a -> T4b.")
    parser.add_argument("--query", action="append", help="Consulta en espanol. Puede repetirse.")
    parser.add_argument("--queries-file", type=Path, help="JSONL o TXT con consultas en espanol.")
    parser.add_argument("--output", type=Path, default=_default_output_path())
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default=os.getenv("RAG_MODE", "hybrid"))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("RAG_TOP_K", "5")))
    parser.add_argument("--model-id", default=os.getenv("LLM_MODEL_ID", DEFAULT_LLM_MODEL_ID))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("LLM_MAX_NEW_TOKENS", "512")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LLM_TEMPERATURE", "0.0")))
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default=os.getenv("LLM_DTYPE", "auto"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--print-steps", action="store_true", help="Imprime salidas intermedias de cada consulta.")
    parser.add_argument("--omit-outputs", action="store_true", help="Guarda solo tiempos, sin textos generados.")
    args = parser.parse_args()

    queries = _load_queries(args.queries_file, args.query)
    runner = PoCRunner(
        mode=args.mode,
        top_k=args.top_k,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        warmup=True,
    )

    print(f"[Latency] cargando modelo={args.model_id}", flush=True)
    load_start = time.perf_counter()
    runner.load(print_steps=True)
    initial_load_s = time.perf_counter() - load_start

    records: list[dict[str, Any]] = []
    for index, item in enumerate(queries, start=1):
        print(f"[Latency] {index}/{len(queries)} id={item.get('id')} query={item['query_es']}", flush=True)
        result = runner.run(
            item["query_es"],
            stream_final=False,
            print_steps=args.print_steps,
            print_timings=False,
        )
        records.append(_record_from_result(index, item, result, args.omit_outputs))

    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "T2-T4 latency audit",
        "mode": args.mode,
        "top_k": args.top_k,
        "model_id": args.model_id,
        "local_files_only": args.local_files_only,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "dtype": args.dtype,
        "initial_load_s": initial_load_s,
        "summary": _summarize(records),
        "examples": records,
    }
    _write_report(report, args.output)

    total_stats = report["summary"]["phases"]["total_s"]
    print(
        "[Latency] "
        f"consultas={len(records)} "
        f"load={initial_load_s:.3f}s "
        f"total_mean={total_stats['mean_s']:.3f}s "
        f"total_p95={total_stats['p95_s']:.3f}s",
        flush=True,
    )
    print(f"[Latency] reporte={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
