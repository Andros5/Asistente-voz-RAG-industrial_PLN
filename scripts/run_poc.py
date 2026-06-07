from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from llm_system import DEFAULT_LLM_MODEL_ID
from poc_system import run_poc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _write_timings_report(output_path: Path, result, args: argparse.Namespace) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "single-query T2-T4 latency audit",
        "mode": args.mode,
        "top_k": args.top_k,
        "model_id": args.model_id,
        "local_files_only": args.local_files_only,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "dtype": args.dtype,
        "query": result.original_query_es,
        "outputs": {
            "normalized_query_es": result.normalized_query_es,
            "normalized_query_en": result.normalized_query_en,
            "retrieved_evidence": result.retrieved_evidence,
            "answer_en": result.answer_en,
            "final_answer_es": result.final_answer_es,
        },
        "timings": result.timings.as_dict(),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    _load_dotenv_if_available()
    parser = argparse.ArgumentParser(description="Ejecuta la PoC completa T2a -> T2b -> T3 -> T4a -> T4b.")
    parser.add_argument("--query", required=True, help="Consulta original en espanol.")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default=os.getenv("RAG_MODE", "hybrid"))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("RAG_TOP_K", "5")))
    parser.add_argument("--model-id", default=os.getenv("LLM_MODEL_ID", DEFAULT_LLM_MODEL_ID))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("LLM_MAX_NEW_TOKENS", "512")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LLM_TEMPERATURE", "0.0")))
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default=os.getenv("LLM_DTYPE", "auto"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--stream-final", action="store_true", help="Emite la respuesta final con streaming.")
    parser.add_argument("--show-timings", action="store_true", help="Imprime latencias por fase T2-T4.")
    parser.add_argument("--timings-output", type=Path, help="Guarda un JSON auditable con salidas y latencias.")
    args = parser.parse_args()

    result = run_poc(
        args.query,
        mode=args.mode,
        top_k=args.top_k,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        stream_final=args.stream_final,
        print_steps=True,
        print_timings=args.show_timings,
    )
    if args.timings_output:
        _write_timings_report(args.timings_output, result, args)
        print(f"[Timings] reporte={args.timings_output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
