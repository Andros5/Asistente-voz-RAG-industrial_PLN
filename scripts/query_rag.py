from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace

from rag_system.config import load_settings
from rag_system.pipeline import RAGPipeline
from rag_system.retriever import format_evidence_for_prompt


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Consulta el sistema RAG.")
    parser.add_argument("query", help="Consulta normalizada en ingles.")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default=settings.retrieval_mode)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument("--json", action="store_true", help="Devuelve los resultados como JSON.")
    args = parser.parse_args()

    settings = replace(settings, collection_name=args.collection)
    pipeline = RAGPipeline(settings)
    try:
        hits = pipeline.retrieve(query_en=args.query, top_k=args.top_k, mode=args.mode)
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            **asdict(hit),
                            "chunk": asdict(hit.chunk),
                        }
                        for hit in hits
                    ],
                    ensure_ascii=True,
                    indent=2,
                )
            )
        else:
            print(format_evidence_for_prompt(hits, settings.max_chars_per_chunk))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
