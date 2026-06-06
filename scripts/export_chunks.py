from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_system.chunking import chunk_strategy_2, write_chunks_jsonl
from rag_system.config import load_settings


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Exporta chunks del Markdown a JSONL para inspeccion/evaluacion.")
    parser.add_argument("--markdown", type=Path, default=settings.markdown_path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks_strategy2_2500c_150o.jsonl"))
    parser.add_argument("--chunk-max-chars", type=int, default=settings.chunk_max_chars)
    parser.add_argument("--chunk-overlap-chars", type=int, default=settings.chunk_overlap_chars)
    args = parser.parse_args()

    chunks = chunk_strategy_2(args.markdown, max_chars=args.chunk_max_chars, overlap=args.chunk_overlap_chars)
    write_chunks_jsonl(chunks, args.output)
    print(f"[RAG] Exportados {len(chunks)} chunks a {args.output}")


if __name__ == "__main__":
    main()
