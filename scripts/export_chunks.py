from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_system.chunking import chunk_markdown_by_words, write_chunks_jsonl, chunk_strategy_2
from rag_system.config import load_settings


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Exporta chunks del Markdown a JSONL para inspeccion/evaluacion.")
    parser.add_argument("--markdown", type=Path, default=settings.markdown_path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks_default_220w_40o.jsonl"))
    parser.add_argument("--chunk-words", type=int, default=settings.chunk_words)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap_words)
    args = parser.parse_args()

    # chunks = chunk_markdown_by_words(args.markdown, args.chunk_words, args.chunk_overlap)
    chunks = chunk_strategy_2(args.markdown, max_chars=2500, overlap=150)
    write_chunks_jsonl(chunks, args.output)
    print(f"[RAG] Exportados {len(chunks)} chunks a {args.output}")


if __name__ == "__main__":
    main()
