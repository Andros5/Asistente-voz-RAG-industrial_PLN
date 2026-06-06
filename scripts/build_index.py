from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from rag_system.config import load_settings
from rag_system.ingest import build_index


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Construye el indice RAG en Weaviate.")
    parser.add_argument("--markdown", type=Path, default=settings.markdown_path)
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument("--chunk-max-chars", type=int, default=settings.chunk_max_chars)
    parser.add_argument("--chunk-overlap-chars", type=int, default=settings.chunk_overlap_chars)
    parser.add_argument("--no-reset", action="store_true", help="No borra la coleccion existente.")
    args = parser.parse_args()

    settings = replace(settings, collection_name=args.collection)
    build_index(
        markdown_path=args.markdown,
        settings=settings,
        reset=not args.no_reset,
        chunk_max_chars=args.chunk_max_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
    )


if __name__ == "__main__":
    main()
