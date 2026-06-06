from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from .chunking import chunk_strategy_2
from .config import Settings, load_settings
from .embeddings import BGEEmbeddings
from .models import DocumentChunk
from .weaviate_store import WeaviateRAGStore


def load_chunks_jsonl(chunks_path: str | Path) -> list[DocumentChunk]:
    path = Path(chunks_path).expanduser().resolve()
    chunks: list[DocumentChunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                properties = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalido en {path}, linea {line_no}") from exc
            chunks.append(DocumentChunk.from_weaviate_properties(properties))
    if not chunks:
        raise RuntimeError(f"No se cargaron chunks desde {path}")
    return chunks


def index_chunks(
    chunks: Sequence[DocumentChunk],
    settings: Settings | None = None,
    reset: bool = True,
) -> int:
    settings = settings or load_settings()
    if not chunks:
        raise RuntimeError("No hay chunks para indexar")

    print(f"[RAG] Chunks cargados: {len(chunks)}", flush=True)
    print(f"[RAG] Modelo de embeddings: {settings.embedding_model}", flush=True)

    # Fail fast if Docker/Weaviate is not reachable before computing embeddings.
    with WeaviateRAGStore(settings):
        pass

    embedder = BGEEmbeddings(settings)
    texts = [chunk.text for chunk in chunks]
    vectors = []
    batch_size = settings.embedding_batch_size
    for start in tqdm(range(0, len(texts), batch_size), desc="[RAG] Embeddings"):
        batch = texts[start : start + batch_size]
        vectors.extend(embedder.encode_documents(batch))

    with WeaviateRAGStore(settings) as store:
        store.ensure_collection(reset=reset)
        store.insert_chunks(chunks, vectors)
        total = store.collection_count()

    print(f"[RAG] Indexacion completada. Objetos en Weaviate: {total}", flush=True)
    return total


def build_index_from_jsonl(
    chunks_path: str | Path,
    settings: Settings | None = None,
    reset: bool = True,
) -> int:
    chunks = load_chunks_jsonl(chunks_path)
    print(f"[RAG] Fuente de chunks: {Path(chunks_path).expanduser().resolve()}", flush=True)
    return index_chunks(chunks, settings=settings, reset=reset)


def build_index_from_markdown(
    markdown_path: str | Path,
    settings: Settings | None = None,
    reset: bool = True,
    chunk_max_chars: int | None = None,
    chunk_overlap_chars: int | None = None,
) -> int:
    settings = settings or load_settings()
    if chunk_max_chars is not None or chunk_overlap_chars is not None:
        settings = replace(
            settings,
            chunk_max_chars=chunk_max_chars or settings.chunk_max_chars,
            chunk_overlap_chars=chunk_overlap_chars
            if chunk_overlap_chars is not None
            else settings.chunk_overlap_chars,
        )

    chunks = chunk_strategy_2(
        markdown_path=markdown_path,
        max_chars=settings.chunk_max_chars,
        overlap=settings.chunk_overlap_chars,
    )
    print(f"[RAG] Fuente Markdown: {Path(markdown_path).expanduser().resolve()}", flush=True)
    return index_chunks(chunks, settings=settings, reset=reset)


def build_index(
    chunks_path: str | Path,
    settings: Settings | None = None,
    reset: bool = True,
) -> int:
    return build_index_from_jsonl(chunks_path=chunks_path, settings=settings, reset=reset)
