from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from .chunking import chunk_strategy_2
from .config import Settings, load_settings
from .embeddings import BGEEmbeddings
from .weaviate_store import WeaviateRAGStore


def build_index(
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
    if not chunks:
        raise RuntimeError(f"No se generaron chunks a partir de {markdown_path}")

    print(f"[RAG] Chunks generados: {len(chunks)}", flush=True)
    print(f"[RAG] Modelo de embeddings: {settings.embedding_model}", flush=True)

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
