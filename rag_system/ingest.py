from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from .chunking import chunk_markdown_by_words
from .config import Settings, load_settings
from .embeddings import BGEEmbeddings
from .weaviate_store import WeaviateRAGStore


def build_index(
    markdown_path: str | Path,
    settings: Settings | None = None,
    reset: bool = True,
    chunk_words: int | None = None,
    chunk_overlap_words: int | None = None,
) -> int:
    settings = settings or load_settings()
    if chunk_words is not None or chunk_overlap_words is not None:
        settings = replace(
            settings,
            chunk_words=chunk_words or settings.chunk_words,
            chunk_overlap_words=chunk_overlap_words
            if chunk_overlap_words is not None
            else settings.chunk_overlap_words,
        )

    chunks = chunk_markdown_by_words(
        markdown_path=markdown_path,
        chunk_words=settings.chunk_words,
        chunk_overlap_words=settings.chunk_overlap_words,
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
