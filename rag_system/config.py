from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _as_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


@dataclass
class Settings:
    weaviate_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051
    collection_name: str = "Sinumerik808DChunks"

    markdown_path: Path = Path("./data/manuals/808D_ADV_diagnostics_man_0718_en-US.md")
    chunk_words: int = 220
    chunk_overlap_words: int = 40

    retrieval_mode: str = "hybrid"
    top_k: int = 5
    hybrid_candidates: int = 25
    rrf_k: int = 60
    max_chars_per_chunk: int = 1800

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = 8
    embedding_max_seq_length: int = 512
    embedding_device: str | None = None

    @property
    def markdown_path_resolved(self) -> Path:
        return self.markdown_path.resolve()


def load_settings() -> Settings:
    _load_dotenv_if_available()

    device = os.getenv("RAG_EMBEDDING_DEVICE") or None
    return Settings(
        weaviate_host=os.getenv("WEAVIATE_HOST", "localhost"),
        weaviate_http_port=_as_int("WEAVIATE_HTTP_PORT", 8080),
        weaviate_grpc_port=_as_int("WEAVIATE_GRPC_PORT", 50051),
        collection_name=os.getenv("RAG_COLLECTION_NAME", "Sinumerik808DChunks"),
        markdown_path=_as_path("RAG_MARKDOWN_PATH", "./data/manuals/808D_ADV_diagnostics_man_0718_en-US.md"),
        chunk_words=_as_int("RAG_CHUNK_WORDS", 220),
        chunk_overlap_words=_as_int("RAG_CHUNK_OVERLAP_WORDS", 40),
        retrieval_mode=os.getenv("RAG_MODE", "hybrid").lower(),
        top_k=_as_int("RAG_TOP_K", 5),
        hybrid_candidates=_as_int("RAG_HYBRID_CANDIDATES", 25),
        rrf_k=_as_int("RAG_RRF_K", 60),
        max_chars_per_chunk=_as_int("RAG_MAX_CHARS_PER_CHUNK", 1800),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
        embedding_query_prefix=os.getenv(
            "RAG_EMBEDDING_QUERY_PREFIX",
            "Represent this sentence for searching relevant passages: ",
        ),
        embedding_batch_size=_as_int("RAG_EMBEDDING_BATCH_SIZE", 8),
        embedding_max_seq_length=_as_int("RAG_EMBEDDING_MAX_SEQ_LENGTH", 512),
        embedding_device=device,
    )
