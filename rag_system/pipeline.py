from __future__ import annotations

from .config import Settings, load_settings
from .retriever import RAGRetriever, format_evidence_for_prompt
from .weaviate_store import WeaviateRAGStore

_PIPELINE: "RAGPipeline | None" = None


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.store = WeaviateRAGStore(self.settings)
        self.retriever = RAGRetriever(self.store, self.settings)

    def retrieve(self, query_en: str, top_k: int | None = None, mode: str | None = None):
        return self.retriever.retrieve(query=query_en, mode=mode, top_k=top_k)

    def answer_context(self, query_en: str, top_k: int | None = None, mode: str | None = None) -> str:
        hits = self.retrieve(query_en=query_en, top_k=top_k, mode=mode)
        return format_evidence_for_prompt(hits, self.settings.max_chars_per_chunk)

    def close(self) -> None:
        self.store.close()


def get_pipeline(settings: Settings | None = None) -> RAGPipeline:
    global _PIPELINE
    if settings is not None:
        return RAGPipeline(settings)
    if _PIPELINE is None:
        _PIPELINE = RAGPipeline()
    return _PIPELINE


def call_rag_system(query_en: str, top_k: int | None = None, mode: str | None = None) -> str:
    """Devuelve evidencia documental formateada para la etapa T4a."""
    return get_pipeline().answer_context(query_en=query_en, top_k=top_k, mode=mode)
