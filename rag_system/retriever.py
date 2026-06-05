from __future__ import annotations

from dataclasses import replace

from .config import Settings
from .embeddings import BGEEmbeddings
from .models import RetrievedChunk
from .weaviate_store import WeaviateRAGStore


class RAGRetriever:
    def __init__(self, store: WeaviateRAGStore, settings: Settings):
        self.store = store
        self.settings = settings
        self._embedder: BGEEmbeddings | None = None

    @property
    def embedder(self) -> BGEEmbeddings:
        if self._embedder is None:
            self._embedder = BGEEmbeddings(self.settings)
        return self._embedder

    def retrieve(self, query: str, mode: str | None = None, top_k: int | None = None) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []

        selected_mode = (mode or self.settings.retrieval_mode).lower()
        selected_top_k = top_k or self.settings.top_k

        if selected_mode == "bm25":
            return self.store.bm25_search(query=query, limit=selected_top_k)
        if selected_mode == "vector":
            vector = self.embedder.encode_query(query)
            return self.store.vector_search(vector=vector, limit=selected_top_k)
        if selected_mode == "hybrid":
            return self._hybrid_search(query=query, top_k=selected_top_k)

        raise ValueError("mode debe ser uno de: bm25, vector, hybrid")

    def _hybrid_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        candidate_limit = max(top_k, self.settings.hybrid_candidates)
        bm25_hits = self.store.bm25_search(query=query, limit=candidate_limit)
        query_vector = self.embedder.encode_query(query)
        vector_hits = self.store.vector_search(vector=query_vector, limit=candidate_limit)

        by_chunk_id: dict[str, RetrievedChunk] = {}
        rrf_scores: dict[str, float] = {}

        def add_hits(hits: list[RetrievedChunk], channel: str) -> None:
            for rank, hit in enumerate(hits, start=1):
                key = hit.chunk_id
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.settings.rrf_k + rank)
                if key not in by_chunk_id:
                    by_chunk_id[key] = replace(hit)
                stored = by_chunk_id[key]
                if channel == "bm25":
                    stored.bm25_rank = rank
                    stored.bm25_score = hit.bm25_score
                else:
                    stored.vector_rank = rank
                    stored.vector_distance = hit.vector_distance

        add_hits(bm25_hits, "bm25")
        add_hits(vector_hits, "vector")

        fused: list[RetrievedChunk] = []
        for chunk_id, hit in by_chunk_id.items():
            score = rrf_scores[chunk_id]
            fused.append(
                RetrievedChunk(
                    chunk=hit.chunk,
                    rank=0,
                    mode="hybrid",
                    score=score,
                    rrf_score=score,
                    bm25_score=hit.bm25_score,
                    vector_distance=hit.vector_distance,
                    bm25_rank=hit.bm25_rank,
                    vector_rank=hit.vector_rank,
                )
            )

        fused.sort(
            key=lambda hit: (
                hit.rrf_score or 0.0,
                -(hit.bm25_rank or 10_000),
                -(hit.vector_rank or 10_000),
            ),
            reverse=True,
        )
        for rank, hit in enumerate(fused[:top_k], start=1):
            hit.rank = rank
        return fused[:top_k]


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].rstrip()
    return f"{truncated} [...]"


def format_evidence_for_prompt(hits: list[RetrievedChunk], max_chars_per_chunk: int = 1800) -> str:
    if not hits:
        return "No evidence retrieved from the manual."

    blocks: list[str] = []
    for ref_idx, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        header = (
            f"[REF:{ref_idx}] internal_id={chunk.chunk_id}; source={chunk.source}; "
            f"section={chunk.section}; lines={chunk.start_line}-{chunk.end_line}"
        )
        blocks.append(f"{header}\n{_truncate_text(chunk.text, max_chars_per_chunk)}")

    return "\n\n---\n\n".join(blocks)
