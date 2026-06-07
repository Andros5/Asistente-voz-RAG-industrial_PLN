from __future__ import annotations

import time
from dataclasses import replace

from .config import Settings
from .embeddings import BGEEmbeddings
from .models import RetrievalTiming, RetrievedChunk
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
        hits, _ = self.retrieve_with_timing(query=query, mode=mode, top_k=top_k)
        return hits

    def retrieve_with_timing(
        self,
        query: str,
        mode: str | None = None,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTiming]:
        query = query.strip()
        selected_mode = (mode or self.settings.retrieval_mode).lower()
        selected_top_k = top_k or self.settings.top_k
        timing = RetrievalTiming(mode=selected_mode, top_k=selected_top_k)

        total_start = time.perf_counter()
        if not query:
            timing.total_s = time.perf_counter() - total_start
            return [], timing

        if selected_mode == "bm25":
            start = time.perf_counter()
            hits = self.store.bm25_search(query=query, limit=selected_top_k)
            timing.bm25_s = time.perf_counter() - start
            timing.retrieval_s = timing.bm25_s
            timing.total_s = time.perf_counter() - total_start
            return hits, timing
        if selected_mode == "vector":
            start = time.perf_counter()
            vector = self.embedder.encode_query(query)
            timing.embedding_s = time.perf_counter() - start

            start = time.perf_counter()
            hits = self.store.vector_search(vector=vector, limit=selected_top_k)
            timing.vector_s = time.perf_counter() - start
            timing.retrieval_s = timing.vector_s
            timing.total_s = time.perf_counter() - total_start
            return hits, timing
        if selected_mode == "hybrid":
            hits, timing = self._hybrid_search_with_timing(
                query=query,
                top_k=selected_top_k,
                total_start=total_start,
            )
            return hits, timing

        raise ValueError("mode debe ser uno de: bm25, vector, hybrid")

    def _hybrid_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        hits, _ = self._hybrid_search_with_timing(query=query, top_k=top_k)
        return hits

    def _hybrid_search_with_timing(
        self,
        query: str,
        top_k: int,
        total_start: float | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTiming]:
        if total_start is None:
            total_start = time.perf_counter()
        candidate_limit = max(top_k, self.settings.hybrid_candidates)
        timing = RetrievalTiming(
            mode="hybrid",
            top_k=top_k,
            candidate_limit=candidate_limit,
        )

        start = time.perf_counter()
        bm25_hits = self.store.bm25_search(query=query, limit=candidate_limit)
        timing.bm25_s = time.perf_counter() - start

        start = time.perf_counter()
        query_vector = self.embedder.encode_query(query)
        timing.embedding_s = time.perf_counter() - start

        start = time.perf_counter()
        vector_hits = self.store.vector_search(vector=query_vector, limit=candidate_limit)
        timing.vector_s = time.perf_counter() - start

        start = time.perf_counter()
        fused = self._fuse_hybrid_hits(bm25_hits, vector_hits, top_k)
        timing.fusion_s = time.perf_counter() - start
        timing.retrieval_s = timing.bm25_s + timing.vector_s + timing.fusion_s
        timing.total_s = time.perf_counter() - total_start
        return fused, timing

    def _fuse_hybrid_hits(
        self,
        bm25_hits: list[RetrievedChunk],
        vector_hits: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
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
