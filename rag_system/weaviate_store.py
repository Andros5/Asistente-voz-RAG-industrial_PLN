from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from .config import Settings
from .models import DocumentChunk, RetrievedChunk


class WeaviateRAGStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client: Any | None = None

    def connect(self) -> None:
        if self.client is not None:
            return
        try:
            import weaviate
            from weaviate.classes.init import AdditionalConfig, Timeout
        except ImportError as exc:
            raise RuntimeError(
                "Falta weaviate-client. Instala las dependencias con: pip install -r requirements.txt"
            ) from exc

        self.client = weaviate.connect_to_local(
            host=self.settings.weaviate_host,
            port=self.settings.weaviate_http_port,
            grpc_port=self.settings.weaviate_grpc_port,
            additional_config=AdditionalConfig(timeout=Timeout(init=60, query=60, insert=120)),
        )
        if not self.client.is_ready():
            raise RuntimeError("Weaviate no esta listo. Arrancalo con: docker compose up -d")

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def __enter__(self) -> "WeaviateRAGStore":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def ensure_collection(self, reset: bool = False) -> None:
        self.connect()
        assert self.client is not None

        try:
            from weaviate.classes.config import Configure, DataType, Property
        except ImportError as exc:
            raise RuntimeError("Version de weaviate-client no compatible con la API v4.") from exc

        if self.client.collections.exists(self.settings.collection_name):
            if reset:
                self.client.collections.delete(self.settings.collection_name)
            else:
                return

        self.client.collections.create(
            name=self.settings.collection_name,
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="source_path", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="section", data_type=DataType.TEXT),
                Property(name="section_path", data_type=DataType.TEXT_ARRAY),
                Property(name="start_line", data_type=DataType.INT),
                Property(name="end_line", data_type=DataType.INT),
                Property(name="word_count", data_type=DataType.INT),
                Property(name="content_sha256", data_type=DataType.TEXT),
            ],
        )

    def collection_count(self) -> int:
        self.connect()
        assert self.client is not None
        collection = self.client.collections.get(self.settings.collection_name)
        aggregate = collection.aggregate.over_all(total_count=True)
        return int(aggregate.total_count or 0)

    def insert_chunks(self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("El numero de chunks y vectores debe coincidir")

        self.connect()
        assert self.client is not None
        collection = self.client.collections.get(self.settings.collection_name)

        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors, strict=True):
                batch.add_object(
                    properties=chunk.to_weaviate_properties(),
                    vector={"default": list(vector)},
                    uuid=self._uuid_for_chunk(chunk),
                )

        failed_objects = getattr(collection.batch, "failed_objects", None)
        if failed_objects:
            raise RuntimeError(f"Weaviate rechazo {len(failed_objects)} objetos durante la indexacion")

    def bm25_search(self, query: str, limit: int) -> list[RetrievedChunk]:
        self.connect()
        assert self.client is not None
        from weaviate.classes.query import MetadataQuery

        collection = self.client.collections.get(self.settings.collection_name)
        response = collection.query.bm25(
            query=query,
            query_properties=["content", "section"],
            limit=limit,
            return_metadata=MetadataQuery(score=True),
        )
        return [
            self._to_retrieved_chunk(obj, rank=rank, mode="bm25")
            for rank, obj in enumerate(response.objects, start=1)
        ]

    def vector_search(self, vector: Sequence[float], limit: int) -> list[RetrievedChunk]:
        self.connect()
        assert self.client is not None
        from weaviate.classes.query import MetadataQuery

        collection = self.client.collections.get(self.settings.collection_name)
        response = collection.query.near_vector(
            near_vector=list(vector),
            target_vector="default",
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )
        return [
            self._to_retrieved_chunk(obj, rank=rank, mode="vector")
            for rank, obj in enumerate(response.objects, start=1)
        ]

    @staticmethod
    def _uuid_for_chunk(chunk: DocumentChunk) -> str:
        key = f"{chunk.source_path}:{chunk.chunk_id}:{chunk.content_sha256}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    @staticmethod
    def _to_retrieved_chunk(obj: Any, rank: int, mode: str) -> RetrievedChunk:
        metadata = getattr(obj, "metadata", None)
        bm25_score = getattr(metadata, "score", None)
        vector_distance = getattr(metadata, "distance", None)
        chunk = DocumentChunk.from_weaviate_properties(dict(obj.properties))
        return RetrievedChunk(
            chunk=chunk,
            rank=rank,
            mode=mode,
            score=bm25_score if mode == "bm25" else vector_distance,
            bm25_score=bm25_score,
            vector_distance=vector_distance,
            bm25_rank=rank if mode == "bm25" else None,
            vector_rank=rank if mode == "vector" else None,
        )
