from __future__ import annotations

import os
from collections.abc import Sequence

from .config import Settings


class BGEEmbeddings:
    """Embeddings BGE para recuperacion densa en T3."""

    def __init__(self, settings: Settings):
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = settings.embedding_model
        self.batch_size = settings.embedding_batch_size
        self.max_seq_length = settings.embedding_max_seq_length
        self.query_prefix = settings.embedding_query_prefix
        self.device = settings.embedding_device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)

    def encode_query(self, query: str) -> list[float]:
        prefixed_query = f"{self.query_prefix}{query}" if self.query_prefix else query
        return self._encode([prefixed_query])[0]

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        import torch
        import torch.nn.functional as F

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state[:, 0]
                embeddings = F.normalize(embeddings, p=2, dim=1)

            vectors.extend(embeddings.detach().cpu().float().tolist())

        return vectors
