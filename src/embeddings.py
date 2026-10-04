"""OpenAI embedding client used by the local FAISS vector store."""
from __future__ import annotations

import os
from collections.abc import Sequence

from openai import OpenAI


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIConfigurationError(RuntimeError):
    """Raised when an OpenAI-backed operation cannot be configured safely."""


def get_api_key() -> str:
    """Read the API key from the environment without ever hardcoding it."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise OpenAIConfigurationError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )
    return api_key


class OpenAIEmbeddings:
    """Generate real OpenAI embeddings in small batches for reliable ingestion."""

    def __init__(self, model: str | None = None, client: OpenAI | None = None) -> None:
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.client = client or OpenAI(api_key=get_api_key())

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 100):
            response = self.client.embeddings.create(
                model=self.model, input=list(texts[start : start + 100])
            )
            vectors.extend(item.embedding for item in response.data)
        if len(vectors) != len(texts) or not vectors:
            raise RuntimeError("OpenAI did not return an embedding for every text chunk.")
        dimension = len(vectors[0])
        if not dimension or any(len(vector) != dimension for vector in vectors):
            raise RuntimeError("OpenAI returned embeddings with inconsistent dimensions.")
        return vectors
