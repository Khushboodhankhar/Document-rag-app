"""Local FAISS persistence and similarity search."""
import json
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from src.embeddings import OpenAIEmbeddings


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FAISSVectorStore:
    def __init__(
        self,
        index,
        documents: list[dict],
        path: Path,
        embeddings: EmbeddingProvider | None = None,
    ) -> None:
        self.index = index
        self.documents = documents
        self.path = Path(path)
        self.embeddings = embeddings or OpenAIEmbeddings()

    @classmethod
    def create_from_documents(
        cls, documents: list[dict], path: Path, embeddings: EmbeddingProvider | None = None
    ) -> "FAISSVectorStore":
        if not documents:
            raise ValueError("Cannot create a vector store without document chunks.")

        embeddings = embeddings or OpenAIEmbeddings()
        vectors = np.asarray(
            embeddings.embed([doc["text"] for doc in documents]),
            dtype="float32",
        )
        if vectors.ndim != 2 or len(vectors) != len(documents):
            raise ValueError("Embedding provider returned an invalid vector matrix.")

        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        store = cls(index, documents, path, embeddings)
        store.save()
        return store

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(self.path / "index.faiss"),
        )

        (self.path / "documents.json").write_text(
            json.dumps(self.documents, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, embeddings: EmbeddingProvider | None = None) -> "FAISSVectorStore":
        path = Path(path)

        index_file = path / "index.faiss"
        document_file = path / "documents.json"

        if not index_file.exists() or not document_file.exists():
            raise FileNotFoundError(
                "No local FAISS index found. Ingest documents first."
            )

        return cls(
            faiss.read_index(str(index_file)),
            json.loads(document_file.read_text(encoding="utf-8")),
            path, embeddings,
        )

    def search(self, query: str, k: int = 4) -> list[dict]:
        if not self.documents:
            return []
        vector = np.asarray(
            self.embeddings.embed([query]),
            dtype="float32",
        )
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError("Embedding provider returned an invalid query vector.")

        faiss.normalize_L2(vector)

        scores, ids = self.index.search(
            vector,
            min(k, len(self.documents)),
        )

        return [
            {
                **self.documents[idx],
                "score": float(score),
            }
            for score, idx in zip(scores[0], ids[0])
            if idx >= 0
        ]
