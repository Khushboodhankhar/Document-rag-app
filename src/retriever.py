"""Retrieval boundary around the local vector store."""
from src.vector_store import FAISSVectorStore


class Retriever:
    def __init__(self, vector_store: FAISSVectorStore, k: int = 4) -> None:
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, query: str) -> list[dict]:
        if not query.strip():
            raise ValueError("Please enter a question.")
        return self.vector_store.search(query, self.k)
