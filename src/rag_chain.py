"""Strict, document-grounded RAG generation using the OpenAI Responses API."""
import os
from dataclasses import dataclass

from openai import OpenAI

from src.embeddings import get_api_key
from src.prompts import NOT_FOUND_ANSWER, SYSTEM_PROMPT, build_user_prompt
from src.retriever import Retriever
from src.vector_store import FAISSVectorStore


@dataclass
class RAGResult:
    answer: str
    sources: list[dict]


class RAGChain:
    def __init__(
        self, vector_store: FAISSVectorStore, model: str | None = None,
        min_relevance_score: float = 0.35, client: OpenAI | None = None
    ) -> None:
        self.retriever = Retriever(vector_store)
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        self.min_relevance_score = min_relevance_score
        self.client = client or OpenAI(api_key=get_api_key())

    def has_sufficient_evidence(self, sources: list[dict]) -> bool:
        return bool(sources) and sources[0]["score"] >= self.min_relevance_score

    def answer(self, question: str) -> RAGResult:
        sources = self.retriever.retrieve(question)
        # A semantic search always returns its nearest neighbours. Avoid asking
        # the model to reason from unrelated text when none is relevant enough.
        if not self.has_sufficient_evidence(sources):
            return RAGResult(answer=NOT_FOUND_ANSWER, sources=sources)
        response = self.client.responses.create(
            model=self.model, instructions=SYSTEM_PROMPT,
            input=build_user_prompt(question, sources), temperature=0,
        )
        answer = (response.output_text or "").strip()
        if not answer:
            raise RuntimeError("OpenAI returned an empty response.")
        return RAGResult(answer=answer, sources=sources)
