from pathlib import Path

from src.chunker import TextChunker
from src.document_loader import DocumentLoader
from src.prompts import NOT_FOUND_ANSWER, SYSTEM_PROMPT, build_user_prompt
from src.rag_chain import RAGChain
from src.vector_store import FAISSVectorStore


SAMPLE = Path("data/sample_company_policy.txt")


class DeterministicEmbeddings:
    """Offline-only vectors for validating FAISS mechanics without API calls."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "annual leave" in lowered or "leave allowance" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "spacecraft" in lowered:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return vectors


def test_sample_policy_loads_into_chunks():
    chunks = DocumentLoader(TextChunker(chunk_size=200, chunk_overlap=40)).load_file(SAMPLE)
    assert len(chunks) >= 3
    assert all(chunk["filename"] == SAMPLE.name for chunk in chunks)


def test_txt_uses_chunk_metadata_not_page():
    chunks = DocumentLoader(TextChunker(chunk_size=200, chunk_overlap=40)).load_file(SAMPLE)
    assert chunks[0]["page"] is None
    assert chunks[0]["chunk_index"] == 0


def test_chunker_preserves_overlap():
    chunks = TextChunker(chunk_size=24, chunk_overlap=8).split("one two three four five six seven eight nine ten")
    assert len(chunks) > 1
    assert "four" in chunks[0] and "four" in chunks[1]


def test_strict_prompt_requires_not_found_response():
    assert NOT_FOUND_ANSWER in SYSTEM_PROMPT
    assert "ONLY" in SYSTEM_PROMPT


def test_prompt_includes_question_and_source():
    prompt = build_user_prompt("How many leave days?", [{"filename": "policy.txt", "page": None, "chunk_index": 2, "text": "Annual leave is 20 days."}])
    assert "How many leave days?" in prompt
    assert "policy.txt" in prompt


def test_out_of_document_question_is_covered_by_prompt_contract():
    """A question about spacecraft is absent from the sample policy and must use the fallback."""
    policy_text = SAMPLE.read_text(encoding="utf-8").lower()
    assert "spacecraft" not in policy_text
    assert NOT_FOUND_ANSWER in SYSTEM_PROMPT


def test_faiss_returns_document_provenance(tmp_path):
    documents = [
        {"text": "Annual leave is 20 days.", "filename": "policy.txt", "page": None, "chunk_index": 0},
        {"text": "MFA is required on accounts.", "filename": "security.txt", "page": None, "chunk_index": 1},
    ]
    store = FAISSVectorStore.create_from_documents(
        documents, tmp_path / "index", embeddings=DeterministicEmbeddings()
    )
    result = store.search("What is the annual leave allowance?")
    assert result[0]["filename"] == "policy.txt"
    assert result[0]["chunk_index"] == 0
    assert result[0]["score"] == 1.0


def test_out_of_document_question_returns_required_fallback(tmp_path):
    documents = [{"text": "Annual leave is 20 days.", "filename": "policy.txt", "page": None, "chunk_index": 0}]
    store = FAISSVectorStore.create_from_documents(
        documents, tmp_path / "index", embeddings=DeterministicEmbeddings()
    )
    # The query embedding is orthogonal to the source, so generation is never called.
    chain = RAGChain(store, client=object(), min_relevance_score=0.35)
    result = chain.answer("What is the spacecraft maintenance policy?")
    assert result.answer == NOT_FOUND_ANSWER
