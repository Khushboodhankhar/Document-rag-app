"""File extraction and chunk creation with document provenance."""
from pathlib import Path
from typing import Iterable

from docx import Document as DocxDocument
from pypdf import PdfReader

from src.chunker import TextChunker


class DocumentLoader:
    """Read PDF, DOCX, and UTF-8 text files into chunk dictionaries."""

    def __init__(self, chunker: TextChunker | None = None) -> None:
        self.chunker = chunker or TextChunker()

    def load_files(self, paths: Iterable[Path]) -> list[dict]:
        chunks: list[dict] = []
        for path in paths:
            chunks.extend(self.load_file(Path(path)))
        return chunks

    def load_file(self, path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = self._read_pdf(path)
        elif suffix == ".docx":
            pages = [(None, self._read_docx(path))]
        elif suffix == ".txt":
            pages = [(None, self._read_txt(path))]
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use PDF, DOCX, or TXT.")

        results = []
        chunk_number = 0
        for page, text in pages:
            for chunk in self.chunker.split(text):
                results.append({"text": chunk, "filename": path.name, "page": page, "chunk_index": chunk_number})
                chunk_number += 1
        return results

    @staticmethod
    def _read_pdf(path: Path) -> list[tuple[int, str]]:
        try:
            reader = PdfReader(str(path))
            return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]
        except Exception as exc:
            raise ValueError(f"Unable to read PDF '{path.name}': {exc}") from exc

    @staticmethod
    def _read_docx(path: Path) -> str:
        try:
            return "\n".join(paragraph.text for paragraph in DocxDocument(str(path)).paragraphs)
        except Exception as exc:
            raise ValueError(f"Unable to read DOCX '{path.name}': {exc}") from exc

    @staticmethod
    def _read_txt(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"TXT file '{path.name}' must be UTF-8 encoded.") from exc
