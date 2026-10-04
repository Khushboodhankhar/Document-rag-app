"""Simple overlap-preserving text chunker."""
import re


class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and chunk_overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start + self.chunk_size // 2:
                    end = boundary
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            # Keep the next chunk on a word boundary. Moving back to the
            # preceding separator retains at least the requested overlap
            # without beginning a chunk in the middle of a word.
            target = end - self.chunk_overlap
            separator = text.rfind(" ", 0, target)
            start = separator + 1 if separator > 0 else target
        return chunks
