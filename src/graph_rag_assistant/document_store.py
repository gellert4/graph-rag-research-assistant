from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from graph_rag_assistant.config import DATA_DIR


@dataclass
class DocumentChunk:
    doc_id: str
    source: str
    text: str
    chunk_index: int


class DocumentStore:
    def __init__(self, data_dir: Path | str = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.documents = self._load_documents()

    def _load_documents(self) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for doc_path in sorted(self.data_dir.glob("*.txt")):
            text = doc_path.read_text(encoding="utf-8")
            split = self._segment_text(text)
            for idx, segment in enumerate(split):
                chunks.append(
                    DocumentChunk(
                        doc_id=f"{doc_path.stem}:{idx}",
                        source=doc_path.name,
                        text=segment,
                        chunk_index=idx,
                    )
                )
        return chunks

    @staticmethod
    def _segment_text(text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [text.strip()]
        return paragraphs

    def search(self, query: str, top_k: int = 3) -> List[DocumentChunk]:
        q = query.lower()
        scored = []
        for chunk in self.documents:
            score = 0
            for token in q.split():
                if token in chunk.text.lower():
                    score += chunk.text.lower().count(token)
            if score > 0:
                scored.append((score, chunk))
        return [chunk for _, chunk in sorted(scored, reverse=True)[:top_k]]
