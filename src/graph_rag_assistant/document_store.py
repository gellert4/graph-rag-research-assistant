from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from graph_rag_assistant.config import DOCUMENTS_DIR


@dataclass
class DocumentChunk:
    doc_id: str
    source: str
    text: str
    chunk_index: int
    title: str = ""
    metadata: dict | None = None


class DocumentStore:
    def __init__(self, data_dir: Path | str = DOCUMENTS_DIR):
        self.data_dir = Path(data_dir)
        self.documents = self._load_documents()

    def _load_documents(self) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        doc_paths = sorted(self.data_dir.rglob("*.txt")) if self.data_dir.exists() else []
        for doc_path in doc_paths:
            text = doc_path.read_text(encoding="utf-8")
            split = self._segment_text(text)
            for idx, segment in enumerate(split):
                chunks.append(
                    DocumentChunk(
                        doc_id=f"{doc_path.stem}:{idx}",
                        source=doc_path.name,
                        text=segment,
                        chunk_index=idx,
                        title=doc_path.stem.replace("_", " ").title(),
                        metadata={
                            "source_file": str(doc_path),
                            "publisher": "NASA",
                            "source_type": "prompt_text",
                        },
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
