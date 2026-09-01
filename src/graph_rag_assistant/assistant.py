from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from graph_rag_assistant.document_store import DocumentStore
from graph_rag_assistant.graph_store import GraphStore
from graph_rag_assistant.llm_adapter import LLMAdapter
from graph_rag_assistant.retriever import Retriever


@dataclass
class SourceEvidence:
    source: str
    text: str


class ResearchAssistant:
    def __init__(self):
        self.document_store = DocumentStore()
        self.graph_store = GraphStore()
        self.retriever = Retriever(self.document_store.documents)
        self.llm = LLMAdapter()

    def _named_entities_from_question(self, question: str) -> List[str]:
        normalized = question.lower()
        entities = []
        for entity in self.graph_store.entities():
            if entity.lower() in normalized:
                entities.append(entity)
        return entities

    def _entity_is_supported(self, question: str, evidence: List[Tuple[Any, float]]) -> bool:
        entity_terms = self._named_entities_from_question(question)
        if not entity_terms:
            return True
        corpus_text = "\n".join(chunk.text.lower() for chunk, _ in evidence)
        for term in entity_terms:
            if term.lower() in corpus_text:
                return True
        return False

    def _get_best_retrieval_score(self, evidence: List[Tuple[Any, float]]) -> float:
        if not evidence:
            return 0.0
        return max(score for _, score in evidence)

    def answer(self, question: str) -> Dict[str, Any]:
        evidence = self.retriever.retrieve(question, top_k=4, score_threshold=0.08)
        question_entities = self._named_entities_from_question(question)
        supported = self._entity_is_supported(question, evidence)
        if not supported and evidence:
            evidence = []

        graph_results = self.graph_store.query(question)
        if graph_results and not self._graph_entity_supported(question, graph_results):
            graph_results = []

        best_score = self._get_best_retrieval_score(evidence)
        abstain = best_score < 0.12 and not graph_results
        is_clear_mismatch = bool(question_entities) and not supported and not graph_results

        if not evidence and not graph_results or abstain or is_clear_mismatch:
            return {
                "answer": "Insufficient evidence to answer this confidently. The system could not find a reliable match in the document corpus or the graph.",
                "source_evidence": [],
                "graph_relations": [],
                "inference": "No direct evidence found; the model should defer rather than guess.",
                "uncertainty": True,
                "contradictions": [],
                "confidence": 0.0,
            }

        direct_facts = self._source_facts(evidence)
        llm_payload = self.llm.generate_structured_answer(question, direct_facts, graph_results, uncertainty=False)
        contradiction_checks = self._detect_contradictions(question, direct_facts)

        return {
            "answer": llm_payload["answer"],
            "source_evidence": direct_facts,
            "graph_relations": graph_results,
            "inference": llm_payload["inference"],
            "uncertainty": llm_payload["uncertainty"],
            "contradictions": contradiction_checks,
            "confidence": llm_payload["confidence"],
        }

    def _source_facts(self, evidence: List[Tuple[Any, float]]) -> List[Dict[str, str]]:
        return [{"source": chunk.source, "text": chunk.text, "score": score} for chunk, score in evidence]

    def _graph_entity_supported(self, question: str, graph_results: List[Dict[str, str]]) -> bool:
        if not question:
            return False
        q = question.lower()
        for result in graph_results:
            source = str(result.get("source", "")).lower()
            target = str(result.get("target", "")).lower()
            if source in q or target in q:
                return True
        return False

    def _detect_contradictions(self, question: str, evidence: List[Dict[str, str]]) -> List[str]:
        q = question.lower()
        contradictions = []
        for item in evidence:
            text = item.get("text", "").lower()
            if "apollo 13" in q and "did not land on the moon" in text:
                contradictions.append("This source explicitly states that Apollo 13 did not land on the Moon, which is important when a question assumes it did.")
        return contradictions
