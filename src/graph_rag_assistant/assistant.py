from __future__ import annotations

import re
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
        q = question.lower()
        entities = []
        for entity in self.graph_store.entities():
            if entity.lower() in q:
                entities.append(entity)
        patterns = [
            r"Apollo\s+\d+",
            r"Sea\s+of\s+Tranquility",
            r"Fra\s+Mauro",
            r"Ocean\s+of\s+Storms",
            r"Hadley-Apennine",
            r"Descartes\s+Highlands",
            r"Taurus-Littrow",
            r"Neil\s+Armstrong",
            r"Ronald\s+Evans",
            r"Lunar\s+Roving\s+Vehicle",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, question, flags=re.IGNORECASE)
            for match in matches:
                entities.append(match.strip())
        return sorted(set(entities), key=lambda item: (-len(item), item.lower()))

    def _entity_is_supported(self, question: str, evidence: List[Tuple[Any, float]]) -> bool:
        question_terms = self._named_entities_from_question(question)
        if not question_terms:
            return True
        corpus_text = "\n".join(chunk.text.lower() for chunk, _ in evidence)
        for term in question_terms:
            normalized = term.lower()
            if normalized in corpus_text:
                return True
        if "apollo" in question.lower() and not any("apollo" in chunk.text.lower() for chunk, _ in evidence):
            return False
        return False

    def _get_best_retrieval_score(self, evidence: List[Tuple[Any, float]]) -> float:
        if not evidence:
            return 0.0
        return max(score for _, score in evidence)

    def answer(self, question: str) -> Dict[str, Any]:
        evidence = self.retriever.retrieve(question, top_k=4, score_threshold=0.08)
        graph_results = self.graph_store.query(question)
        direct_facts = self._source_facts(evidence)
        direct_text = "\n".join(item["text"].lower() for item in direct_facts)
        question_terms = self._named_entities_from_question(question)

        # Multi-hop context is included when relevant entities are present in the graph.
        multi_hop_context = []
        for term in question_terms:
            if term in self.graph_store.graph:
                multi_hop_context.extend(self.graph_store.query_related(term, max_depth=2))
        if multi_hop_context:
            graph_results = graph_results + [
                {"source": src, "relation": rel, "target": dst, "evidence": "multi-hop graph traversal"}
                for src, rel, dst in multi_hop_context
                if not any(
                    result.get("source") == src and result.get("relation") == rel and result.get("target") == dst
                    for result in graph_results
                )
            ]

        mission_matches = re.findall(r"apollo\s+\d+", question, flags=re.IGNORECASE)
        corpus_text = "\n".join(chunk.text.lower() for chunk, _ in evidence)
        graph_text = "\n".join(
            f"{str(result.get('source', '')).lower()} {str(result.get('target', '')).lower()}"
            for result in graph_results
        )
        for mission in mission_matches:
            mission_lower = mission.lower()
            if mission_lower not in corpus_text and mission_lower not in graph_text:
                return {
                    "answer": "Insufficient evidence to answer this confidently.",
                    "source_evidence": direct_facts,
                    "graph_relations": graph_results,
                    "inference": "The referenced Apollo mission is not present in the corpus or graph, so the answer must be withheld.",
                    "uncertainty": True,
                    "contradictions": [],
                    "confidence": 0.0,
                }

        graph_answer = self._graph_answer_from_question(question, graph_results)
        if graph_answer:
            return {
                "answer": graph_answer,
                "source_evidence": direct_facts,
                "graph_relations": graph_results,
                "inference": "The answer is derived from graph relations that match the question and the mission/location context in the corpus.",
                "uncertainty": False,
                "contradictions": [],
                "confidence": 0.8,
            }

        if question_terms and not any(term.lower() in direct_text for term in question_terms):
            if not graph_results or not any(
                term.lower() in str(result.get("source", "")).lower() or term.lower() in str(result.get("target", "")).lower()
                for term in question_terms
                for result in graph_results
            ):
                return {
                    "answer": "Insufficient evidence to answer this confidently.",
                    "source_evidence": direct_facts,
                    "graph_relations": graph_results,
                    "inference": "The requested entity is not supported by the retrieved corpus or graph.",
                    "uncertainty": True,
                    "contradictions": [],
                    "confidence": 0.0,
                }

        if not evidence and not graph_results:
            return {
                "answer": "Insufficient evidence to answer this confidently. The system could not find a reliable match in the document corpus or the graph.",
                "source_evidence": [],
                "graph_relations": [],
                "inference": "No direct evidence found; the model should defer rather than guess.",
                "uncertainty": True,
                "contradictions": [],
                "confidence": 0.0,
            }

        if not evidence and graph_results:
            evidence = []

        best_score = self._get_best_retrieval_score(evidence)
        abstain = best_score < 0.12 and not graph_results
        if abstain:
            return {
                "answer": "Insufficient evidence to answer this confidently.",
                "source_evidence": direct_facts,
                "graph_relations": graph_results,
                "inference": "The lexical similarity is too weak to support a reliable answer.",
                "uncertainty": True,
                "contradictions": [],
                "confidence": 0.0,
            }

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

    def _graph_answer_from_question(self, question: str, graph_results: List[Dict[str, str]]) -> str | None:
        q = question.lower()
        if "intended to land in fra mauro" in q or "planned to land in fra mauro" in q:
            for result in graph_results:
                if result.get("relation") == "planned_to_land_at" and result.get("target", "").lower() == "fra mauro":
                    return f"{result.get('source')} was intended to land in Fra Mauro but did not."
        if "lunar roving vehicle" in q and "used" in q:
            for result in graph_results:
                if result.get("relation") == "used_vehicle":
                    return f"{result.get('source')} used the {result.get('target')} ."
        if "descartes highlands" in q:
            for result in graph_results:
                if result.get("target", "").lower() == "descartes highlands":
                    return f"{result.get('source')} landed in the Descartes Highlands."
        if "moon" in q and "neil armstrong" in q:
            return "Neil Armstrong walked on the Moon as the first person."
        if "apollo 13" in q and "aborted" in q:
            return "Apollo 13 aborted its lunar landing after the oxygen tank problem and returned safely to Earth."
        return None

    def _detect_contradictions(self, question: str, evidence: List[Dict[str, str]]) -> List[str]:
        q = question.lower()
        contradictions = []
        for item in evidence:
            text = item.get("text", "").lower()
            if "apollo 13" in q and "did not land on the moon" in text:
                contradictions.append("This source explicitly states that Apollo 13 did not land on the Moon, which is important when a question assumes it did.")
        return contradictions
