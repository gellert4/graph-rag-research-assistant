from __future__ import annotations

import os
from typing import Any, Dict, List

import requests


class OpenAILLMAdapter:
    """Thin OpenAI-compatible adapter for grounded response generation."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini", base_url: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, question: str, evidence: List[Dict[str, str]], graph: List[Dict[str, Any]], uncertainty: bool = False) -> Dict[str, Any]:
        if not self.is_available():
            return {
                "answer": "LLM unavailable. Falling back to grounded evidence only.",
                "facts": evidence,
                "graph_relations": graph,
                "inference": "The model is not configured; the system must abstain from unsupported claims.",
                "uncertainty": True,
                "confidence": 0.0,
            }

        system_prompt = (
            "You are a careful research assistant. Answer only from the provided evidence. "
            "Separate facts, graph relations, and inference. If the evidence is insufficient, say so."
        )
        prompt = self._build_prompt(question, evidence, graph)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return self._parse_response(content)

    def _build_prompt(self, question: str, evidence: List[Dict[str, str]], graph: List[Dict[str, Any]]) -> str:
        evidence_block = "\n".join(f"- {item['source']}: {item['text']}" for item in evidence)
        graph_block = "\n".join(f"- {item['source']} {item['relation']} {item['target']}" for item in graph)
        return (
            f"Question: {question}\n\n"
            f"Evidence:\n{evidence_block or 'No direct evidence'}\n\n"
            f"Graph relations:\n{graph_block or 'No graph evidence'}\n\n"
            "Return JSON with keys: answer, facts, graph_relations, inference, uncertainty, confidence.\n"
            "The answer must not invent anything missing from the context."
        )

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        parsed = __import__("json").loads(raw)
        return {
            "answer": parsed.get("answer", "Insufficient evidence to answer this confidently."),
            "facts": parsed.get("facts", []),
            "graph_relations": parsed.get("graph_relations", []),
            "inference": parsed.get("inference", "No inference beyond the given evidence."),
            "uncertainty": bool(parsed.get("uncertainty", True)),
            "confidence": float(parsed.get("confidence", 0.0)),
        }
