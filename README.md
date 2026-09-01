# GraphRAG Research Assistant

A compact, explainable research assistant built around a small Apollo-era knowledge base.

## Overview

This project demonstrates a practical LLM + RAG + graph workflow that can answer research questions about a controlled domain without relying on a huge dataset. The repository combines:

- a document store with curated source text,
- a lightweight retrieval layer,
- a graph of entities and relationships,
- an answer component that distinguishes evidence, graph links, and model inference.

## Directory structure

- `data/` – curated source documents and metadata
- `src/graph_rag_assistant/` – application code
- `tests/` – evaluation set and sample checks

## Planned implementation

1. Build a small, domain-specific corpus.
2. Create a graph of entities and relationships.
3. Retrieve relevant evidence from text and graph.
4. Answer with explicit separation between source facts, graph-derived relations, and LLM inference.
5. Evaluate with a test set of retrieval and reasoning questions.
