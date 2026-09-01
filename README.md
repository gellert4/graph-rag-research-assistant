# GraphRAG Research Assistant

This project implements a compact Apollo research assistant that blends retrieval, graph traversal, and grounded answer generation. The goal is to show how a practical LLM-style workflow can be designed in a small but realistic setting: source documents are retrieved, related entities are checked in a graph, and a final answer is produced only when the evidence supports it.

## What the system demonstrates

- document-backed RAG over a curated corpus of Apollo mission materials
- a lightweight NetworkX knowledge graph for mission-to-location and person-to-mission relations
- separation of source evidence, graph relations, and final reasoning
- explicit abstention when the evidence is weak or missing
- a small evaluation suite to check answer quality on realistic questions

## Architecture

1. Corpus layer
   - The program loads Apollo-related text files from `data/apollo_docs/`.
   - Each document is segmented into natural chunks and treated as a verifiable evidence source.

2. Retrieval layer
   - `Retriever` uses TF-IDF similarity over the chunk corpus.
   - This provides fast, deterministic ranking for questions such as mission location, crew member, or surface activity queries.

3. Graph layer
   - `GraphStore` links missions, locations, and people with typed relationships.
   - Examples include:
     - `Apollo 11` → `landed_at` → `Sea of Tranquility`
     - `Neil Armstrong` → `walked_on` → `Moon`
     - `Apollo 13` → `planned_to_land_at` → `Fra Mauro`
   - This supports relation-driven questions and multi-hop reasoning such as person → mission → landing site.

4. Answer generation layer
   - `ResearchAssistant.answer()` gathers evidence from retrieval and graph lookup.
   - It avoids unsupported answers by checking whether the requested entity appears in the evidence or graph.
   - The response payload includes:
     - `answer`
     - `source_evidence`
     - `graph_relations`
     - `inference`
     - `uncertainty`
     - `contradictions`
     - `confidence`

## Why this stack

- `scikit-learn` for local, explainable TF-IDF retrieval
- `networkx` for compact graph reasoning over mission metadata
- `requests` for optional OpenAI-compatible API calls
- Python as the orchestration layer for the end-to-end workflow

This combination is intentionally lightweight and explainable. In a small knowledge base, the most important engineering challenge is not model size, but retrieval quality, grounding, and the discipline to abstain when the evidence is insufficient.

## Real LLM integration

The project is designed to support a real OpenAI-compatible model when an API key is configured. The adapter lives in `src/graph_rag_assistant/openai_llm.py` and is used by `LLMAdapter` when `OPENAI_API_KEY` is present.

Example:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:PYTHONPATH = "src"
python -m graph_rag_assistant.cli "Which Apollo mission landed in the Sea of Tranquility?"
```

If no API key is set, the system remains usable in a grounded fallback mode and explicitly refuses unsupported claims instead of guessing.

## Getting started

```bash
python -m pip install -r requirements.txt
set PYTHONPATH=src
pytest -q
python -m graph_rag_assistant.cli "Which Apollo mission landed in the Sea of Tranquility?"
```

## Evaluation set

The repository includes a compact evaluation suite in `tests/eval_questions.py` with 11 questions covering:

- direct fact retrieval
- crew and mission relations
- landing-site questions
- multi-hop reasoning
- under-supported questions
- a false-premise Apollo 13 case

A representative sample of results is:

- `Which Apollo mission landed in the Sea of Tranquility?` → `Apollo 11`
- `Who was the first person to walk on the Moon?` → `Neil Armstrong`
- `Which mission was intended to land in Fra Mauro but did not?` → `Apollo 13`
- `What is the connection between Neil Armstrong and the Moon?` → `Neil Armstrong walked on the Moon as the first person`
- `What did Apollo 7 do on the far side of the Moon?` → `Insufficient evidence` (expected abstention)

## Results

Verified in the current workspace with:

```powershell
$env:PYTHONPATH = "src"
pytest -q
python tests/eval_questions.py
```

Actual output from the last successful run:

```text
6 passed in 3.43s
Summary: 11/11 passed
```

## Data sources

The corpus is deliberately curated from Apollo mission documentation and NASA-era summary material. For external provenance, the project uses the same mission facts documented in NASA Apollo program and mission summaries, including:

- NASA Apollo program overview: https://www.nasa.gov/mission_pages/apollo/
- NASA Apollo mission pages: https://www.nasa.gov/mission_pages/apollo/missions/
- Apollo 11 mission summary: https://www.nasa.gov/mission_pages/apollo/missions/apollo11.html
- Apollo 13 mission summary: https://www.nasa.gov/mission_pages/apollo/missions/apollo13.html
- Apollo 15 mission summary: https://www.nasa.gov/mission_pages/apollo/missions/apollo15.html

## Known limitations

- The corpus is intentionally curated and compact; it is not a full mission encyclopedia.
- The retrieval layer is lightweight TF-IDF, so it is best suited for controlled domains rather than open-ended web-scale research.
- The answer generator is grounded, but a real LLM call still depends on the availability of an API key and network access.
- The project is a demonstration of LLM system design, not a production-grade scientific knowledge engine.

## Deliverable note

This project is intentionally designed as a practical demonstration of how to reason about LLM-based systems in a small repository: retrieval, graph grounding, uncertainty control, evaluation, and transparent generation. The focus is on the engineering pattern rather than on building an extremely large general-purpose system.
