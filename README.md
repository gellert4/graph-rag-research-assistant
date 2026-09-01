# GraphRAG Research Assistant

A small, explainable research assistant built for a controlled Apollo mission knowledge base. The goal is not to fake a massive production system, but to show how a practical LLM-style workflow can be designed: document retrieval, graph reasoning, evidence separation, uncertainty handling, and evaluation.

## What this project demonstrates

- RAG-style document retrieval over a small curated corpus
- structured entity graph with relationships like mission → landing site, crew member → mission, person → Moon
- clear separation between source evidence, graph relations, and model inference
- explicit uncertainty handling when the answer is not supported by the data
- an evaluation set of realistic research questions

## Architecture

1. Corpus layer
   - The project uses a small set of Apollo mission text files under `data/apollo_docs/`.
   - Each file is treated as a source document and split into natural paragraphs.

2. Retrieval layer
   - `Retriever` uses TF-IDF similarity to rank document chunks by question relevance.
   - This is intentionally lightweight and deterministic, which is appropriate for a compact data set.

3. Graph layer
   - `GraphStore` builds a NetworkX graph of entities and relations such as:
     - `Apollo 11` → `landed_at` → `Sea of Tranquility`
     - `Neil Armstrong` → `walked_on` → `Moon`
     - `Apollo 13` → `planned_landing_site` → `Fra Mauro`
   - This allows the system to answer multi-hop and relation-based questions that a single document may not cover fully.

4. Answer layer
   - `ResearchAssistant.answer()` gathers evidence from both sources.
   - The response structure separates:
     - `source_evidence`
     - `graph_relations`
     - `inference`
     - `uncertainty`
     - `contradictions`

## Why this stack

- `scikit-learn` for fast, local TF-IDF retrieval
- `networkx` for lightweight graph modeling
- plain Python for deterministic orchestration

This choice is intentional: for a 10–20 document corpus, the main engineering problem is not raw model size, but good prompt design, retrieval quality, graph grounding, and transparent answer boundaries.

## Model decisions

This project does not use a remote LLM API because the dataset is small and the value is in the system design, not in an expensive model call. In a larger system, I would split responsibilities like this:

- embedding model: dense retrieval for semantic matching
- reranker: improve top-k selection
- graph/entity extractor: identify mission, person, and location names
- generative model: produce summaries and structured answers from retrieved context

For this demo, the logic is intentionally deterministic and explainable: retrieval and graph lookups do the heavy lifting, while the reasoning layer is kept constrained.

## Running the project

```bash
python -m pip install -r requirements.txt
set PYTHONPATH=src
pytest -q
python -m graph_rag_assistant.cli "Which Apollo mission landed in the Sea of Tranquility?"
```

## Evaluation questions

The project includes a compact evaluation set with 10 questions covering:

- simple retrieval
- multi-document reasoning
- graph-based relationship checks
- inference-heavy questions
- a question with insufficient evidence

The evaluation file is in `tests/eval_questions.py` and can be run with:

```bash
python tests/eval_questions.py
```

## Key trade-offs

- RAG is necessary because the question domain is a knowledge base, not a single fact.
- The graph is useful for structured relations and multi-hop reasoning, but not required for every question.
- The system does not blindly trust model output: it always distinguishes concrete source text from inferred interpretation.
- When the evidence is weak or contradictory, the system is allowed to say that the answer cannot be defended.
