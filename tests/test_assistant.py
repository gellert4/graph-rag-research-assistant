from graph_rag_assistant.assistant import ResearchAssistant
from graph_rag_assistant.llm_adapter import LLMAdapter


def test_question_about_moon_landing_works():
    assistant = ResearchAssistant()
    answer = assistant.answer("Which Apollo mission landed in the Sea of Tranquility?")
    assert "apollo 11" in answer["answer"].lower()
    assert answer["source_evidence"]
    assert answer["uncertainty"] is False


def test_missing_data_is_handled_gracefully():
    assistant = ResearchAssistant()
    answer = assistant.answer("What did Apollo 7 do on the far side of the Moon?")
    assert "insufficient evidence" in answer["answer"].lower()
    assert answer["uncertainty"] is True


def test_document_count_and_retrieval_scores_make_sense():
    assistant = ResearchAssistant()
    assert len(assistant.document_store.documents) >= 15
    retrieved = assistant.retriever.retrieve("Apollo 14 landing site", top_k=3)
    assert retrieved
    assert retrieved[0][1] > 0


def test_graph_path_and_entity_types_work():
    assistant = ResearchAssistant()
    graph = assistant.graph_store
    path = graph.path_between("Neil Armstrong", "Sea of Tranquility")
    assert path
    assert graph.graph.nodes["Neil Armstrong"]["entity_type"] == "Person"
    assert graph.graph.nodes["Apollo 11"]["entity_type"] == "Mission"


def test_graph_query_supports_incoming_edges_for_fra_mauro():
    assistant = ResearchAssistant()
    graph_results = assistant.graph_store.query("Which mission was intended to land in Fra Mauro but did not?")
    assert any(result["relation"] == "planned_to_land_at" and result["target"] == "Fra Mauro" for result in graph_results)
    answer = assistant.answer("Which mission was intended to land in Fra Mauro but did not?")
    assert "apollo 13" in answer["answer"].lower()


def test_llm_adapter_exposes_last_error_for_invalid_json(monkeypatch):
    adapter = LLMAdapter(api_key="test-key")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "not-json"}}]}

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("graph_rag_assistant.openai_llm.requests.post", fake_post)
    try:
        adapter.generate_structured_answer("test", [{"source": "x", "text": "y"}], [])
    except ValueError:
        pass
    assert adapter.last_error is not None or adapter.model.last_error is not None
