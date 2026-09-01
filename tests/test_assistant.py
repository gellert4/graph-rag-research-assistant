from graph_rag_assistant.assistant import ResearchAssistant


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
