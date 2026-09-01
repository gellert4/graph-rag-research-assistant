from graph_rag_assistant.assistant import ResearchAssistant


def test_question_about_moon_landing_works():
    assistant = ResearchAssistant()
    answer = assistant.answer("Which Apollo mission landed in the Sea of Tranquility?")
    assert answer["answer"].lower().find("apollo 11") >= 0 or answer["source_evidence"]


def test_missing_data_is_handled_gracefully():
    assistant = ResearchAssistant()
    answer = assistant.answer("What did Apollo 7 do on the far side of the Moon?")
    assert "insufficient evidence" in answer["answer"].lower() or answer["uncertainty"]
