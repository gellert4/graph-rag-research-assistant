from graph_rag_assistant.assistant import ResearchAssistant

QUESTIONS = [
    ("Which Apollo mission landed in the Sea of Tranquility?", "Apollo 11"),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong"),
    ("Which mission was intended to land in Fra Mauro but did not?", "Apollo 13"),
    ("What was the landing site of Apollo 14?", "Fra Mauro"),
    ("Which mission used a Lunar Roving Vehicle?", "Apollo 15"),
    ("Which crew member remained in orbit during Apollo 17?", "Ronald Evans"),
    ("What is the connection between Neil Armstrong and the Moon?", "He walked on it"),
    ("Which Apollo mission landed in the Descartes Highlands?", "Apollo 16"),
    ("What happened to Apollo 13 after the oxygen tank problem?", "The landing was aborted"),
    ("What did Apollo 7 do on the far side of the Moon?", "Insufficient evidence"),
]


def run_eval():
    assistant = ResearchAssistant()
    passed = 0
    print("Apollo GraphRAG evaluation")
    print("-" * 70)

    for idx, (question, expected_answer) in enumerate(QUESTIONS, start=1):
        result = assistant.answer(question)
        answer_text = str(result.get("answer", "")).lower()
        expected = expected_answer.lower()
        passed_flag = expected in answer_text or answer_text.startswith(expected)
        status = "PASS" if passed_flag else "FAIL"
        if passed_flag:
            passed += 1

        print(f"Q{idx}: {question}")
        print(f"Expected: {expected_answer}")
        print(f"Actual: {result.get('answer')}")
        print(f"Status: {status}")
        print(f"Uncertainty: {result.get('uncertainty')}")
        print("---")

    total = len(QUESTIONS)
    print(f"Summary: {passed}/{total} passed")
    return passed, total


if __name__ == "__main__":
    run_eval()
