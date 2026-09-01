from graph_rag_assistant.assistant import ResearchAssistant

QUESTIONS = [
    ("Which Apollo mission landed in the Sea of Tranquility?", ["apollo 11", "sea of tranquility"]),
    ("Who was the first person to walk on the Moon?", ["neil armstrong", "first person", "moon"]),
    ("Which mission was intended to land in Fra Mauro but did not?", ["apollo 13", "fra mauro", "did not"]),
    ("What was the landing site of Apollo 14?", ["apollo 14", "fra mauro"]),
    ("Which mission used a Lunar Roving Vehicle?", ["apollo 15", "lunar roving vehicle"]),
    ("Which crew member remained in orbit during Apollo 17?", ["ronald evans", "apollo 17", "orbit"]),
    ("What is the connection between Neil Armstrong and the Moon?", ["neil armstrong", "walked on", "moon"]),
    ("Which Apollo mission landed in the Descartes Highlands?", ["apollo 16", "descartes highlands"]),
    ("What happened to Apollo 13 after the oxygen tank problem?", ["apollo 13", "aborted", "returned safely"]),
    ("What did Apollo 7 do on the far side of the Moon?", ["insufficient evidence"]),
    ("Where did Apollo 13 land on the Moon?", ["apollo 13", "did not land", "fra mauro"]),
]


def run_eval():
    assistant = ResearchAssistant()
    passed = 0
    print("Apollo GraphRAG evaluation")
    print("-" * 70)

    for idx, (question, expected_terms) in enumerate(QUESTIONS, start=1):
        result = assistant.answer(question)
        answer_text = str(result.get("answer", "")).lower()
        passed_flag = all(term in answer_text for term in expected_terms)
        status = "PASS" if passed_flag else "FAIL"
        if passed_flag:
            passed += 1

        print(f"Q{idx}: {question}")
        print(f"Expected terms: {expected_terms}")
        print(f"Actual: {result.get('answer')}")
        print(f"Status: {status}")
        print(f"Uncertainty: {result.get('uncertainty')}")
        print("---")

    total = len(QUESTIONS)
    print(f"Summary: {passed}/{total} passed")
    return passed, total


def test_evaluation_set_passes():
    passed, total = run_eval()
    assert passed == total


if __name__ == "__main__":
    run_eval()
