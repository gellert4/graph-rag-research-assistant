from graph_rag_assistant.assistant import ResearchAssistant

QUESTIONS = [
    "Which Apollo mission landed in the Sea of Tranquility?",
    "Who was the first person to walk on the Moon?",
    "Which mission was intended to land in Fra Mauro but did not?",
    "What was the landing site of Apollo 14?",
    "Which mission used a Lunar Roving Vehicle?",
    "Which crew member remained in orbit during Apollo 17?",
    "What is the connection between Neil Armstrong and the Moon?",
    "Which Apollo mission landed in the Descartes Highlands?",
    "What happened to Apollo 13 after the oxygen tank problem?",
    "Which mission is the last to have landed on the Moon?",
]


def run_eval():
    assistant = ResearchAssistant()
    for idx, question in enumerate(QUESTIONS, start=1):
        result = assistant.answer(question)
        status = "uncertain" if result.get("uncertainty") else "confident"
        print(f"Q{idx}: {question}")
        print(f"Status: {status}")
        print(f"Answer: {result['answer']}")
        print("---")


if __name__ == "__main__":
    run_eval()
