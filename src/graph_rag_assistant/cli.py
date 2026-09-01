from __future__ import annotations

import argparse
import json

from graph_rag_assistant.assistant import ResearchAssistant


def main():
    parser = argparse.ArgumentParser(description="Apollo GraphRAG research assistant")
    parser.add_argument("question", nargs="?", default="Which Apollo mission landed in the Sea of Tranquility?", help="Question to ask")
    args = parser.parse_args()

    assistant = ResearchAssistant()
    response = assistant.answer(args.question)
    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
