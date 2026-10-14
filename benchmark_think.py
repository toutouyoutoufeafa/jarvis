"""Measure the chat path end to end, including the RAG recall/store."""
import asyncio
import time

from jarvis import chat


async def main() -> None:
    questions = [
        "Bonjour",
        "Je m'appelle Tony et mon fichier principal s'appelle rapport.py",
        "Comment je m'appelle d'apres toi ?",
        "Et quel est le nom de mon fichier principal ?",
    ]

    snapshot: list[dict] = []
    total = 0.0
    for question in questions:
        snapshot.append({"role": "user", "content": question})
        start = time.perf_counter()
        reply = await chat.respond(question, snapshot=snapshot[:-1], llm_available=True)
        elapsed = time.perf_counter() - start
        total += elapsed
        snapshot.append({"role": "assistant", "content": reply.text})
        print(f"tour {elapsed:5.1f}s  source={reply.source:8s} -> {reply.text[:80]!r}")

    print(f"\nTOTAL {total:.1f}s pour {len(questions)} questions")


if __name__ == "__main__":
    asyncio.run(main())
