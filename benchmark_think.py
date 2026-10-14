"""Where does a chat turn spend its time?"""
import asyncio
import time

from jarvis.chat import SYSTEM_PROMPT, build_history
from jarvis.intent import get_intent_recognizer
from jarvis.llm import get_llm_engine


async def main() -> None:
    recognizer = get_intent_recognizer()
    llm = get_llm_engine()
    await llm.initialize()

    questions = [
        "Bonjour",
        "C'est quoi la photosynthese ?",
        "Et pourquoi les feuilles sont-elles vertes ?",
    ]
    snapshot: list[dict] = []

    for question in questions:
        t0 = time.perf_counter()
        await recognizer.recognize(question)
        t_intent = time.perf_counter() - t0

        history = build_history(snapshot[:-1])
        t0 = time.perf_counter()
        response = await llm.complete(
            question,
            system_prompt=SYSTEM_PROMPT,
            history=history,
            options={"num_predict": chat_tokens()},
        )
        t_llm = time.perf_counter() - t0

        prompt_chars = len(SYSTEM_PROMPT) + sum(
            len(m.content) for m in history
        ) + len(question)
        print(
            f"intent={t_intent:5.2f}s  llm={t_llm:6.2f}s  "
            f"prompt={prompt_chars:5d} chars (~{prompt_chars // 3} tokens)  "
            f"reponse={len(response)} chars"
        )
        snapshot.append({"role": "user", "content": question})
        snapshot.append({"role": "assistant", "content": response})


def chat_tokens() -> int:
    from jarvis.chat import CHAT_MAX_TOKENS
    return CHAT_MAX_TOKENS


if __name__ == "__main__":
    asyncio.run(main())
