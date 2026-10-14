"""Tests for the conversational turn: routing and history shaping."""
from __future__ import annotations

import pytest

from jarvis import chat
from jarvis.agent.memory import MemoryEntry
from jarvis.chat import ChatReply, build_history, respond
from jarvis.intent import INTENT_LLM_OPTIONS, Intent, IntentRecognizer, canned_reply


class _FakeLLM:
    """Stands in for LLMEngine and records what the chat asked for."""

    def __init__(self, reply: str = "Bonjour Tony", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[dict] = []
        self.model: str | None = None

    def set_model(self, model: str) -> bool:
        self.model = model
        return True

    def get_model(self) -> str | None:
        return self.model

    async def complete(self, user_input, provider=None, system_prompt=None,
                       stream=False, history=None, options=None) -> str:
        self.calls.append({
            "user_input": user_input,
            "system_prompt": system_prompt,
            "history": list(history or []),
            "options": options,
        })
        if self.fail:
            raise RuntimeError("modele hors ligne")
        return self.reply


class _FakeRecognizer:
    def __init__(self, name: str, confidence: float) -> None:
        self.name = name
        self.confidence = confidence
        self.seen: list[str] = []

    async def recognize(self, text: str) -> Intent:
        self.seen.append(text)
        return Intent(name=self.name, confidence=self.confidence,
                      entities={}, raw_text=text)


class _FakeExecutor:
    def __init__(self) -> None:
        self.intents: list[Intent] = []
        self.plain: list[str] = []

    async def execute_intent(self, intent: Intent) -> str:
        self.intents.append(intent)
        return f"execute:{intent.name}"

    async def process_text(self, text: str) -> str:
        self.plain.append(text)
        return f"fallback:{text}"


class _FakeMemory:
    """Records what the chat stored and replays a canned recall."""

    def __init__(self, results: list | None = None) -> None:
        self.results = results or []
        self.stored: list[tuple[str, str]] = []
        self.initialized = 0

    async def initialize(self) -> None:
        self.initialized += 1

    async def search_ltm(self, query, limit=10, filter_metadata=None):
        return list(self.results[:limit])

    async def remember_exchange(self, user_text: str, reply: str):
        self.stored.append((user_text, reply))


def _wire(monkeypatch, recognizer=None, executor=None, llm=None, memory=None):
    monkeypatch.setattr(chat, "get_intent_recognizer",
                        lambda: recognizer or _FakeRecognizer("chat", 0.1))
    monkeypatch.setattr(chat, "get_action_executor", lambda: executor or _FakeExecutor())
    monkeypatch.setattr(chat, "get_llm_engine", lambda: llm or _FakeLLM())
    monkeypatch.setattr(chat, "get_memory_manager", lambda: memory or _FakeMemory())
    monkeypatch.setattr(chat, "_memory_ready", False)


@pytest.mark.asyncio
class TestRouting:
    async def test_confident_intent_short_circuits_the_llm(self, monkeypatch) -> None:
        recognizer = _FakeRecognizer("open_app", 0.95)
        executor = _FakeExecutor()
        llm = _FakeLLM()
        _wire(monkeypatch, recognizer, executor, llm)

        reply = await respond("ouvre chrome")

        assert reply == ChatReply(text="execute:open_app", source="intent")
        assert llm.calls == []  # no tokens spent on a command we can run

    async def test_a_chat_intent_does_not_hijack_conversation(self, monkeypatch) -> None:
        """'chat' is the recognizer's default, so it must never win outright."""
        _wire(monkeypatch, _FakeRecognizer("chat", 0.99))

        reply = await respond("parle-moi de ton architecture")

        assert reply.source == "llm"

    async def test_greeting_is_answered_before_any_model_call(self, monkeypatch) -> None:
        """A greeting must cost zero LLM round-trips, intent or chat."""
        recognizer = _FakeRecognizer("chat", 0.1)
        llm = _FakeLLM()
        _wire(monkeypatch, recognizer, llm=llm)

        reply = await respond("Bonjour")

        assert reply.source == "intent"
        assert "JARVIS" in reply.text
        assert recognizer.seen == []  # never even classified
        assert llm.calls == []

    async def test_greeting_inside_a_real_question_still_reaches_the_llm(
        self, monkeypatch
    ) -> None:
        _wire(monkeypatch, _FakeRecognizer("chat", 0.1))

        reply = await respond("bonjour, pourquoi le ciel est-il bleu ?")

        assert reply.source == "llm"

    async def test_weak_intent_falls_through_to_the_llm(self, monkeypatch) -> None:
        executor = _FakeExecutor()
        llm = _FakeLLM()
        _wire(monkeypatch, _FakeRecognizer("open_app", 0.3), executor, llm)

        reply = await respond("peut-etre chrome")

        assert reply.source == "llm"
        assert executor.intents == []

    async def test_llm_turn_is_short_capped_and_persona_bound(self, monkeypatch) -> None:
        llm = _FakeLLM()
        _wire(monkeypatch, llm=llm)

        await respond("explique la photosynthese")

        call = llm.calls[0]
        assert call["options"] == {"num_predict": chat.CHAT_MAX_TOKENS}
        assert "JARVIS" in call["system_prompt"]
        assert "francais" in call["system_prompt"]

    async def test_model_override_reaches_the_engine(self, monkeypatch) -> None:
        llm = _FakeLLM()
        _wire(monkeypatch, llm=llm)

        await respond("explique la photosynthese", model="qwen2.5:3b")

        assert llm.model == "qwen2.5:3b"

    async def test_unreachable_model_degrades_instead_of_raising(self, monkeypatch) -> None:
        _wire(monkeypatch, llm=_FakeLLM(fail=True))

        reply = await respond("explique la photosynthese")

        assert reply.source == "fallback"
        assert "hors ligne" in reply.text

    async def test_no_llm_uses_the_text_fallback(self, monkeypatch) -> None:
        executor = _FakeExecutor()
        _wire(monkeypatch, executor=executor)

        reply = await respond("dis-moi un truc", llm_available=False)

        assert reply == ChatReply(text="fallback:dis-moi un truc", source="fallback")
        assert executor.plain == ["dis-moi un truc"]

    async def test_a_greeting_needs_no_llm_at_all(self, monkeypatch) -> None:
        """Even with every model down, 'bonjour' must be answered."""
        _wire(monkeypatch, executor=_FakeExecutor())

        reply = await respond("bonjour", llm_available=False)

        assert reply.source == "intent"
        assert "JARVIS" in reply.text


@pytest.mark.asyncio
class TestMemory:
    """The RAG half of the chat: recall before answering, store after."""

    async def test_recalled_exchanges_are_injected_into_the_prompt(
        self, monkeypatch
    ) -> None:
        llm = _FakeLLM()
        memory = _FakeMemory(results=[
            MemoryEntry(id="1", content="Utilisateur: mon fichier s'appelle rapport.py",
                        metadata={}),
        ])
        _wire(monkeypatch, llm=llm, memory=memory)

        reply = await respond("et mon fichier ?")

        assert reply.source == "llm"
        prompt = llm.calls[0]["user_input"]
        assert "ECHANGES PASSEES PERTINENTS" in prompt
        assert "rapport.py" in prompt

    async def test_empty_memory_leaves_the_prompt_untouched(self, monkeypatch) -> None:
        llm = _FakeLLM()
        _wire(monkeypatch, llm=llm)

        await respond("explique la photosynthese ?")

        assert llm.calls[0]["user_input"] == "explique la photosynthese ?"

    async def test_every_llm_turn_is_persisted(self, monkeypatch) -> None:
        llm = _FakeLLM(reply="Voila")
        memory = _FakeMemory()
        _wire(monkeypatch, llm=llm, memory=memory)

        await respond("quelle heure est-il ?")

        assert memory.stored == [("quelle heure est-il ?", "Voila")]
        assert memory.initialized == 1  # opened once, not once per turn

    async def test_memory_outage_never_breaks_the_chat(self, monkeypatch) -> None:
        class _Broken:
            async def initialize(self) -> None:
                raise RuntimeError("chroma mort")

            async def search_ltm(self, *args, **kwargs):
                raise RuntimeError("chroma mort")

            async def remember_exchange(self, *args, **kwargs):
                raise RuntimeError("chroma mort")

        _wire(monkeypatch, llm=_FakeLLM(reply="reponse quand meme"),
              memory=_Broken())

        reply = await respond("c'est quoi un entrelacement ?")

        assert reply == ChatReply(text="reponse quand meme", source="llm")


class TestCheapRouting:
    """The two functions that decide whether an LLM call is needed at all."""

    def test_greetings_match_exactly(self) -> None:
        assert "JARVIS" in (canned_reply("  Bonjour ! ") or "")

    def test_a_greeting_inside_a_sentence_is_not_canned(self) -> None:
        assert canned_reply("bonjour, pourquoi le ciel est-il bleu ?") is None

    def test_thanks_are_canned_even_with_extras(self) -> None:
        assert canned_reply("merci beaucoup") is not None

    def test_small_talk_is_canned_but_follow_ups_are_not(self) -> None:
        assert canned_reply("ça va ?") is not None
        # A follow-up on the same topic is still conversation for the model.
        assert canned_reply("ça va bien, et toi ?") is None

    def test_plain_prose_is_left_to_the_model(self) -> None:
        assert canned_reply("explique la photosynthese") is None

    @pytest.mark.parametrize("text", [
        "Pourquoi le ciel est-il bleu ?",
        "comment ça marche",
        "Quel est le meilleur langage",
        "peux-tu m'aider",
    ])
    def test_question_detector(self, text: str) -> None:
        assert IntentRecognizer._looks_like_question(text) is True

    @pytest.mark.parametrize("text", [
        "ouvre chrome",
        "lance le serveur",
        "installe le paquet",
    ])
    def test_commands_are_not_questions(self, text: str) -> None:
        assert IntentRecognizer._looks_like_question(text) is False


class TestIntentRecognitionOrdering:
    """Recognition must not spend a model call when a rule or a question
    already settles it -- that round-trip alone measured 14-17s."""

    @pytest.fixture()
    def recognizer(self):
        return IntentRecognizer()

    @pytest.mark.asyncio
    async def test_a_rule_matches_without_calling_the_llm(self, recognizer) -> None:
        recorder = _FakeLLM()
        recognizer.llm = recorder

        intent = await recognizer.recognize("ouvre chrome")

        assert intent.name == "launch_app"
        assert intent.entities.get("app") == "chrome"
        assert recorder.calls == []

    @pytest.mark.asyncio
    async def test_a_question_skips_the_llm(self, recognizer) -> None:
        recorder = _FakeLLM()
        recognizer.llm = recorder

        intent = await recognizer.recognize("pourquoi le ciel est-il bleu ?")

        assert intent.name == "chat"
        assert recorder.calls == []

    @pytest.mark.asyncio
    async def test_ambiguous_text_still_asks_the_llm(self, recognizer) -> None:
        recorder = _FakeLLM(reply='{"intent":"system_info","confidence":0.8}')
        recognizer.llm = recorder

        intent = await recognizer.recognize("il fait une chaleur de canard")

        assert intent.name == "system_info"
        assert recorder.calls[0]["options"] == INTENT_LLM_OPTIONS

    @pytest.mark.asyncio
    async def test_unparsable_llm_answer_degrades_to_chat(self, recognizer) -> None:
        recorder = _FakeLLM(reply="je ne sais pas vraiment")
        recognizer.llm = recorder

        intent = await recognizer.recognize("dis-moi quelque chose")

        assert intent.name == "chat"

    def test_keeps_recent_turns_verbatim_and_trims_older_ones(self) -> None:
        snapshot = [
            {"role": "user", "content": "question ancienne " + "x" * 300},
            {"role": "assistant", "content": "reponse ancienne " + "y" * 300},
            {"role": "user", "content": "question recente"},
            {"role": "assistant", "content": "reponse recente"},
        ]

        history = build_history(snapshot, old_chars=40, fresh=2)

        assert len(history[0].content) == 40          # trimmed
        assert history[2].content == "question recente"  # untouched
        assert history[3].content == "reponse recente"

    def test_caps_the_number_of_turns(self) -> None:
        snapshot = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"t{i}"}
            for i in range(40)
        ]

        history = build_history(snapshot, keep=3, fresh=3)

        assert len(history) == 6  # keep=3 exchanges -> 6 messages
        assert history[-1].content == "t39"

    def test_ignores_unrelated_roles_and_the_current_message(self) -> None:
        snapshot = [
            {"role": "system", "content": "ignore moi"},
            {"role": "user", "content": "question"},
            {"role": "tool", "content": "sortie d'outil"},
        ]

        history = build_history(snapshot)

        assert [m.content for m in history] == ["question"]
        assert all(m.role in ("user", "assistant") for m in history)

    def test_empty_snapshot_is_safe(self) -> None:
        assert build_history([]) == []
