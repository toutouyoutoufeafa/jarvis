"""One conversational turn for JARVIS.

Kept out of the UI so the CLI, the voice loop and the tests all share the
same routing, and so a turn can be benchmarked without Streamlit.

Routing, cheapest path first:

1. ``intent``   -- a recognised command (open an app, system control, ...).
                   Deterministic and immediate.
2. ``llm``      -- direct chat with conversation history.
3. ``fallback`` -- the intent executor's text handler, used when no LLM
                   provider is reachable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from loguru import logger

from jarvis.agent.memory import get_memory_manager
from jarvis.config import get_config
from jarvis.intent import (
    canned_reply,
    get_action_executor,
    get_intent_recognizer,
)
from jarvis.llm import Message, get_llm_engine
from jarvis.plugins import get_plugin_manager

# Number of past exchanges (user+assistant pairs) sent to the LLM.
# Prompt evaluation runs at ~37 tok/s on this CPU-only box, so every
# retained turn is paid for on every message: 10 turns measured ~1000
# extra tokens (~27s) per question.
MAX_HISTORY_TURNS = 6

# Oldest turns are trimmed to this many characters rather than dropped.
OLD_TURN_CHARS = 160

# Number of most recent turns kept verbatim.
FRESH_TURNS = 4

# Generation cap for chat. The provider default (1024) is a worst case of
# ~100s at ~10 tok/s; 300 keeps a full answer near 30s and typical ones
# well under.
CHAT_MAX_TOKENS = 300

# Confidence at or above which a recognised intent is executed directly.
INTENT_CONFIDENCE = 0.7

# How much past context a question may pull in. Each entry costs prompt
# evaluation (~37 tok/s on this box), so this stays deliberately small.
MEMORY_ENTRIES = 3
MEMORY_CHARS = 200

SYSTEM_PROMPT = """Tu es JARVIS, l'assistant IA personnel de Tony (inspire d'Iron Man).
Tu reponds TOUJOURS en francais, de facon concise et utile.

REGLES:
- Sois direct: 1 a 4 phrases sauf demande de detail.
- Tu as acces a des outils (fichiers, shell, web, git). Si une action est
  demandee, fais-la ou donne exactement la commande a executer.
- Tu te souviens des echanges precedents: appelle l'utilisateur Tony.
- Si tu ne sais pas, dis-le franchement au lieu d'inventer.
- Texte simple dans le chat, pas de markdown lourd.

EXEMPLE:
Utilisateur: Bonjour
JARVIS: Bonjour Tony. Je suis en ligne et pret a travailler."""


@dataclass(frozen=True)
class ChatReply:
    """A turn's answer plus which path produced it (useful for the UI/tests)."""

    text: str
    source: str  # "intent" | "llm" | "fallback"


def build_history(
    snapshot: Sequence[dict],
    keep: int = MAX_HISTORY_TURNS,
    old_chars: int = OLD_TURN_CHARS,
    fresh: int = FRESH_TURNS,
) -> list[Message]:
    """Turn a list of ``{"role", "content"}`` dicts into LLM history.

    The current user message is expected to already be excluded. Only the
    last ``fresh`` turns survive verbatim; older ones are trimmed, which
    keeps continuity at a fraction of the prompt cost.
    """
    turns = [m for m in snapshot if m.get("role") in ("user", "assistant")]
    turns = turns[-keep * 2:]

    history: list[Message] = []
    for i, turn in enumerate(turns):
        content = str(turn.get("content") or "")
        if i < len(turns) - fresh:
            content = content[:old_chars]
        history.append(Message(role=turn["role"], content=content))
    return history


async def respond(
    text: str,
    snapshot: Sequence[dict] = (),
    llm_available: bool = True,
    model: str | None = None,
) -> ChatReply:
    """Answer one user message.

    ``snapshot`` is the message history *including* the current user
    message; it is copied from session state on the caller's thread before
    this coroutine hops to a worker, so nothing here may touch the UI.
    """
    # Social formulas first: a "Bonjour" answered in 20s reads as broken,
    # and routing it through intent classification would cost a round-trip
    # before anyone reaches the model.
    instant = canned_reply(text)
    if instant:
        return ChatReply(text=instant, source="intent")

    # Extensions get a chance before the model: the weather plugin answers
    # a city forecast in ~1s, where the LLM would invent one in ~20s.
    await _ensure_plugins()
    intent = await get_intent_recognizer().recognize(text)
    if intent.confidence >= INTENT_CONFIDENCE and intent.name != "chat":
        return ChatReply(
            text=await get_action_executor().execute_intent(intent),
            source="plugin" if intent.answer else "intent",
        )

    if llm_available:
        llm = get_llm_engine()
        if model:
            llm.set_model(model)
        try:
            # Pull in whatever past exchanges are relevant. This is what
            # lets JARVIS answer "et mon fichier ?" three sessions later;
            # an empty memory just yields the bare question.
            remembered = await recall(text)
            prompt = text
            if remembered:
                prompt = (
                    f"ECHANGES PASSEES PERTINENTS:\n{remembered}\n\nDemande: {text}"
                )

            response = await llm.complete(
                prompt,
                system_prompt=SYSTEM_PROMPT,
                history=build_history(snapshot),
                options={"num_predict": CHAT_MAX_TOKENS},
            )
            await remember(text, str(response))
            return ChatReply(text=response, source="llm")
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the chat
            return ChatReply(
                text=f"Le modele local ne repond pas: {exc}", source="fallback"
            )

    return ChatReply(
        text=await get_action_executor().process_text(text), source="fallback"
    )


def sources() -> Iterable[str]:
    """The routing paths, for diagnostics."""
    return ("intent", "plugin", "llm", "fallback")


_memory_ready = False
_plugins_ready = False


async def _ensure_plugins() -> None:
    """Load extensions once, so they can answer intents (weather, ...).

    The plugin manager is a singleton with its own guard, so calling this
    on every turn is cheap after the first.
    """
    global _plugins_ready
    if _plugins_ready:
        return
    _plugins_ready = True
    try:
        manager = get_plugin_manager()
        manager.load_all(get_config().plugins.directories)
        await manager.initialize_all()
        logger.debug(f"Plugins ready: {[p.name for p in manager.list_plugins()]}")
    except Exception as exc:  # noqa: BLE001 - extensions are optional
        logger.warning(f"Plugin load failed: {exc}")


async def _memory():
    """Lazily initialised shared memory (ChromaDB + embedding model)."""
    global _memory_ready
    memory = get_memory_manager()
    if not _memory_ready:
        await memory.initialize()
        _memory_ready = True
    return memory


async def recall(text: str) -> str:
    """Past exchanges relevant to ``text``, or '' if memory adds nothing."""
    try:
        entries = await (await _memory()).search_ltm(text, limit=MEMORY_ENTRIES)
    except Exception as exc:  # noqa: BLE001 - memory must never break a chat
        logger.debug(f"Memory recall failed: {exc}")
        return ""
    return "\n".join(e.content[:MEMORY_CHARS] for e in entries if e.content)


async def remember(text: str, reply: str) -> None:
    """Persist one exchange to long-term memory (survives restarts)."""
    try:
        await (await _memory()).remember_exchange(text, reply)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Memory store failed: {exc}")
