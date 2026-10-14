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

from jarvis.intent import (
    canned_reply,
    get_action_executor,
    get_intent_recognizer,
)
from jarvis.llm import Message, get_llm_engine

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

    intent = await get_intent_recognizer().recognize(text)
    if intent.confidence >= INTENT_CONFIDENCE and intent.name != "chat":
        return ChatReply(
            text=await get_action_executor().execute_intent(intent),
            source="intent",
        )

    if llm_available:
        llm = get_llm_engine()
        if model:
            llm.set_model(model)
        try:
            response = await llm.complete(
                text,
                system_prompt=SYSTEM_PROMPT,
                history=build_history(snapshot),
                options={"num_predict": CHAT_MAX_TOKENS},
            )
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
    return ("intent", "llm", "fallback")
