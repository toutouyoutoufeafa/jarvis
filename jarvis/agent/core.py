"""Core agent runtime with ReAct loop."""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from loguru import logger

from jarvis.agent.config import get_agent_config, get_workspace_dir
from jarvis.agent.memory import get_memory_manager, MemoryEntry
from jarvis.agent.planner import get_planner, get_executor, Plan, PlanStep
from jarvis.agent.tools import get_tool_registry
from jarvis.llm import get_llm_engine, Message, extract_json
from jarvis.events import Event, Events, get_event_bus


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING = "waiting"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentTask:
    id: str
    goal: str
    context: dict = field(default_factory=dict)
    state: AgentState = AgentState.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    result: Any = None
    error: str | None = None
    plan: Plan | None = None
    iterations: int = 0
    # Anti-hallucination bookkeeping: surfaced back into the think prompt so
    # the model cannot claim a tool call succeeded when it demonstrably failed.
    last_observation: str | None = None
    tool_failures: int = 0
    answer_rejected: int = 0
    # Anti-deadlock: small models happily re-run the same successful call
    # forever instead of concluding.
    last_action_key: str | None = None
    repeats: int = 0
    tools_used: int = 0


# Small models routinely name a required argument differently from the schema.
# Map canonical parameter -> acceptable synonyms found in model output.
_TOOL_ARG_ALIASES: dict[str, tuple[str, ...]] = {
    "content": ("data", "text", "contents", "body", "file_content", "value"),
    "path": ("file", "filename", "filepath", "file_path", "target"),
    "dest_path": ("destination", "output", "out"),
    "dst": ("dest", "destination", "target"),
    "command": ("cmd", "cmdline", "shell", "input"),
    "code": ("source", "script", "python", "program"),
    "query": ("q", "search", "question", "prompt"),
    "url": ("uri", "link", "site"),
    "message": ("msg", "text"),
    "repo": ("repository", "git_url"),
    "branch": ("name", "ref"),
}

_ACTION_TYPES = {"tool", "answer", "plan", "execute", "wait"}

# A tool decision is one short JSON object. Capping generation keeps a small
# local model from rambling: measured ~45s -> ~5s per decision on CPU.
THINK_OPTIONS = {"num_predict": 96, "temperature": 0.1}

# Tools shown with a full signature; the rest are listed by name only. Keeps
# the decision prompt near 400 tokens, which is ~10s of prompt evaluation.
_DETAILED_TOOLS = {
    "fs_read", "fs_write", "fs_list", "fs_mkdir", "fs_exists",
    "fs_copy", "fs_move",
    "shell_exec", "python_exec",
    "web_search", "web_fetch",
}


def _admits_failure(text: str) -> bool:
    """True when the model is honestly reporting that it could not succeed."""
    head = (text or "").strip().lower()[:40]
    return head.startswith(("failed", "erreur", "error", "échec", "echec", "impossible"))


class AgentCore:
    """Main agent runtime implementing ReAct (Reason + Act) loop."""

    def __init__(self) -> None:
        self.config = get_agent_config()
        self.llm = get_llm_engine()
        self.memory = get_memory_manager()
        self.tools = get_tool_registry()
        self.planner = get_planner()
        self.executor = get_executor()
        self.event_bus = get_event_bus()

        self._current_task: AgentTask | None = None
        self._running = False
        self._llm_failures = 0  # Consecutive LLM failures counter

    async def initialize(self) -> None:
        await self.memory.initialize()
        await self.llm.initialize()

        # Register built-in tools
        self._register_builtin_tools()

        # Subscribe to events
        self.event_bus.subscribe(Events.LLM_RESPONSE, self._on_llm_response)

        logger.info("Agent core initialized")

    def _register_builtin_tools(self) -> None:
        """Import and register all built-in tools."""
        from jarvis.agent.tools import fs, shell, python, web, git
        fs.register_fs_tools()
        shell.register_shell_tools()
        python.register_python_tools()
        web.register_web_tools()
        git.register_git_tools()

    async def run_task(self, goal: str, context: dict | None = None) -> AgentTask:
        """Run a task to completion using ReAct loop."""
        task = AgentTask(
            id=str(uuid.uuid4())[:8],
            goal=goal,
            context=context or {},
        )

        self._current_task = task
        self._running = True

        try:
            # Store goal in memory
            self.memory.add_stm(
                f"Task started: {goal}",
                metadata={"task_id": task.id, "type": "task_start"},
            )

            # ReAct loop
            while self._running and task.iterations < self.config.max_iterations:
                task.iterations += 1
                task.updated_at = datetime.now()

                # Check if task is complete
                if await self._is_task_complete(task):
                    task.state = AgentState.DONE
                    break

                # Think: decide next action
                task.state = AgentState.THINKING
                action = self._normalize_action(await self._think(task))

                if action.get("type") == "plan":
                    task.state = AgentState.PLANNING
                    task.plan = await self.planner.create_plan(goal, await self._build_context(task))
                    await self._emit_task_update(task)

                elif action.get("type") == "execute":
                    task.state = AgentState.EXECUTING
                    if task.plan:
                        task.plan = await self.executor.execute(task.plan)
                        await self._emit_task_update(task)

                elif action.get("type") == "tool":
                    # Direct tool use. Failures are fed back as observations so
                    # the agent can correct itself (e.g. wrong tool name) instead
                    # of aborting the whole task.
                    tool_name = str(action.get("tool", ""))
                    tool_args = action.get("args", {})
                    action_key = json.dumps(
                        {"tool": tool_name, "args": tool_args}, sort_keys=True, default=str
                    )

                    # Deadlock breaker: the same successful call repeated means
                    # the model is spinning rather than progressing.
                    if action_key == task.last_action_key and not task.tool_failures:
                        task.repeats += 1
                        if task.repeats >= 2:
                            task.result = (
                                "Objectif deja realise (boucle interrompue). "
                                f"Derniere observation: {task.last_observation}"
                            )
                            task.state = AgentState.DONE
                            break
                        observation = (
                            "ALREADY DONE: this exact call already returned OK. "
                            'If the goal is satisfied reply with {"type":"answer","content":"..."} '
                            "now, otherwise do something different."
                        )
                        task.last_observation = observation
                        self.memory.add_stm(
                            observation, metadata={"task_id": task.id, "type": "nudge"}
                        )
                        continue

                    task.repeats = 0
                    task.last_action_key = action_key
                    task.tools_used += 1

                    try:
                        result = await self.tools.execute(tool_name, **tool_args)
                        observation = f"OK ({tool_name}): {result}"
                        task.tool_failures = 0
                    except Exception as e:
                        if tool_name not in self.tools.list_names():
                            hint = (
                                f"Unknown tool '{tool_name}'. "
                                f"Use ONLY these: {', '.join(sorted(self.tools.list_names()))}"
                            )
                        else:
                            hint = f"Tool '{tool_name}' failed: {e}"
                        observation = f"ERROR: {hint}"
                        task.tool_failures += 1
                        logger.warning(f"Tool call rejected: {hint}")

                    task.last_observation = observation
                    self.memory.add_stm(
                        observation,
                        metadata={"task_id": task.id, "tool": tool_name, "type": "observation"},
                    )

                elif action.get("type") == "answer":
                    content = str(action.get("content") or "")
                    # A conclusion reached without touching a single tool is
                    # almost always a shortcut: the model declares success on
                    # the goal's wording alone. Ask once, then accept.
                    if not task.tools_used and task.answer_rejected < 1:
                        task.answer_rejected += 1
                        refusal = (
                            "REFUSED: you answered without calling any tool, so "
                            f"nothing was actually done for: {task.goal!r}. "
                            "Use a tool to perform the action, then answer. If the "
                            "goal genuinely needs no tool, answer again to say so."
                        )
                        task.last_observation = refusal
                        self.memory.add_stm(
                            refusal, metadata={"task_id": task.id, "type": "refusal"}
                        )
                        logger.warning(f"Rejected tool-free answer: {content[:120]}")
                        continue

                    # Refuse a "success" report issued while tool calls are
                    # still failing; give the model two attempts to actually
                    # perform the action before accepting its conclusion.
                    if (
                        task.tool_failures
                        and task.answer_rejected < 2
                        and not _admits_failure(content)
                    ):
                        task.answer_rejected += 1
                        refusal = (
                            "REFUSED: you reported a result while the last tool "
                            f"call failed. Last observation: {task.last_observation}. "
                            "Retry with a valid tool name/arguments (see TOOLS list "
                            "and WORKSPACE above), or answer with 'FAILED: <reason>'."
                        )
                        task.last_observation = refusal
                        self.memory.add_stm(
                            refusal,
                            metadata={"task_id": task.id, "type": "refusal"},
                        )
                        logger.warning(f"Rejected premature answer: {content[:120]}")
                        continue

                    task.result = content
                    task.state = AgentState.DONE
                    break

                elif action.get("type") == "wait":
                    task.state = AgentState.WAITING
                    await asyncio.sleep(action.get("duration", 1))

                else:
                    logger.warning(f"Unknown action type: {action}")

            if task.state != AgentState.DONE:
                task.state = AgentState.ERROR
                task.error = f"Max iterations ({self.config.max_iterations}) reached"

            # Store final result
            if task.result:
                self.memory.add_stm(
                    f"Task completed: {task.result}",
                    metadata={"task_id": task.id, "type": "task_result"},
                )

            await self._emit_task_update(task)
            return task

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.state = AgentState.ERROR
            task.error = str(e)
            await self._emit_task_update(task)
            return task

        finally:
            self._current_task = None
            self._running = False

    async def _build_context(self, task: AgentTask) -> str:
        """Build context string for LLM."""
        parts = [f"Goal: {task.goal}"]

        if task.context:
            parts.append(f"Context: {json.dumps(task.context)}")

        # Bounded memory window: this is the only part of the prompt that
        # changes between iterations, so keep it small enough to re-evaluate
        # quickly (~38 tok/s on CPU).
        mem_context = await self.memory.get_context(task.goal, max_tokens=250)
        if mem_context:
            parts.append(f"Memory:\n{mem_context}")

        # Add plan status
        if task.plan:
            done = [s.id for s in task.plan.steps if s.status == "done"]
            pending = [s.id for s in task.plan.steps if s.status == "pending"]
            failed = [s.id for s in task.plan.steps if s.status == "failed"]
            parts.append(f"Plan: {len(done)} done, {len(pending)} pending, {len(failed)} failed")

        return "\n\n".join(parts)

    def _normalize_action(self, action: dict) -> dict:
        """Coerce whatever the model emitted into the canonical action shape.

        Small models frequently reply ``{"type":"fs_write","args":{...}}``
        instead of ``{"type":"tool","tool":"fs_write",...}``, or wrap the tool
        in ``execute``. Silently dropping those would stall the loop, so we
        translate them rather than reject them.
        """
        if not isinstance(action, dict):
            return {"type": "wait", "duration": 1, "reasoning": "Non-JSON action"}

        atype = str(action.get("type") or "").strip()
        known = self.tools.list_names()

        # {"type":"fs_write", "args":{...}} -- type IS the tool name
        if atype and atype not in _ACTION_TYPES and atype in known:
            return self._as_tool_action(atype, action)

        if atype in ("tool", "call", "function", "action"):
            tool = action.get("tool") or action.get("name") or action.get("function")
            if isinstance(tool, str) and tool:
                return self._as_tool_action(tool, action)

        if atype == "execute":
            args = action.get("args") if isinstance(action.get("args"), dict) else {}
            if isinstance(action.get("tool"), str):
                return self._as_tool_action(action["tool"], action)
            if isinstance(args.get("tool"), str):
                inner = {k: v for k, v in args.items() if k not in ("tool", "args")}
                inner.update(args.get("args") or {})
                return self._as_tool_action(args["tool"], {"args": inner})
            # Bare payload: infer the tool from the argument name.
            for key, tool_name in (
                ("cmd", "shell_exec"),
                ("command", "shell_exec"),
                ("code", "python_exec"),
                ("script", "python_exec"),
            ):
                if key in args:
                    logger.info(f"execute payload mapped to {tool_name}")
                    return self._as_tool_action(tool_name, {"args": args})
            logger.warning(f"Unusable execute action: {action}")
            return {"type": "wait", "duration": 1, "reasoning": "Unusable execute action"}

        if atype in ("final", "response", "result", "done") and "content" in action:
            return {"type": "answer", "content": action.get("content")}

        if not atype:
            if "tool" in action:
                return self._as_tool_action(str(action["tool"]), action)
            if "content" in action:
                return {"type": "answer", "content": action.get("content")}
            logger.warning(f"Action without type: {action}")
            return {"type": "wait", "duration": 1, "reasoning": "Action without type"}

        if atype not in _ACTION_TYPES:
            logger.warning(f"Unknown action type '{atype}', waiting")
            return {"type": "wait", "duration": 1, "reasoning": f"Unknown type {atype}"}

        if atype == "tool":
            tool = action.get("tool")
            if not isinstance(tool, str) or not tool:
                logger.warning(f"tool action without tool name: {action}")
                return {"type": "wait", "duration": 1, "reasoning": "Missing tool name"}
            return self._as_tool_action(tool, action)

        if atype == "answer" and "content" not in action:
            for key in ("text", "result", "message"):
                if key in action:
                    return {"type": "answer", "content": action.get(key)}

        return action

    def _as_tool_action(self, tool_name: str, action: dict) -> dict:
        raw = action.get("args")
        if not isinstance(raw, dict):
            reserved = {"type", "tool", "name", "function", "args", "reasoning", "content"}
            raw = {k: v for k, v in action.items() if k not in reserved}
        return {"type": "tool", "tool": tool_name, "args": self._coerce_args(tool_name, raw)}

    def _coerce_args(self, tool_name: str, args: dict) -> dict:
        """Rename aliased keys so required parameters are actually supplied."""
        tool = self.tools.get(tool_name)
        if tool is None or not args:
            return args

        coerced = dict(args)
        for param in tool.get_schema().parameters:
            if not param.required or param.name in coerced:
                continue
            for alias in _TOOL_ARG_ALIASES.get(param.name, ()):
                if alias in coerced:
                    coerced[param.name] = coerced.pop(alias)
                    break
        return coerced

    def _agent_system_prompt(self) -> str:
        """Static half of the decision prompt.

        Prompt evaluation costs ~44 tok/s on this CPU-only machine, so token
        count is latency. Only tools whose argument names are not obvious get
        a signature; the remaining ones are listed by name alone.
        """
        workspace = get_workspace_dir()
        catalogue = self.tools.describe_all(full=_DETAILED_TOOLS)
        return f"""Tu es JARVIS, un agent autonome. Reponds UNIQUEMENT par un objet JSON.

WORKSPACE (seul endroit ou creer des fichiers): {workspace}

OUTILS ({len(self.tools.list_names())} disponibles):
{catalogue}

Reponds par EXACTEMENT UN objet JSON:
{{"type":"tool","tool":"fs_write","args":{{"path":"<fichier>","content":"<texte>"}}}}
{{"type":"answer","content":"<resultat>"}}
{{"type":"plan"}}

Regles:
- "type" vaut TOUJOURS "tool" pour un appel d'outil; le nom va dans "tool".
- Les cles de "args" doivent correspondre a la signature de l'outil.
- Chemins relatifs = dans le workspace.
- Ne declare un succes que si la derniere observation commence par OK.
- N'answer qu'apres avoir reellement execute l'action demandee avec un outil.
- Des que l'objectif est atteint, reponds answer au lieu d'appeler d'autres outils.
JSON:"""

    async def _think(self, task: AgentTask) -> dict:
        """Decide next action using LLM."""
        context = await self._build_context(task)

        # Make a failed last step impossible to ignore: small models happily
        # report "done" right after an error unless it is spelled out.
        if task.tool_failures:
            context += (
                f"\n\nDERNIERE ETAPE EN ECHEC ({task.tool_failures} echec(s) consecutif(s)):\n"
                f"{task.last_observation}\n"
                "Ne declare PAS de succes. Corrige le nom de l'outil ou ses arguments, "
                "ou reponds 'FAILED: <raison>'."
            )
        elif task.last_observation:
            context += f"\n\nRESULTAT DE LA DERNIERE ETAPE:\n{task.last_observation}"

        try:
            response = await self.llm.complete(
                context,
                system_prompt=self._agent_system_prompt(),
                options=THINK_OPTIONS,
            )
            self._llm_failures = 0  # Reset on success
            logger.debug(f"Agent think: {response[:300]}...")

            action = extract_json(response)
            if isinstance(action, dict) and action:
                # `type` may be missing; _normalize_action handles that shape.
                return action
            logger.warning(f"No usable action in think response: {response[:200]}")
        except Exception as e:
            self._llm_failures += 1
            logger.error(f"Think failed ({self._llm_failures} consecutive): {e}")

            if self._llm_failures >= 3:
                return {
                    "type": "answer",
                    "content": f"LLM unavailable after {self._llm_failures} attempts. Error: {e}. Please start Ollama or configure an LLM provider.",
                    "reasoning": "LLM is unreachable, aborting"
                }

        # Fallback
        return {"type": "wait", "duration": 1, "reasoning": "Error in thinking, waiting"}

    async def _is_task_complete(self, task: AgentTask) -> bool:
        """Check if task goal is achieved."""
        if task.plan:
            return all(s.status == "done" for s in task.plan.steps)

        # If agent produced a result, trust it (avoids extra LLM call on small models)
        if task.result:
            return True

        return False

    async def _on_llm_response(self, event: Event) -> None:
        pass  # Could hook streaming here

    async def _emit_task_update(self, task: AgentTask) -> None:
        await self.event_bus.emit(Event("agent.task_update", {
            "task_id": task.id,
            "state": task.state.value,
            "iterations": task.iterations,
            "result": task.result,
            "error": task.error,
            "plan": task.plan.to_json() if task.plan else None,
        }))

    def get_status(self) -> dict:
        if not self._current_task:
            return {"state": "idle", "task": None}

        return {
            "state": self._current_task.state.value,
            "task": {
                "id": self._current_task.id,
                "goal": self._current_task.goal,
                "iterations": self._current_task.iterations,
                "plan_steps": len(self._current_task.plan.steps) if self._current_task.plan else 0,
            }
        }

    async def stop(self) -> None:
        self._running = False
        logger.info("Agent core stopped")


# Global instance
_agent_core: AgentCore | None = None


def get_agent_core() -> AgentCore:
    global _agent_core
    if _agent_core is None:
        _agent_core = AgentCore()
    return _agent_core