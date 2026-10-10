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

from jarvis.agent.config import get_agent_config
from jarvis.agent.memory import get_memory_manager, MemoryEntry
from jarvis.agent.planner import get_planner, get_executor, Plan, PlanStep
from jarvis.agent.tools import get_tool_registry
from jarvis.llm import get_llm_engine, Message
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
                action = await self._think(task)

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
                    # Direct tool use
                    result = await self.tools.execute(action["tool"], **action.get("args", {}))
                    self.memory.add_stm(
                        f"Tool {action['tool']} result: {result}",
                        metadata={"task_id": task.id, "tool": action["tool"]},
                    )

                elif action.get("type") == "answer":
                    task.result = action.get("content")
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

        # Add memory context
        mem_context = await self.memory.get_context(task.goal)
        if mem_context:
            parts.append(f"Memory:\n{mem_context}")

        # Add STM summary
        parts.append(self.memory.get_stm_summary())

        # Add plan status
        if task.plan:
            done = [s.id for s in task.plan.steps if s.status == "done"]
            pending = [s.id for s in task.plan.steps if s.status == "pending"]
            failed = [s.id for s in task.plan.steps if s.status == "failed"]
            parts.append(f"Plan: {len(done)} done, {len(pending)} pending, {len(failed)} failed")

        return "\n\n".join(parts)

    async def _think(self, task: AgentTask) -> dict:
        """Decide next action using LLM."""
        context = await self._build_context(task)

        tool_schemas = self.tools.get_openai_schemas()
        tools_desc = json.dumps(tool_schemas, indent=2)

        prompt = f"""You are JARVIS, an autonomous agent. Decide the next action to achieve the goal.

{context}

AVAILABLE TOOLS:
{tools_desc}

ACTION TYPES:
- "plan": Create a new multi-step plan (use when goal is complex)
- "execute": Execute the current plan
- "tool": Call a single tool directly (use for simple actions)
- "answer": Task complete, return final result
- "wait": Wait before next iteration

Return ONLY valid JSON:
{{
  "type": "plan|execute|tool|answer|wait",
  "tool": "tool_name",  // required if type=tool
  "args": {{}},         // required if type=tool
  "content": "...",     // required if type=answer
  "duration": 1,        // required if type=wait
  "reasoning": "why this action"
}}

JSON:"""

        try:
            response = await self.llm.complete(prompt)
            self._llm_failures = 0  # Reset on success
            logger.debug(f"Agent think: {response[:300]}...")

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
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

        if task.result:
            # Verify with LLM
            prompt = f"""Goal: {task.goal}
Result: {task.result}

Is the goal fully achieved? Answer only YES or NO."""
            response = await self.llm.complete(prompt)
            return "YES" in response.upper()

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