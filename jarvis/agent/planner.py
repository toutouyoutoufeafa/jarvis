"""Planner for task decomposition and execution planning."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from jarvis.agent.config import get_agent_config
from jarvis.llm import get_llm_engine, Message
from jarvis.agent.tools import get_tool_registry


@dataclass
class PlanStep:
    id: str
    description: str
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, done, failed
    result: Any = None
    error: str | None = None


@dataclass
class Plan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    current_step: int = 0

    def to_json(self) -> str:
        return json.dumps({
            "goal": self.goal,
            "steps": [
                {
                    "id": s.id,
                    "description": s.description,
                    "tool": s.tool,
                    "args": s.args,
                    "depends_on": s.depends_on,
                    "status": s.status,
                }
                for s in self.steps
            ],
        }, indent=2)


class Planner:
    """Decomposes goals into executable plans using LLM."""

    def __init__(self) -> None:
        self.config = get_agent_config()
        self.llm = get_llm_engine()
        self.tools = get_tool_registry()

    def _build_planning_prompt(self, goal: str, context: str = "") -> str:
        tool_schemas = self.tools.get_openai_schemas()
        tools_desc = json.dumps(tool_schemas, indent=2)

        return f"""You are a planning agent. Decompose the goal into a sequence of steps using available tools.

GOAL: {goal}

CONTEXT:
{context}

AVAILABLE TOOLS:
{tools_desc}

Return a JSON plan with this structure:
{{
  "steps": [
    {{
      "id": "step_1",
      "description": "What this step does",
      "tool": "tool_name",  // or null for reasoning-only step
      "args": {{"arg1": "value"}},  // arguments for the tool
      "depends_on": []  // list of step IDs this depends on
    }}
  ]
}}

Rules:
- Use ONLY tools from the available list
- Each step should be atomic and verifiable
- Include depends_on for ordering
- Max {self.config.planner.max_steps} steps
- If goal needs reasoning without tools, use tool: null
- Prefer parallel steps when possible (no depends_on)

JSON Plan:"""

    async def create_plan(self, goal: str, context: str = "") -> Plan:
        """Create execution plan for a goal."""
        prompt = self._build_planning_prompt(goal, context)

        try:
            response = await self.llm.complete(prompt)
            logger.debug(f"Planner response: {response[:500]}...")

            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in planner response")

            data = json.loads(json_match.group())
            steps = []

            for i, step_data in enumerate(data.get("steps", [])):
                step = PlanStep(
                    id=step_data.get("id", f"step_{i+1}"),
                    description=step_data.get("description", ""),
                    tool=step_data.get("tool"),
                    args=step_data.get("args", {}),
                    depends_on=step_data.get("depends_on", []),
                )
                steps.append(step)

            return Plan(goal=goal, steps=steps)

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Fallback: simple single-step plan
            return Plan(goal=goal, steps=[
                PlanStep(
                    id="step_1",
                    description=goal,
                    tool=None,
                    args={},
                )
            ])

    async def refine_plan(self, plan: Plan, feedback: str) -> Plan:
        """Refine plan based on execution feedback."""
        prompt = f"""The previous plan execution had issues. Refine the plan.

ORIGINAL GOAL: {plan.goal}
CURRENT PLAN:
{plan.to_json()}

FEEDBACK: {feedback}

Return improved JSON plan with same structure."""

        try:
            response = await self.llm.complete(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                steps = []
                for i, step_data in enumerate(data.get("steps", [])):
                    steps.append(PlanStep(
                        id=step_data.get("id", f"step_{i+1}"),
                        description=step_data.get("description", ""),
                        tool=step_data.get("tool"),
                        args=step_data.get("args", {}),
                        depends_on=step_data.get("depends_on", []),
                    ))
                plan.steps = steps
        except Exception as e:
            logger.error(f"Plan refinement failed: {e}")

        return plan


class PlanExecutor:
    """Executes a plan step by step."""

    def __init__(self, planner: Planner | None = None) -> None:
        self.planner = planner or Planner()
        self.tools = get_tool_registry()

    async def execute(self, plan: Plan, max_retries: int | None = None) -> Plan:
        """Execute all steps in plan."""
        max_retries = max_retries or self.config.planner.max_retries
        completed = set()

        while plan.current_step < len(plan.steps):
            step = plan.steps[plan.current_step]

            # Check dependencies
            if not all(dep in completed for dep in step.depends_on):
                # Find next executable step
                found = False
                for i, s in enumerate(plan.steps):
                    if s.status == "pending" and all(d in completed for d in s.depends_on):
                        plan.current_step = i
                        step = s
                        found = True
                        break
                if not found:
                    # All remaining steps blocked
                    break

            step.status = "running"
            logger.info(f"Executing step {step.id}: {step.description}")

            try:
                if step.tool:
                    # Execute tool
                    result = await self.tools.execute(step.tool, **step.args)
                    step.result = result
                    step.status = "done"
                else:
                    # Reasoning step - use LLM
                    result = await self._reasoning_step(step, plan)
                    step.result = result
                    step.status = "done"

                completed.add(step.id)
                plan.current_step += 1

            except Exception as e:
                step.status = "failed"
                step.error = str(e)
                logger.error(f"Step {step.id} failed: {e}")

                # Retry logic
                if max_retries > 0:
                    logger.info(f"Retrying step {step.id} ({max_retries} retries left)")
                    max_retries -= 1
                    step.status = "pending"
                    continue
                else:
                    # Try to refine plan
                    logger.info("Attempting plan refinement...")
                    plan = await self.planner.refine_plan(plan, f"Step {step.id} failed: {e}")
                    plan.current_step = 0
                    completed.clear()
                    for s in plan.steps:
                        s.status = "pending"
                        s.result = None
                        s.error = None
                    max_retries = self.config.planner.max_retries

        return plan

    async def _reasoning_step(self, step: PlanStep, plan: Plan) -> str:
        """Execute a reasoning-only step using LLM."""
        context = f"Goal: {plan.goal}\nCompleted steps: {[s.id for s in plan.steps if s.status == 'done']}"
        prompt = f"""{context}

Current step: {step.description}
Think through this step and provide the result or next actions needed.

Response:"""

        return await self.llm.complete(prompt)


# Global instances
_planner: Planner | None = None
_executor: PlanExecutor | None = None


def get_planner() -> Planner:
    global _planner
    if _planner is None:
        _planner = Planner()
    return _planner


def get_executor() -> PlanExecutor:
    global _executor
    if _executor is None:
        _executor = PlanExecutor(get_planner())
    return _executor