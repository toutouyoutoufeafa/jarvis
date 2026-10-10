"""Reflector Agent - Meta-reasoning, self-critique, improvement."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from jarvis.agent.config import get_agent_config
from jarvis.agent.core import get_agent_core
from jarvis.agent.memory import get_memory_manager
from jarvis.llm import get_llm_engine


@dataclass
class Reflection:
    id: str
    topic: str
    depth: int
    insights: list[str]
    critique: str
    improvements: list[str]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class ReflectorAgent:
    """Agent for deep reflection, self-critique, and meta-reasoning."""

    def __init__(self) -> None:
        self.config = get_agent_config()
        self.llm = get_llm_engine()
        self.memory = get_memory_manager()
        self.core = get_agent_core()

    async def initialize(self) -> None:
        await self.core.initialize()
        logger.info("Reflector agent initialized")

    async def reflect(
        self,
        topic: str,
        context: str = "",
        depth: int | None = None,
    ) -> Reflection:
        """Perform deep reflection on a topic."""
        depth = depth or self.config.reflector.reflection_depth

        # Multi-pass reflection
        insights = []
        current_context = context

        for level in range(depth):
            prompt = self._build_reflection_prompt(topic, current_context, level + 1, depth)
            response = await self.llm.complete(prompt)

            # Parse insights
            level_insights = self._parse_insights(response)
            insights.extend(level_insights)

            # Build context for next level
            current_context += f"\n\nLevel {level+1} insights:\n" + "\n".join(f"- {i}" for i in level_insights)

        # Final synthesis
        critique = await self._synthesize_critique(topic, insights, context)
        improvements = await self._generate_improvements(topic, insights, critique, context)

        reflection = Reflection(
            id=f"refl_{int(datetime.now().timestamp() * 1000)}",
            topic=topic,
            depth=depth,
            insights=insights,
            critique=critique,
            improvements=improvements,
            confidence=self._calculate_confidence(insights),
        )

        # Store in memory
        self.memory.add_stm(
            f"Reflection on '{topic}': {critique[:200]}...",
            metadata={"type": "reflection", "topic": topic, "depth": depth},
        )

        return reflection

    def _build_reflection_prompt(self, topic: str, context: str, level: int, max_depth: int) -> str:
        focus = {
            1: "What are the key aspects, assumptions, and components?",
            2: "What are potential flaws, edge cases, and failure modes?",
            3: "What alternative approaches exist? Trade-offs?",
            4: "What are second-order effects and long-term implications?",
            5: "How would an expert in this domain critique this?",
        }.get(level, "Deep analysis and synthesis.")

        return f"""Reflect deeply on: {topic}

Context: {context}

Reflection Level {level}/{max_depth}: {focus}

Provide 3-5 specific, non-obvious insights.
Be critical, thorough, and precise.
Format as bullet points."""

    def _parse_insights(self, response: str) -> list[str]:
        """Extract insights from response."""
        insights = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith(("- ", "* ", "• ", "1.", "2.", "3.")):
                insight = line.lstrip("- *•123456789. ").strip()
                if len(insight) > 20:  # Filter trivial
                    insights.append(insight)
        return insights[:5]

    async def _synthesize_critique(self, topic: str, insights: list[str], context: str) -> str:
        """Synthesize insights into coherent critique."""
        prompt = f"""Synthesize these insights into a coherent critique of: {topic}

Context: {context}

Insights:
{chr(10).join(f'- {i}' for i in insights)}

Write a structured critique covering:
1. Strengths
2. Weaknesses / Gaps
3. Risks / Blind spots
4. Overall assessment

Be specific and actionable."""

        return await self.llm.complete(prompt)

    async def _generate_improvements(
        self,
        topic: str,
        insights: list[str],
        critique: str,
        context: str,
    ) -> list[str]:
        """Generate concrete improvement suggestions."""
        prompt = f"""Based on this critique, generate 3-5 concrete, actionable improvements for: {topic}

Critique:
{critique}

Context: {context}

Format as numbered list with specific steps."""

        response = await self.llm.complete(prompt)
        improvements = []

        for line in response.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith(("- ", "* "))):
                imp = line.lstrip("0123456789.-* ").strip()
                if len(imp) > 10:
                    improvements.append(imp)

        return improvements[:5]

    def _calculate_confidence(self, insights: list[str]) -> float:
        """Calculate confidence based on insight quality."""
        if not insights:
            return 0.0
        # Simple heuristic: more insights + longer insights = higher confidence
        avg_len = sum(len(i) for i in insights) / len(insights)
        return min(0.9, 0.3 + len(insights) * 0.1 + avg_len / 500)

    async def critique_plan(self, plan_json: str, goal: str) -> dict:
        """Critique an execution plan."""
        prompt = f"""Critique this execution plan for goal: {goal}

Plan:
{plan_json}

Analyze:
1. Completeness - does it achieve the goal?
2. Correctness - are tools used properly?
3. Efficiency - any redundant steps?
4. Risk - what could go wrong?
5. Missing dependencies or error handling?

Return JSON:
{{
  "score": 0-100,
  "issues": ["..."],
  "suggestions": ["..."],
  "missing_steps": ["..."]
}}"""

        response = await self.llm.complete(prompt)
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        return {"score": 50, "issues": ["Parse failed"], "suggestions": [], "missing_steps": []}

    async def post_mortem(self, task_goal: str, task_result: str, task_error: str | None = None) -> Reflection:
        """Perform post-mortem analysis on completed task."""
        context = f"Goal: {task_goal}\nResult: {task_result}"
        if task_error:
            context += f"\nError: {task_error}"

        return await self.reflect(
            topic=f"Post-mortem: {task_goal}",
            context=context,
            depth=3,
        )

    async def learn_from_experience(self, experience: str) -> list[str]:
        """Extract learnings from experience."""
        prompt = f"""Extract key learnings from this experience:

{experience}

Provide 3-5 generalizable principles or patterns.
Format as bullet points."""

        response = await self.llm.complete(prompt)
        learnings = []

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith(("- ", "* ", "• ")):
                learning = line[2:].strip()
                if len(learning) > 15:
                    learnings.append(learning)

        # Store learnings
        for learning in learnings:
            self.memory.add_stm(
                f"Learning: {learning}",
                metadata={"type": "learning", "source": "reflector"},
            )

        return learnings


# Global instance
_reflector: ReflectorAgent | None = None


def get_reflector_agent() -> ReflectorAgent:
    global _reflector
    if _reflector is None:
        _reflector = ReflectorAgent()
    return _reflector