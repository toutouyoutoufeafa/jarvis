"""Sub-Agent Spawner - Dynamic agent creation and delegation."""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from jarvis.agent.config import get_agent_config
from jarvis.agent.core import AgentCore, AgentTask, AgentState, get_agent_core
from jarvis.agent.designer import DesignerAgent, DesignTask, get_designer_agent
from jarvis.agent.monitor import MonitorAgent, get_monitor_agent
from jarvis.agent.reflector import ReflectorAgent, get_reflector_agent
from jarvis.agent.memory import get_memory_manager
from jarvis.llm import get_llm_engine
from jarvis.events import Event, get_event_bus


class AgentType(Enum):
    GENERIC = "generic"
    DESIGNER = "designer"
    MONITOR = "monitor"
    REFLECTOR = "reflector"
    CUSTOM = "custom"


@dataclass
class SubAgentSpec:
    name: str
    agent_type: AgentType = AgentType.GENERIC
    goal: str = ""
    instructions: str = ""
    tools: list[str] = field(default_factory=list)  # Empty = all tools
    timeout: int = 300
    token_budget: int = 50000
    context: dict = field(default_factory=dict)
    parent_id: str | None = None


@dataclass
class SubAgent:
    id: str
    spec: SubAgentSpec
    core: AgentCore
    state: AgentState = AgentState.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    result: Any = None
    error: str | None = None
    tokens_used: int = 0


class AgentSpawner:
    """Manages creation, execution, and supervision of sub-agents."""

    def __init__(self) -> None:
        self.config = get_agent_config()
        self.llm = get_llm_engine()
        self.memory = get_memory_manager()
        self.event_bus = get_event_bus()

        self._agents: dict[str, SubAgent] = {}
        self._running_agents: dict[str, asyncio.Task] = {}

    async def initialize(self) -> None:
        logger.info("Agent spawner initialized")

    async def spawn(self, spec: SubAgentSpec) -> SubAgent:
        """Create and start a sub-agent."""
        if len(self._running_agents) >= self.config.spawner.max_concurrent_agents:
            raise RuntimeError(f"Max concurrent agents ({self.config.spawner.max_concurrent_agents}) reached")

        agent_id = spec.name or f"agent_{uuid.uuid4().hex[:8]}"
        if agent_id in self._agents:
            agent_id = f"{agent_id}_{uuid.uuid4().hex[:4]}"

        # Create appropriate core
        if spec.agent_type == AgentType.DESIGNER:
            designer = get_designer_agent()
            await designer.initialize()
            core = designer.core
        elif spec.agent_type == AgentType.MONITOR:
            monitor = get_monitor_agent()
            await monitor.initialize()
            core = monitor.core
        elif spec.agent_type == AgentType.REFLECTOR:
            reflector = get_reflector_agent()
            await reflector.initialize()
            core = reflector.core
        else:
            core = AgentCore()
            await core.initialize()

        # Filter tools if specified
        if spec.tools:
            # Create filtered tool registry
            pass  # Tools are global for now

        sub_agent = SubAgent(
            id=agent_id,
            spec=spec,
            core=core,
        )

        self._agents[agent_id] = sub_agent

        # Start execution
        task = asyncio.create_task(self._run_agent(sub_agent))
        self._running_agents[agent_id] = task

        # Track in memory
        self.memory.add_stm(
            f"Spawned sub-agent: {agent_id} ({spec.agent_type.value}) - {spec.goal[:100]}",
            metadata={"type": "spawn", "agent_id": agent_id, "parent_id": spec.parent_id},
        )

        logger.info(f"Spawned agent {agent_id} of type {spec.agent_type.value}")
        return sub_agent

    async def _run_agent(self, agent: SubAgent) -> None:
        """Run sub-agent with supervision."""
        agent.state = AgentState.EXECUTING

        try:
            # Build task for the agent
            task = AgentTask(
                id=agent.id,
                goal=agent.spec.goal,
                context={
                    **agent.spec.context,
                    "agent_type": agent.spec.agent_type.value,
                    "instructions": agent.spec.instructions,
                    "parent_id": agent.spec.parent_id,
                },
            )

            # Execute with timeout
            result = await asyncio.wait_for(
                agent.core.run_task(task),
                timeout=agent.spec.timeout,
            )

            agent.result = result.result
            agent.error = result.error
            agent.state = AgentState.DONE if not result.error else AgentState.ERROR

        except asyncio.TimeoutError:
            agent.state = AgentState.ERROR
            agent.error = f"Timeout after {agent.spec.timeout}s"
            logger.warning(f"Agent {agent.id} timed out")

        except Exception as e:
            agent.state = AgentState.ERROR
            agent.error = str(e)
            logger.error(f"Agent {agent.id} failed: {e}")

        finally:
            # Cleanup
            self._running_agents.pop(agent.id, None)

            # Store result in memory
            self.memory.add_stm(
                f"Sub-agent {agent.id} completed: {agent.state.value}",
                metadata={
                    "type": "agent_result",
                    "agent_id": agent.id,
                    "state": agent.state.value,
                    "error": agent.error,
                },
            )

            # Emit completion event
            await self.event_bus.emit(Event("spawner.agent_done", {
                "agent_id": agent.id,
                "state": agent.state.value,
                "result": agent.result,
                "error": agent.error,
            }))

    async def spawn_and_wait(self, spec: SubAgentSpec) -> SubAgent:
        """Spawn agent and wait for completion."""
        agent = await self.spawn(spec)

        # Wait for completion
        while agent.id in self._running_agents:
            await asyncio.sleep(0.5)

        return agent

    async def spawn_pipeline(self, specs: list[SubAgentSpec]) -> list[SubAgent]:
        """Spawn multiple agents in sequence, passing results."""
        results = []
        previous_result = None

        for spec in specs:
            if previous_result:
                spec.context["previous_result"] = previous_result

            agent = await self.spawn_and_wait(spec)
            results.append(agent)
            previous_result = agent.result

        return results

    async def spawn_parallel(self, specs: list[SubAgentSpec]) -> list[SubAgent]:
        """Spawn multiple agents in parallel."""
        agents = await asyncio.gather(*[self.spawn(spec) for spec in specs])

        # Wait for all
        while any(a.id in self._running_agents for a in agents):
            await asyncio.sleep(0.5)

        return agents

    def get_agent(self, agent_id: str) -> SubAgent | None:
        return self._agents.get(agent_id)

    def list_agents(self) -> list[dict]:
        return [
            {
                "id": a.id,
                "type": a.spec.agent_type.value,
                "goal": a.spec.goal[:100],
                "state": a.state.value,
                "created": a.created_at.isoformat(),
                "tokens_used": a.tokens_used,
            }
            for a in self._agents.values()
        ]

    async def terminate(self, agent_id: str) -> bool:
        """Force terminate a running agent."""
        task = self._running_agents.get(agent_id)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            agent = self._agents.get(agent_id)
            if agent:
                agent.state = AgentState.ERROR
                agent.error = "Terminated by supervisor"

            return True
        return False

    async def terminate_all(self) -> None:
        """Terminate all running agents."""
        for agent_id in list(self._running_agents.keys()):
            await self.terminate(agent_id)

    async def create_specialist(self, domain: str, goal: str) -> SubAgent:
        """Create a custom specialist agent for a domain."""
        # Use LLM to design the specialist
        prompt = f"""Design a specialist agent for domain: {domain}

Goal: {goal}

Specify:
1. System prompt / instructions
2. Required tools (from: fs_read, fs_write, fs_list, shell_exec, python_exec, web_search, web_fetch, git_*)
3. Recommended agent type (generic/designer/monitor/reflector)
4. Estimated timeout
5. Token budget

Return JSON."""

        response = await self.llm.complete(prompt)

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                design = json.loads(json_match.group())

                spec = SubAgentSpec(
                    name=f"specialist_{domain}_{uuid.uuid4().hex[:6]}",
                    agent_type=AgentType(design.get("agent_type", "generic")),
                    goal=goal,
                    instructions=design.get("instructions", ""),
                    tools=design.get("tools", []),
                    timeout=design.get("timeout", 300),
                    token_budget=design.get("token_budget", 50000),
                )

                return await self.spawn(spec)
        except Exception as e:
            logger.error(f"Specialist creation failed: {e}")

        # Fallback
        return await self.spawn(SubAgentSpec(
            name=f"specialist_{domain}",
            agent_type=AgentType.GENERIC,
            goal=goal,
        ))


# Global instance
_spawner: AgentSpawner | None = None


def get_spawner() -> AgentSpawner:
    global _spawner
    if _spawner is None:
        _spawner = AgentSpawner()
    return _spawner