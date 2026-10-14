"""Run the real agent task while timing every LLM call."""
import asyncio
import os
import time
from pathlib import Path

from jarvis.agent.core import AgentCore


async def main() -> None:
    core = AgentCore()
    core._register_builtin_tools()
    await core.initialize()

    calls = []
    original = core.llm.complete

    async def timed(user_input, *args, **kwargs):
        start = time.perf_counter()
        out = await original(user_input, *args, **kwargs)
        elapsed = time.perf_counter() - start
        calls.append(elapsed)
        system = kwargs.get("system_prompt") or ""
        print(f"  LLM {elapsed:6.2f}s  user={len(user_input):5d} chars  "
              f"sys={len(system):5d} chars")
        return out

    core.llm.complete = timed

    model = os.environ.get("JARVIS_BENCH_MODEL")
    if model:
        core.llm.set_model(model)
    print(f"modele: {core.llm.get_model()}")

    start = time.perf_counter()
    task = await core.run_task("Cree un fichier vitesse.py qui affiche VITE")
    total = time.perf_counter() - start

    workspace = Path.home() / ".jarvis" / "workspace"
    produced = (workspace / "vitesse.py").read_text(encoding="utf-8") \
        if (workspace / "vitesse.py").exists() else "(absent)"

    print(f"\nTOTAL  : {total:.1f}s")
    print(f"appels : {len(calls)} LLM, cumules {sum(calls):.1f}s")
    print(f"state  : {task.state.value}  iterations={task.iterations}")
    print(f"result : {task.result}")
    print(f"fichier: {produced!r}")


if __name__ == "__main__":
    asyncio.run(main())
