"""Shell command tools for agents."""
from __future__ import annotations

import asyncio
import shlex
import subprocess
from typing import Any

from jarvis.agent.tools import tool, get_tool_registry
from jarvis.agent.config import get_agent_config


@tool("shell_exec", "Execute shell command")
async def shell_exec(
    command: str,
    timeout: int = 30,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    """Execute a shell command and return result."""
    config = get_agent_config()

    if not config.sandbox.enabled:
        raise PermissionError("Sandbox disabled, shell commands not allowed")

    # Prepare environment
    full_env = {**os.environ}
    if env:
        full_env.update(env)

    # Resolve cwd
    work_dir = Path(cwd).expanduser().resolve() if cwd else Path.cwd()

    # Validate cwd against allowed paths
    allowed = False
    for allowed_path in config.sandbox.allowed_paths:
        try:
            work_dir.relative_to(Path(allowed_path).expanduser().resolve())
            allowed = True
            break
        except ValueError:
            pass

    if not allowed:
        raise PermissionError(f"Working directory not allowed: {work_dir}")

    try:
        # Use asyncio subprocess for better control
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(work_dir),
            env=full_env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "returncode": -1,
                "timed_out": True,
            }

        return {
            "success": proc.returncode == 0,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "returncode": proc.returncode,
            "timed_out": False,
        }

    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "timed_out": False,
        }


@tool("shell_exec_stream", "Execute shell command with streaming output")
async def shell_exec_stream(
    command: str,
    timeout: int = 60,
    cwd: str | None = None,
) -> str:
    """Execute command and return combined output as string."""
    result = await shell_exec(command, timeout, cwd)
    output = result["stdout"]
    if result["stderr"]:
        output += f"\n--- STDERR ---\n{result['stderr']}"
    if not result["success"]:
        output += f"\n[Exit code: {result['returncode']}]"
    return output


@tool("which", "Find executable in PATH")
async def which(command: str) -> str:
    """Find executable path."""
    import shutil
    path = shutil.which(command)
    return path or f"Command not found: {command}"


import os
from pathlib import Path


def register_shell_tools() -> None:
    pass