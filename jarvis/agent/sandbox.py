"""Sandboxed code execution (subprocess, Docker, nsjail)."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import docker
from loguru import logger

from jarvis.agent.config import get_agent_config, get_workspace_dir


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False
    artifacts: dict[str, Any] | None = None


class Sandbox:
    """Base sandbox class."""

    def __init__(self) -> None:
        self.config = get_agent_config()
        self._repl_sessions: dict[str, dict] = {}

    async def run_python(
        self,
        code: str,
        timeout: int = 30,
        packages: list[str] | None = None,
        return_vars: list[str] | None = None,
    ) -> dict:
        raise NotImplementedError

    async def run_repl(
        self,
        code: str,
        session_id: str | None = None,
        timeout: int = 30,
    ) -> dict:
        raise NotImplementedError

    async def install_package(self, package: str, timeout: int = 60) -> dict:
        raise NotImplementedError

    def _prepare_code(
        self,
        code: str,
        packages: list[str] | None = None,
        return_vars: list[str] | None = None,
    ) -> str:
        """Wrap code with imports and return value capture."""
        lines = []

        # Install packages if needed
        if packages:
            lines.append("import subprocess, sys")
            for pkg in packages:
                lines.append(f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', '{pkg}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)")

        # User code
        lines.append(code)

        # Capture return variables
        if return_vars:
            lines.append("\n# Capture return values")
            lines.append("import json")
            lines.append("__result__ = {}")
            for var in return_vars:
                lines.append(f"try: __result__['{var}'] = {var}")
                lines.append("except NameError: pass")
            lines.append("print('__RESULT__', json.dumps(__result__, default=str))")

        return "\n".join(lines)


class SubprocessSandbox(Sandbox):
    """Subprocess-based sandbox (current process isolation)."""

    def __init__(self) -> None:
        super().__init__()
        self._session_counter = 0
        self._repl_globals: dict[str, dict] = {}

    async def run_python(
        self,
        code: str,
        timeout: int = 30,
        packages: list[str] | None = None,
        return_vars: list[str] | None = None,
    ) -> dict:
        wrapped = self._prepare_code(code, packages, return_vars)

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapped)
            script_path = Path(f.name)

        try:
            # Run in subprocess with restricted environment
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONUNBUFFERED"] = "1"

            # Ensure workspace directory exists
            workspace = get_workspace_dir()

            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(workspace),
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Execution timed out after {timeout}s",
                    "returncode": -1,
                    "timed_out": True,
                }

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Parse result
            result_vars = {}
            if return_vars:
                for line in stdout_str.split("\n"):
                    if line.startswith("__RESULT__"):
                        try:
                            result_vars = json.loads(line[11:])
                        except json.JSONDecodeError:
                            pass

            return {
                "success": proc.returncode == 0,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "returncode": proc.returncode,
                "timed_out": False,
                "variables": result_vars,
            }

        finally:
            script_path.unlink(missing_ok=True)

    async def run_repl(
        self,
        code: str,
        session_id: str | None = None,
        timeout: int = 30,
    ) -> dict:
        if session_id is None:
            session_id = f"repl_{uuid.uuid4().hex[:8]}"

        if session_id not in self._repl_globals:
            self._repl_globals[session_id] = {}

        # Simple approach: execute in persistent process via stdin
        # For now, just execute with previous globals simulated
        globals_dict = self._repl_globals[session_id]

        # Build code that updates globals
        wrapped = f"""
import sys
import json

# Restore globals
{chr(10).join(f"{k} = {repr(v)}" for k, v in globals_dict.items())}

# User code
{code}

# Capture updated globals
__updated__ = {{}}
for k in list(globals().keys()):
    if not k.startswith('__') and k not in {set(globals_dict.keys())}:
        try:
            json.dumps(globals()[k], default=str)
            __updated__[k] = globals()[k]
        except:
            pass

print('__GLOBALS__', json.dumps(__updated__, default=str))
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapped)
            script_path = Path(f.name)

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {"success": False, "stdout": "", "stderr": "Timeout", "returncode": -1, "timed_out": True}

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Extract globals
            for line in stdout_str.split("\n"):
                if line.startswith("__GLOBALS__"):
                    try:
                        updated = json.loads(line[12:])
                        globals_dict.update(updated)
                    except json.JSONDecodeError:
                        pass

            return {
                "success": proc.returncode == 0,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "returncode": proc.returncode,
                "timed_out": False,
                "session_id": session_id,
            }

        finally:
            script_path.unlink(missing_ok=True)

    async def install_package(self, package: str, timeout: int = 60) -> dict:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", package,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"success": False, "error": "Timeout"}

        return {
            "success": proc.returncode == 0,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
        }


class DockerSandbox(Sandbox):
    """Docker-based sandbox for stronger isolation."""

    def __init__(self) -> None:
        super().__init__()
        self._client: docker.DockerClient | None = None
        self._image = self.config.sandbox.docker_image

    async def _get_client(self) -> docker.DockerClient:
        if self._client is None:
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(None, docker.from_env)
        return self._client

    async def _ensure_image(self) -> None:
        client = await self._get_client()
        try:
            await asyncio.get_event_loop().run_in_executor(None, client.images.get, self._image)
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling Docker image: {self._image}")
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.images.pull(self._image),
            )

    async def run_python(
        self,
        code: str,
        timeout: int = 30,
        packages: list[str] | None = None,
        return_vars: list[str] | None = None,
    ) -> dict:
        await self._ensure_image()
        client = await self._get_client()

        # Prepare script
        script = self._prepare_code(code, packages, return_vars)

        # Create container
        container = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.containers.create(
                self._image,
                command=["python3", "-c", script],
                mem_limit=self.config.sandbox.memory_limit,
                cpu_quota=int(self.config.sandbox.cpu_limit * 100000),
                network_mode="none" if not self.config.sandbox.network_enabled else "bridge",
                remove=True,
            ),
        )

        try:
            # Start and wait
            await asyncio.get_event_loop().run_in_executor(None, container.start)

            try:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, container.wait, timeout),
                    timeout=timeout + 5,
                )
            except asyncio.TimeoutError:
                await asyncio.get_event_loop().run_in_executor(None, container.kill)
                return {"success": False, "stdout": "", "stderr": "Timeout", "returncode": -1, "timed_out": True}

            logs = await asyncio.get_event_loop().run_in_executor(None, container.logs, True, True)
            stdout, stderr = logs

            return {
                "success": result["StatusCode"] == 0,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "returncode": result["StatusCode"],
                "timed_out": False,
            }

        except Exception as e:
            logger.error(f"Docker sandbox error: {e}")
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}

    async def run_repl(self, code: str, session_id: str | None = None, timeout: int = 30) -> dict:
        # Docker REPL is complex, fallback to subprocess
        return await SubprocessSandbox().run_repl(code, session_id, timeout)

    async def install_package(self, package: str, timeout: int = 60) -> dict:
        # Install in image by committing
        await self._ensure_image()
        client = await self._get_client()

        container = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.containers.run(
                self._image,
                command=["pip", "install", package],
                detach=True,
                remove=False,
            ),
        )

        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, container.wait, timeout),
                timeout=timeout + 10,
            )
            logs = await asyncio.get_event_loop().run_in_executor(None, container.logs, True, True)
            stdout, stderr = logs

            if result["StatusCode"] == 0:
                # Commit new image
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: container.commit(self._image),
                )

            return {
                "success": result["StatusCode"] == 0,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
            }
        finally:
            await asyncio.get_event_loop().run_in_executor(None, container.remove, True)


# Global sandbox instance
_sandbox: Sandbox | None = None


def get_sandbox() -> Sandbox:
    global _sandbox
    if _sandbox is None:
        config = get_agent_config()
        if config.sandbox.type == "docker":
            _sandbox = DockerSandbox()
        else:
            _sandbox = SubprocessSandbox()
    return _sandbox