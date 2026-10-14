"""File system tools for agents."""
from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

from jarvis.agent.tools import tool, get_tool_registry
from jarvis.agent.config import get_agent_config, get_workspace_dir
from jarvis.config import get_config


def _resolve_path(path: str) -> Path:
    """Resolve and validate a path against the allowed roots.

    Relative paths are anchored in the agent workspace rather than the process
    CWD, so a bare ``hello.py`` lands somewhere writable instead of being
    rejected outright.
    """
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = get_workspace_dir() / raw
    target = raw.resolve()

    allowed_roots = [
        Path(p).expanduser().resolve()
        for p in (
            list(get_config().system_control.allowed_paths)
            + list(get_agent_config().sandbox.allowed_paths)
        )
        if p
    ]

    for root in allowed_roots:
        try:
            target.relative_to(root)
            return target
        except ValueError:
            continue

    raise PermissionError(
        f"Path not allowed: {target}. "
        f"Allowed roots: {', '.join(str(r) for r in allowed_roots) or '(none)'}. "
        f"Write a relative path to work inside the workspace: {get_workspace_dir()}"
    )


@tool("fs_read", "Read file contents")
async def fs_read(path: str) -> str:
    """Read a file and return its contents."""
    target = _resolve_path(path)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {target}")
    if not target.is_file():
        raise ValueError(f"Not a file: {target}")

    return await asyncio.to_thread(target.read_text, encoding="utf-8")


@tool("fs_write", "Write content to file")
async def fs_write(path: str, content: str, create_dirs: bool = True) -> str:
    """Write content to a file."""
    target = _resolve_path(path)
    if create_dirs:
        target.parent.mkdir(parents=True, exist_ok=True)

    await asyncio.to_thread(target.write_text, content, encoding="utf-8")
    return f"Written {len(content)} chars to {target}"


@tool("fs_append", "Append content to file")
async def fs_append(path: str, content: str) -> str:
    """Append content to a file."""
    target = _resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    await asyncio.to_thread(lambda: target.open("a", encoding="utf-8").write(content))
    return f"Appended {len(content)} chars to {target}"


@tool("fs_list", "List directory contents")
async def fs_list(path: str = ".", recursive: bool = False, pattern: str = "*") -> str:
    """List files and directories."""
    target = _resolve_path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")
    if not target.is_dir():
        raise ValueError(f"Not a directory: {target}")

    if recursive:
        items = list(target.rglob(pattern))
    else:
        items = list(target.glob(pattern))

    lines = []
    for item in sorted(items):
        rel = item.relative_to(target)
        prefix = "📁" if item.is_dir() else "📄"
        size = item.stat().st_size if item.is_file() else 0
        lines.append(f"{prefix} {rel} ({size} bytes)")

    return "\n".join(lines) if lines else "(empty)"


@tool("fs_delete", "Delete file or directory")
async def fs_delete(path: str, recursive: bool = False) -> str:
    """Delete a file or directory."""
    target = _resolve_path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    if target.is_dir():
        if recursive:
            await asyncio.to_thread(shutil.rmtree, target)
            return f"Deleted directory {target} recursively"
        else:
            await asyncio.to_thread(target.rmdir)
            return f"Deleted empty directory {target}"
    else:
        await asyncio.to_thread(target.unlink)
        return f"Deleted file {target}"


@tool("fs_copy", "Copy file or directory")
async def fs_copy(src: str, dst: str, overwrite: bool = False) -> str:
    """Copy file or directory."""
    src_path = _resolve_path(src)
    dst_path = _resolve_path(dst)

    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {src_path}")

    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"Destination exists: {dst_path}")

    if src_path.is_dir():
        await asyncio.to_thread(shutil.copytree, src_path, dst_path, dirs_exist_ok=overwrite)
        return f"Copied directory {src_path} to {dst_path}"
    else:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, src_path, dst_path)
        return f"Copied file {src_path} to {dst_path}"


@tool("fs_move", "Move/rename file or directory")
async def fs_move(src: str, dst: str, overwrite: bool = False) -> str:
    """Move or rename file or directory."""
    src_path = _resolve_path(src)
    dst_path = _resolve_path(dst)

    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {src_path}")

    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"Destination exists: {dst_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(shutil.move, str(src_path), str(dst_path))
    return f"Moved {src_path} to {dst_path}"


@tool("fs_exists", "Check if path exists")
async def fs_exists(path: str) -> dict:
    """Check if path exists and get info."""
    target = Path(path).expanduser().resolve()
    exists = target.exists()

    result = {"exists": exists, "path": str(target)}
    if exists:
        result["is_file"] = target.is_file()
        result["is_dir"] = target.is_dir()
        stat = target.stat()
        result["size"] = stat.st_size
        result["modified"] = stat.st_mtime

    return result


@tool("fs_mkdir", "Create directory")
async def fs_mkdir(path: str, parents: bool = True, exist_ok: bool = True) -> str:
    """Create directory."""
    target = _resolve_path(path)
    await asyncio.to_thread(target.mkdir, parents=parents, exist_ok=exist_ok)
    return f"Created directory {target}"


# Register all tools
def register_fs_tools() -> None:
    """Ensure all FS tools are registered."""
    pass  # Decorators auto-register