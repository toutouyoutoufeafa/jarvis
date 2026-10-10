"""Git tools for agents."""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

from jarvis.agent.tools import tool
from jarvis.agent.tools.fs import _resolve_path


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run git command."""
    return subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
    )


@tool("git_init", "Initialize git repository")
async def git_init(path: str = ".", bare: bool = False) -> str:
    target = _resolve_path(path)
    target.mkdir(parents=True, exist_ok=True)
    result = await asyncio.to_thread(_run_git, ["init"] + (["--bare"] if bare else []), target)
    if result.returncode != 0:
        raise RuntimeError(f"git init failed: {result.stderr}")
    return f"Initialized git repo at {target}"


@tool("git_clone", "Clone git repository")
async def git_clone(url: str, path: str, branch: str | None = None) -> str:
    target = _resolve_path(path)
    if target.exists():
        raise FileExistsError(f"Path exists: {target}")

    args = ["clone"]
    if branch:
        args.extend(["--branch", branch])
    args.extend([url, str(target)])

    result = await asyncio.to_thread(_run_git, args, target.parent)
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr}")
    return f"Cloned {url} to {target}"


@tool("git_status", "Get git status")
async def git_status(path: str = ".") -> dict:
    target = _resolve_path(path)
    result = await asyncio.to_thread(_run_git, ["status", "--porcelain"], target)
    lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
    return {
        "clean": len(lines) == 0,
        "changes": lines,
        "branch": await git_branch(path),
    }


@tool("git_branch", "Get current branch")
async def git_branch(path: str = ".") -> str:
    target = _resolve_path(path)
    result = await asyncio.to_thread(_run_git, ["branch", "--show-current"], target)
    return result.stdout.strip() or "unknown"


@tool("git_log", "Get git log")
async def git_log(path: str = ".", max_count: int = 10, oneline: bool = True) -> list[dict]:
    target = _resolve_path(path)
    args = ["log", f"--max-count={max_count}"]
    if oneline:
        args.append("--oneline")
    else:
        args.extend(["--pretty=format:%H|%an|%ad|%s", "--date=iso"])

    result = await asyncio.to_thread(_run_git, args, target)
    commits = []

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        if oneline:
            parts = line.split(" ", 1)
            commits.append({"hash": parts[0], "message": parts[1] if len(parts) > 1 else ""})
        else:
            parts = line.split("|", 3)
            commits.append({
                "hash": parts[0],
                "author": parts[1],
                "date": parts[2],
                "message": parts[3] if len(parts) > 3 else "",
            })

    return commits


@tool("git_add", "Stage files")
async def git_add(path: str = ".", files: list[str] | None = None) -> str:
    target = _resolve_path(path)
    args = ["add"] + (files or ["."])
    result = await asyncio.to_thread(_run_git, args, target)
    if result.returncode != 0:
        raise RuntimeError(f"git add failed: {result.stderr}")
    return "Files staged"


@tool("git_commit", "Create commit")
async def git_commit(message: str, path: str = ".", author: str | None = None) -> str:
    target = _resolve_path(path)
    args = ["commit", "-m", message]
    if author:
        args.extend(["--author", author])

    result = await asyncio.to_thread(_run_git, args, target)
    if result.returncode != 0:
        raise RuntimeError(f"git commit failed: {result.stderr}")

    # Get commit hash
    hash_result = await asyncio.to_thread(_run_git, ["rev-parse", "HEAD"], target)
    return f"Committed: {hash_result.stdout.strip()[:8]}"


@tool("git_push", "Push to remote")
async def git_push(path: str = ".", remote: str = "origin", branch: str | None = None, force: bool = False) -> str:
    target = _resolve_path(path)
    args = ["push", remote]
    if branch:
        args.append(branch)
    if force:
        args.append("--force")

    result = await asyncio.to_thread(_run_git, args, target)
    if result.returncode != 0:
        raise RuntimeError(f"git push failed: {result.stderr}")
    return "Push successful"


@tool("git_pull", "Pull from remote")
async def git_pull(path: str = ".", remote: str = "origin", branch: str | None = None) -> str:
    target = _resolve_path(path)
    args = ["pull", remote]
    if branch:
        args.append(branch)

    result = await asyncio.to_thread(_run_git, args, target)
    if result.returncode != 0:
        raise RuntimeError(f"git pull failed: {result.stderr}")
    return result.stdout.strip()


@tool("git_diff", "Show diff")
async def git_diff(path: str = ".", staged: bool = False, file: str | None = None) -> str:
    target = _resolve_path(path)
    args = ["diff"]
    if staged:
        args.append("--cached")
    if file:
        args.append(file)

    result = await asyncio.to_thread(_run_git, args, target)
    return result.stdout


@tool("git_checkout", "Checkout branch or commit")
async def git_checkout(path: str = ".", target: str = "", create: bool = False) -> str:
    repo_path = _resolve_path(path)
    args = ["checkout"]
    if create:
        args.append("-b")
    args.append(target)

    result = await asyncio.to_thread(_run_git, args, repo_path)
    if result.returncode != 0:
        raise RuntimeError(f"git checkout failed: {result.stderr}")
    return f"Checked out {target}"


def register_git_tools() -> None:
    pass