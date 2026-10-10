"""Designer Agent - Code generation, architecture, documentation."""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from jarvis.agent.config import get_agent_config
from jarvis.agent.core import get_agent_core
from jarvis.agent.memory import get_memory_manager
from jarvis.agent.tools import get_tool_registry
from jarvis.llm import get_llm_engine


@dataclass
class DesignTask:
    type: str  # code, architecture, docs, review, refactor, scaffold
    spec: str
    language: str = "python"
    output_path: str | None = None
    context: dict = None


class DesignerAgent:
    """Specialized agent for design tasks."""

    def __init__(self) -> None:
        self.config = get_agent_config()
        self.llm = get_llm_engine()
        self.memory = get_memory_manager()
        self.tools = get_tool_registry()
        self.core = get_agent_core()

    async def initialize(self) -> None:
        await self.core.initialize()
        logger.info("Designer agent initialized")

    async def generate_code(self, task: DesignTask) -> dict:
        """Generate code from specification."""
        prompt = self._build_code_prompt(task)

        response = await self.llm.complete(prompt)
        code = self._extract_code(response, task.language)

        result = {
            "code": code,
            "language": task.language,
            "spec": task.spec,
        }

        # Save if path provided
        if task.output_path:
            await self.tools.execute("fs_write", path=task.output_path, content=code)
            result["saved_to"] = task.output_path

        # Store in memory
        self.memory.add_stm(
            f"Generated {task.language} code for: {task.spec[:100]}",
            metadata={"type": "code_generation", "language": task.language},
        )

        return result

    async def design_architecture(self, task: DesignTask) -> dict:
        """Create architecture design with Mermaid diagrams."""
        prompt = f"""Design a software architecture for: {task.spec}

Provide:
1. High-level overview
2. Component diagram (Mermaid)
3. Data flow diagram (Mermaid)
4. Technology choices with rationale
5. API contracts (OpenAPI spec if applicable)
6. Database schema (if applicable)
7. Deployment considerations
8. Security considerations
9. Scalability patterns

Format as structured markdown with Mermaid code blocks."""

        response = await self.llm.complete(prompt)

        result = {
            "architecture": response,
            "spec": task.spec,
        }

        if task.output_path:
            await self.tools.execute("fs_write", path=task.output_path, content=response)
            result["saved_to"] = task.output_path

        return result

    async def write_documentation(self, task: DesignTask) -> dict:
        """Generate documentation from code or spec."""
        # Read source files if path provided
        source_content = ""
        if task.output_path and Path(task.output_path).exists():
            source_content = await self.tools.execute("fs_read", path=task.output_path)

        prompt = f"""Write comprehensive documentation for: {task.spec}

Source code context:
{source_content[:5000] if source_content else 'N/A'}

Generate:
1. README.md with overview, installation, usage
2. API reference (if applicable)
3. Architecture decision records (ADR)
4. Configuration guide
5. Troubleshooting section
6. Contributing guidelines

Use clear markdown with code examples."""

        response = await self.llm.complete(prompt)

        result = {"documentation": response, "spec": task.spec}

        if task.output_path:
            await self.tools.execute("fs_write", path=task.output_path, content=response)
            result["saved_to"] = task.output_path

        return result

    async def review_code(self, task: DesignTask) -> dict:
        """Perform code review."""
        code = ""
        if task.output_path and Path(task.output_path).exists():
            code = await self.tools.execute("fs_read", path=task.output_path)
        elif "code" in (task.context or {}):
            code = task.context["code"]

        prompt = f"""Perform a thorough code review for {task.language} code:

```{task.language}
{code[:8000]}
```

Analyze:
1. Correctness & logic errors
2. Security vulnerabilities
3. Performance issues
4. Code style & best practices
5. Maintainability & readability
6. Test coverage gaps
7. Documentation gaps

Provide:
- Summary rating (Excellent/Good/Needs Work/Poor)
- List of issues by severity (Critical/High/Medium/Low)
- Specific suggestions with line references
- Refactoring recommendations"""

        response = await self.llm.complete(prompt)

        return {"review": response, "language": task.language}

    async def refactor_code(self, task: DesignTask) -> dict:
        """Refactor code for quality improvements."""
        code = ""
        if task.output_path and Path(task.output_path).exists():
            code = await self.tools.execute("fs_read", path=task.output_path)

        prompt = f"""Refactor this {task.language} code for: {task.spec}

Original code:
```{task.language}
{code[:8000]}
```

Apply:
- SOLID principles
- Design patterns where appropriate
- Remove duplication
- Improve naming
- Add type hints
- Extract functions/classes
- Add docstrings

Return ONLY the refactored code in a markdown code block."""

        response = await self.llm.complete(prompt)
        refactored = self._extract_code(response, task.language)

        result = {"refactored_code": refactored, "original_spec": task.spec}

        if task.output_path:
            # Backup original
            backup = f"{task.output_path}.bak"
            await self.tools.execute("fs_copy", src=task.output_path, dst=backup)
            # Write refactored
            await self.tools.execute("fs_write", path=task.output_path, content=refactored)
            result["saved_to"] = task.output_path
            result["backup"] = backup

        return result

    async def scaffold_project(self, task: DesignTask) -> dict:
        """Create project structure from template."""
        prompt = f"""Create a complete project scaffold for: {task.spec}

Language: {task.language}

Generate:
1. Directory structure
2. Configuration files (pyproject.toml, package.json, Cargo.toml, etc.)
3. CI/CD pipeline (GitHub Actions / GitLab CI)
4. Dockerfile + docker-compose.yml
5. README.md
6. .gitignore
7. Basic test structure
8. Linting/formatting config (ruff, prettier, etc.)
9. Pre-commit hooks
10. Development environment setup (devcontainer, Makefile, justfile)

Return as JSON with file paths and contents."""

        response = await self.llm.complete(prompt)

        # Parse JSON and write files
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        created = []

        if json_match:
            try:
                files = json.loads(json_match.group())
                for rel_path, content in files.items():
                    if task.output_path:
                        full_path = Path(task.output_path) / rel_path
                    else:
                        full_path = Path(rel_path)

                    await self.tools.execute("fs_write", path=str(full_path), content=content)
                    created.append(str(full_path))
            except json.JSONDecodeError:
                pass

        return {"scaffolded": created, "spec": task.spec}

    def _build_code_prompt(self, task: DesignTask) -> str:
        context = ""
        if task.context:
            context = f"\nContext: {json.dumps(task.context)}"

        return f"""Write {task.language} code for: {task.spec}{context}

Requirements:
- Clean, idiomatic {task.language}
- Type hints / type annotations
- Comprehensive docstrings
- Error handling
- Logging where appropriate
- Follow best practices and style guides
- No placeholder comments - complete implementation

Return ONLY the code in a markdown code block with language tag."""

    def _extract_code(self, response: str, language: str) -> str:
        """Extract code from markdown code block."""
        pattern = rf"```{language}\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: any code block
        match = re.search(r"```(?:\w+)?\n(.*?)\n```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        return response.strip()


# Global instance
_designer: DesignerAgent | None = None


def get_designer_agent() -> DesignerAgent:
    global _designer
    if _designer is None:
        _designer = DesignerAgent()
    return _designer