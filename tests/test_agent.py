"""Tests for the agent runtime: action parsing, config loading, file safety."""
from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.agent.core import AgentCore
from jarvis.agent.tools import get_tool_registry
from jarvis.agent.tools import fs as fs_tools
from jarvis.llm import extract_json


class TestExtractJson:
    def test_plain_object(self) -> None:
        assert extract_json('{"type":"plan"}') == {"type": "plan"}

    def test_markdown_fences(self) -> None:
        raw = 'Voici:\n```json\n{"type":"answer","content":"x"}\n```'
        assert extract_json(raw) == {"type": "answer", "content": "x"}

    def test_trailing_garbage(self) -> None:
        raw = '{"type":"tool","tool":"fs_write","args":{}}}'
        assert extract_json(raw) == {"type": "tool", "tool": "fs_write", "args": {}}

    def test_leading_prose(self) -> None:
        assert extract_json('Sure, here you go: {"type":"plan"} done') == {"type": "plan"}

    def test_no_json(self) -> None:
        assert extract_json("no json at all") is None
        assert extract_json("") is None
        assert extract_json(None) is None

    def test_array(self) -> None:
        assert extract_json('[{"a":1}]') == [{"a": 1}]


class _ActionNormalizer:
    """Reuses AgentCore's normalization without booting the whole runtime."""

    _normalize_action = AgentCore._normalize_action
    _as_tool_action = AgentCore._as_tool_action
    _coerce_args = AgentCore._coerce_args

    def __init__(self, tools) -> None:
        self.tools = tools


@pytest.fixture()
def normalizer() -> _ActionNormalizer:
    from jarvis.agent.tools import shell as shell_tools

    fs_tools.register_fs_tools()
    shell_tools.register_shell_tools()
    return _ActionNormalizer(tools=get_tool_registry())


class TestActionNormalization:
    def test_tool_shorthand(self, normalizer: _ActionNormalizer) -> None:
        """Small models put the tool name in `type` instead of `tool`."""
        action = normalizer._normalize_action(
            {"type": "fs_write", "args": {"path": "a.py", "content": "x = 1"}}
        )
        assert action["type"] == "tool"
        assert action["tool"] == "fs_write"
        assert action["args"] == {"path": "a.py", "content": "x = 1"}

    def test_tool_shorthand_without_args(self, normalizer: _ActionNormalizer) -> None:
        action = normalizer._normalize_action(
            {"type": "fs_read", "path": "a.py"}
        )
        assert action["type"] == "tool"
        assert action["tool"] == "fs_read"
        assert action["args"] == {"path": "a.py"}

    def test_execute_with_command_maps_to_shell(self, normalizer: _ActionNormalizer) -> None:
        action = normalizer._normalize_action({"type": "execute", "args": {"cmd": "dir"}})
        assert action == {"type": "tool", "tool": "shell_exec", "args": {"command": "dir"}}

    def test_execute_with_code_maps_to_python(self, normalizer: _ActionNormalizer) -> None:
        action = normalizer._normalize_action(
            {"type": "execute", "args": {"code": "print(1)"}}
        )
        assert action["tool"] == "python_exec"
        assert action["args"]["code"] == "print(1)"

    def test_execute_wrapping_tool(self, normalizer: _ActionNormalizer) -> None:
        action = normalizer._normalize_action(
            {"type": "execute", "args": {"tool": "fs_exists", "args": {"path": "x"}}}
        )
        assert action["tool"] == "fs_exists"
        assert action["args"] == {"path": "x"}

    def test_aliased_argument_renamed(self, normalizer: _ActionNormalizer) -> None:
        """`data`/`text` are accepted in place of the required `content`."""
        action = normalizer._normalize_action(
            {"type": "tool", "tool": "fs_write",
             "args": {"path": "a.py", "data": "print(1)"}}
        )
        assert action["args"]["content"] == "print(1)"
        assert "data" not in action["args"]

    def test_canonical_actions_pass_through(self, normalizer: _ActionNormalizer) -> None:
        assert normalizer._normalize_action({"type": "plan"}) == {"type": "plan"}
        answer = normalizer._normalize_action({"type": "answer", "content": "ok"})
        assert answer == {"type": "answer", "content": "ok"}

    def test_content_alias_for_answer(self, normalizer: _ActionNormalizer) -> None:
        action = normalizer._normalize_action({"type": "answer", "text": "ok"})
        assert action == {"type": "answer", "content": "ok"}

    def test_unknown_type_waits_instead_of_crashing(self, normalizer: _ActionNormalizer) -> None:
        action = normalizer._normalize_action({"type": "dance"})
        assert action["type"] == "wait"

    def test_non_dict_action(self, normalizer: _ActionNormalizer) -> None:
        assert normalizer._normalize_action(["nope"])["type"] == "wait"

    def test_tool_without_name_waits(self, normalizer: _ActionNormalizer) -> None:
        assert normalizer._normalize_action({"type": "tool"})["type"] == "wait"


class TestToolSchema:
    def test_compact_line_mentions_signature(self) -> None:
        fs_tools.register_fs_tools()
        described = get_tool_registry().describe_all()
        assert "fs_write(" in described
        assert "content" in described.split("fs_write(", 1)[1].split("\n", 1)[0]


class TestAgentConfig:
    def test_yaml_section_is_loaded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from jarvis.agent import config as agent_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "agent:\n  max_iterations: 3\n  default_model: 'qwen2.5:3b'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("JARVIS_CONFIG", str(cfg_file))
        monkeypatch.setattr(agent_config, "_agent_config", None)

        cfg = agent_config.get_agent_config()
        assert cfg.max_iterations == 3
        assert cfg.default_model == "qwen2.5:3b"
        # Defaults still apply for keys absent from the file
        assert cfg.sandbox.timeout == 30

    def test_missing_file_falls_back_to_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jarvis.agent import config as agent_config

        monkeypatch.setenv("JARVIS_CONFIG", str(tmp_path / "absent.yaml"))
        monkeypatch.setattr(agent_config, "_agent_config", None)

        cfg = agent_config.get_agent_config()
        assert cfg.max_iterations >= 1
        assert cfg.default_model

    def test_workspace_is_created(self) -> None:
        from jarvis.agent.config import get_workspace_dir

        workspace = get_workspace_dir()
        assert workspace.is_dir()


class TestPathSafety:
    def test_relative_path_lands_in_workspace(self) -> None:
        from jarvis.agent.config import get_workspace_dir
        from jarvis.agent.tools.fs import _resolve_path

        resolved = _resolve_path("hello.py")
        assert resolved == get_workspace_dir() / "hello.py"

    def test_workspace_itself_allowed(self) -> None:
        from jarvis.agent.config import get_workspace_dir
        from jarvis.agent.tools.fs import _resolve_path

        assert _resolve_path(str(get_workspace_dir() / "sub" / "x.txt"))

    @pytest.mark.parametrize("path", [
        "C:\\Windows\\System32\\evil.dll",
        "C:\\secret.txt",
    ])
    def test_outside_roots_rejected(self, path: str) -> None:
        from jarvis.agent.tools.fs import _resolve_path

        with pytest.raises(PermissionError):
            _resolve_path(path)


class TestPlannerParsing:
    def test_unknown_tool_dropped(self) -> None:
        from jarvis.agent.planner import get_planner

        planner = get_planner()
        plan = planner._parse_plan(
            '{"steps": [{"id": "s1", "description": "d", "tool": "file_write", "args": {}}]}',
            "goal",
        )
        assert len(plan.steps) == 1
        assert plan.steps[0].tool is None

    def test_known_tool_kept(self) -> None:
        from jarvis.agent.planner import get_planner

        planner = get_planner()
        plan = planner._parse_plan(
            '{"steps": [{"id": "s1", "description": "d", "tool": "fs_list", "args": {"path": "."}}]}',
            "goal",
        )
        assert plan.steps[0].tool == "fs_list"

    def test_invalid_json_raises(self) -> None:
        from jarvis.agent.planner import get_planner

        planner = get_planner()
        with pytest.raises(ValueError):
            planner._parse_plan("no json here", "goal")


class TestSandbox:
    @pytest.mark.asyncio
    async def test_run_python(self) -> None:
        from jarvis.agent.sandbox import get_sandbox

        result = await get_sandbox().run_python("print(21 * 2)", timeout=30)
        assert result["success"] is True
        assert "42" in result["stdout"]

    @pytest.mark.asyncio
    async def test_run_python_timeout_is_reported(self) -> None:
        from jarvis.agent.sandbox import get_sandbox

        result = await get_sandbox().run_python("import time; time.sleep(20)", timeout=2)
        assert result["success"] is False
        assert result["timed_out"] is True

    @pytest.mark.asyncio
    async def test_run_python_captures_variables(self) -> None:
        from jarvis.agent.sandbox import get_sandbox

        result = await get_sandbox().run_python(
            "answer = 6 * 7", timeout=30, return_vars=["answer"]
        )
        assert result["variables"] == {"answer": 42}

    def test_prepare_code_is_shared_by_both_sandboxes(self) -> None:
        from jarvis.agent.sandbox import DockerSandbox, SubprocessSandbox

        assert hasattr(DockerSandbox(), "_prepare_code")
        assert SubprocessSandbox()._prepare_code("x = 1") == "x = 1"


class TestShellTool:
    @pytest.mark.asyncio
    async def test_default_cwd_is_the_workspace(self) -> None:
        from jarvis.agent.config import get_workspace_dir
        from jarvis.agent.tools.shell import shell_exec

        result = await shell_exec("cd", timeout=15)
        assert result["success"] is True
        assert str(get_workspace_dir()).lower() in result["stdout"].lower()

    @pytest.mark.asyncio
    async def test_missing_binary_returns_failure(self) -> None:
        from jarvis.agent.tools.shell import shell_exec

        result = await shell_exec("definitely_not_a_command_xyz", timeout=15)
        assert result["success"] is False


class _FakeLLM:
    """Canned responses, consumed one per call."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    async def complete(self, *args, **kwargs) -> str:
        if self.responses:
            return self.responses.pop(0)
        return '{"type":"answer","content":"plus de reponses"}'

    async def initialize(self) -> None:
        pass


class _FakeMemory:
    def __init__(self) -> None:
        self.entries: list[str] = []

    def add_stm(self, content: str, metadata: dict | None = None) -> None:
        self.entries.append(content)

    async def get_context(self, *args, **kwargs) -> str:
        return "\n".join(self.entries[-10:])

    async def initialize(self) -> None:
        pass


def _make_core(responses: list[str]) -> AgentCore:
    core = AgentCore()
    core.llm = _FakeLLM(responses)
    core.memory = _FakeMemory()
    core._register_builtin_tools()
    return core


@pytest.mark.asyncio
class TestAgentLoop:
    async def test_unknown_tool_is_an_observation_not_a_crash(self) -> None:
        core = _make_core([
            '{"type":"tool","tool":"file_write","args":{"path":"x.txt","content":"y"}}',
            '{"type":"answer","content":"FAILED: outil file_write inexistant"}',
        ])
        task = await core.run_task("ecris un fichier")

        assert task.state.value == "done"
        assert task.result.startswith("FAILED")
        assert any(e.startswith("ERROR") for e in core.memory.entries)
        assert task.tool_failures >= 1

    async def test_task_succeeds_and_reports_result(self) -> None:
        from jarvis.agent.config import get_workspace_dir

        name = "test_agent_ok.txt"
        (get_workspace_dir() / name).unlink(missing_ok=True)

        core = _make_core([
            '{"type":"tool","tool":"fs_write","args":{"path":"'
            + name + '","content":"bonjour"}}',
            '{"type":"answer","content":"fichier cree"}',
        ])
        task = await core.run_task("ecris un fichier")

        assert task.state.value == "done"
        assert task.result == "fichier cree"
        assert (get_workspace_dir() / name).read_text(encoding="utf-8") == "bonjour"

    async def test_tool_shorthand_action_is_executed(self) -> None:
        from jarvis.agent.config import get_workspace_dir

        name = "test_agent_shorthand.txt"
        (get_workspace_dir() / name).unlink(missing_ok=True)

        core = _make_core([
            '{"type":"fs_write","args":{"path":"' + name + '","content":"ok"}}',
            '{"type":"answer","content":"fait"}',
        ])
        task = await core.run_task("ecris un fichier")

        assert task.result == "fait"
        assert (get_workspace_dir() / name).exists()

    async def test_repeated_identical_call_stops_the_loop(self) -> None:
        call = '{"type":"tool","tool":"fs_mkdir","args":{"path":"loop_dir"}}'
        core = _make_core([call, call, call, call, call])
        task = await core.run_task("cree un dossier")

        assert task.state.value == "done"
        assert "boucle interrompue" in str(task.result)
        assert task.iterations < 5

    async def test_answer_without_any_tool_is_refused_once(self) -> None:
        """A goal-free 'done' on iteration 1 is a shortcut, not a result."""
        core = _make_core([
            '{"type":"answer","content":"C\'est fait"}',
            '{"type":"answer","content":"C\'est fait"}',
        ])
        task = await core.run_task("cree un fichier x.txt")

        assert task.answer_rejected == 1
        assert any("without calling any tool" in e for e in core.memory.entries)

    async def test_tool_free_answer_is_kept_on_second_attempt(self) -> None:
        """Genuinely conversational goals pay one extra round-trip, then win."""
        core = _make_core([
            '{"type":"answer","content":"Aucun outil necessaire"}',
            '{"type":"answer","content":"Aucun outil necessaire, confirme"}',
        ])
        task = await core.run_task("bonjour")

        assert task.result == "Aucun outil necessaire, confirme"
        assert task.answer_rejected == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
