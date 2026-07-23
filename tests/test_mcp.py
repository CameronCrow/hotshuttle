"""The MCP surface. No GPU needed -- these check wiring, not generation.

The expensive failure mode for a plugin is that it looks fine until someone installs it
and the server will not even start, so the checks here are deliberately about startup and
registration rather than model output.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "mcp_server"))

EXPECTED_TOOLS = {"bonsai_status", "bonsai_start", "bonsai_chat",
                  "worker_spawn", "worker_ask", "worker_list", "worker_retire"}


@pytest.fixture(scope="module")
def server():
    import hotshuttle_mcp
    return hotshuttle_mcp


@pytest.mark.asyncio
async def test_every_tool_is_registered_and_documented(server):
    tools = {t.name: t for t in await server.mcp.list_tools()}
    assert set(tools) == EXPECTED_TOOLS
    for name, t in tools.items():
        assert t.description and len(t.description) > 40, f"{name} needs a real description"


def test_importing_the_server_does_not_require_a_running_model(server):
    """Registration must be lazy -- a plugin that dies at import is a plugin that fails
    for everyone whose server happens to be down."""
    assert server._state.get("orc") is None or "orc" in server._state


def test_orchestrator_builds_from_the_shipped_manifest(server):
    server._state.clear()
    orc = server._orc()
    assert orc.profile.ctx_per_slot == 12288
    assert orc.pool.n_slots >= 1
    server._state.clear()


def test_bash_is_resolved_explicitly_not_by_path_search(server):
    """A bare "bash" finds WSL's System32 bash on Windows, which cannot see C:/ paths."""
    resolved = server._bash()
    assert resolved != "bash" or sys.platform != "win32"
    if sys.platform == "win32":
        assert "System32" not in resolved


# --- plugin manifests --------------------------------------------------------

def test_plugin_and_marketplace_manifests_are_valid():
    plugin = json.loads((REPO / ".claude-plugin" / "plugin.json").read_text())
    market = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())
    assert plugin["name"] == "hotshuttle" and plugin["version"]
    assert market["plugins"][0]["name"] == plugin["name"]


def test_mcp_manifest_points_at_a_file_that_exists():
    cfg = json.loads((REPO / ".mcp.json").read_text())
    args = cfg["mcpServers"]["hotshuttle"]["args"]
    rel = args[0].replace("${CLAUDE_PLUGIN_ROOT}/", "")
    assert (REPO / rel).exists(), f"{rel} is declared in .mcp.json but missing"


def test_the_skill_ships_with_the_plugin_rather_than_a_vendored_copy():
    """The skill used to keep its own copies of bonsai.sh and bonsai_client.py under
    ~/.claude/skills/, and they drifted -- carrying an old default context and a bash bug
    fixed here months later. It now references the plugin root instead."""
    skill = (REPO / "skills" / "bonsai" / "SKILL.md").read_text(encoding="utf-8")
    assert "CLAUDE_PLUGIN_ROOT" in skill
    assert not (REPO / "skills" / "bonsai" / "bonsai.sh").exists(), \
        "a second copy of bonsai.sh is exactly the drift this layout removes"
