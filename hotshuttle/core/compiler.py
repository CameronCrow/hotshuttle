"""Prompt compiler -- owns everything about prompt *bytes*, because I3 makes bytes
load-bearing.

A worker's prompt is always:

    [seed: system + tools (+ "Prior progress: <summary>")]   byte-stable for its life
    [turn 1 user] [turn 1 model output] [turn 2 user] ...    append-only, forever

`cache_prompt` reuses only the common leading *prefix*, so rewriting anything early --
re-topping a summary, editing a prior turn, even a whitespace change in the seed --
invalidates the cache from that byte on and silently re-prefills. That is why compaction
is a reset rather than an edit (I4) and why `PrefixGuard` exists.

Pure functions from (role, history, task) to bytes. No retrieval, no ranking, no
summarization -- those are orchestrator policy.
"""
from __future__ import annotations

from .profile import ChatTemplate


def render_seed(tmpl: ChatTemplate, system: str, tools: list[str] = (),
                prior: str | None = None) -> str:
    """The byte-stable prefix: system block, tool list, and (after a compaction) a summary
    of what the worker had already done."""
    body = system.rstrip()
    if tools:
        body += "\n\nTools available: " + ", ".join(tools)
    if prior:
        body += "\n\nPrior progress:\n" + prior.strip()
    return tmpl.system(body)


def render_turn(tmpl: ChatTemplate, instruction: str, attach: list[str] = ()) -> str:
    """One user turn plus the assistant handoff, ready for the model to continue from.

    `attach` is the context the orchestrator deliberately selected for THIS turn (I5) --
    a worker never sees the orchestrator's own history.
    """
    body = instruction.strip()
    if attach:
        body += "\n\n" + "\n\n".join(f"<context>\n{a.strip()}\n</context>" for a in attach)
    return tmpl.user(body) + tmpl.assistant_open()


def close_turn(tmpl: ChatTemplate, output: str) -> str:
    """Append the model's own output verbatim, so the next prompt extends this one."""
    return output + tmpl.assistant_close()


class PrefixGuard:
    """Asserts the append-only contract (I3): every prompt emitted for a worker must be a
    byte-extension of the previous one.

    Cheap enough to leave on in production -- a violation means the next dispatch would
    silently re-prefill, and finding that from a latency graph is miserable.
    """

    def __init__(self) -> None:
        self._last: dict[str, str] = {}

    def check(self, worker_id: str, prompt: str) -> None:
        prev = self._last.get(worker_id)
        if prev is not None and not prompt.startswith(prev):
            i = _first_divergence(prev, prompt)
            raise AssertionError(
                f"worker {worker_id}: prompt is not append-only, diverges at byte {i}.\n"
                f"  was: ...{prev[max(0, i - 40):i + 40]!r}\n"
                f"  now: ...{prompt[max(0, i - 40):i + 40]!r}\n"
                f"This invalidates the prefix cache from that byte on (I3). If the seed "
                f"legitimately changed, that is a compaction -- call reset() first.")
        self._last[worker_id] = prompt

    def reset(self, worker_id: str) -> None:
        """A worker was compacted: its new seed is allowed to differ (I4)."""
        self._last.pop(worker_id, None)


def _first_divergence(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))
