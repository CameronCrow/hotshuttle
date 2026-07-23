#!/usr/bin/env python3
"""hotshuttle MCP server -- Bonsai and its worker pool, for any MCP client.

Two surfaces, because they answer different questions:

  bonsai_*   one-shot calls to the local model. Stateless, no slot bookkeeping. This is
             issue #6: it exists so a caller gets non-empty text without having to know
             that Bonsai burns its entire budget on <think> unless thinking is disabled.
  worker_*   persistent workers sharing one GPU slot. Each keeps its own context between
             calls even though only one fits in VRAM at a time; the pool saves and
             restores their state around every switch. This is what hotshuttle is for.

Worker state lives in this process, so it persists across tool calls within a session and
is gone when the server stops. That is deliberate -- the saved blobs are a cache, not a
database, and a worker's value is its warm context, which the model would have to be
re-fed anyway after a restart.

Registered by the plugin's .mcp.json; run standalone with:
    python mcp_server/hotshuttle_mcp.py
"""
from __future__ import annotations

import asyncio
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP                      # noqa: E402

from hotshuttle.core import manifest                        # noqa: E402
from hotshuttle.core.client import Llama                    # noqa: E402
from hotshuttle.core.orchestrator import Orchestrator       # noqa: E402
from hotshuttle.core.pool import SlotPool                   # noqa: E402
from hotshuttle.core.worker import Role, Task               # noqa: E402
from hotshuttle.profiles.bonsai import BonsaiProfile        # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
mcp = FastMCP("hotshuttle")

_state: dict = {}


def _orc() -> Orchestrator:
    """Build the orchestrator lazily, so importing this module never needs a live server."""
    if "orc" not in _state:
        try:
            m = manifest.load(REPO / "workers.yaml")
            profile = BonsaiProfile(**m.model_kwargs())
            roles = m.roles
        except manifest.ManifestError:
            profile = BonsaiProfile()               # no manifest? measured defaults still work
            roles = {}
        llama = Llama(profile.server_url)
        pool = SlotPool(llama, n_slots=profile.n_slots, slot_dir=profile.slot_save_path)
        _state.update(orc=Orchestrator(llama, pool, profile), roles=roles,
                      profile=profile, llama=llama)
    return _state["orc"]


def _bash() -> str:
    # Never a bare "bash": Windows searches System32 before PATH and finds WSL's bash,
    # which cannot see C:/... paths and fails with a confusing "No such file or directory".
    return shutil.which("bash") or "bash"


# --- one-shot surface (issue #6) ---------------------------------------------

@mcp.tool()
async def bonsai_status() -> str:
    """Is the local Bonsai server up? Returns its endpoint and current config."""
    orc = _orc()
    if not await orc.llama.healthy():
        return f"down (expected at {orc.profile.server_url}) -- call bonsai_start"
    slots = await orc.llama.slots()
    p = orc.profile
    return (f"up: {p.server_url} | model {p.name} | {p.n_slots} slot(s) "
            f"@ {p.ctx_per_slot} ctx, KV {p.kv_quant} | "
            f"busy={[s.get('is_processing') for s in slots]}")


@mcp.tool()
async def bonsai_start() -> str:
    """Start the local Bonsai server if it is not already running. Blocks until ready
    (model load takes roughly 30-60s cold)."""
    orc = _orc()
    if await orc.llama.healthy():
        return f"already running: {orc.profile.server_url}"
    r = await asyncio.to_thread(
        subprocess.run, [_bash(), (REPO / "bonsai.sh").as_posix(), "start"],
        capture_output=True, text=True, timeout=600)
    return (r.stdout + r.stderr).strip()[-500:] or "started"


@mcp.tool()
async def bonsai_chat(prompt: str, system: str = "", max_tokens: int = 1024,
                      think: bool = False) -> str:
    """One-shot call to the local Bonsai 27B model. Stateless -- no memory between calls.

    Thinking is disabled unless you ask for it: Bonsai is a reasoning model and will
    otherwise spend its whole token budget inside <think> and return empty content.

    Use this for scoped, self-contained work: drafting, summarizing, extraction,
    classification. For anything that needs memory across several calls, use the worker_*
    tools instead. Bonsai is 2-bit quantized -- treat output as a serviceable draft and
    verify any factual claim.
    """
    orc = _orc()
    if not await orc.llama.healthy():
        return "Bonsai is not running -- call bonsai_start first."
    tmpl = orc.profile.template
    text = tmpl.system(system or "You are a helpful assistant.") + tmpl.user(prompt)
    text += "<|im_start|>assistant\n<think>\n" if think else tmpl.assistant_open()
    resp = await orc.llama.complete(text, id_slot=0, n_predict=max_tokens,
                                    cache_prompt=True, **orc.profile.sampling_defaults)
    return resp.content.strip() or "(empty response -- try think=false)"


# --- worker surface ----------------------------------------------------------

@mcp.tool()
async def worker_spawn(worker_id: str, system: str, role: str = "",
                       ctx_budget: int = 0, max_out: int = 1024) -> str:
    """Create a persistent worker that keeps its own context across calls.

    Spawn as many as you like -- they share the GPU slot and are paged in and out
    automatically, so more workers than slots is the normal case, not a problem. Give each
    a narrow `system` brief; scoped workers are what a 2-bit model is good at.

    `role` optionally names a role from workers.yaml to inherit tools and budgets from.
    """
    orc = _orc()
    if worker_id in orc.workers:
        return f"worker {worker_id!r} already exists ({orc.workers[worker_id].state})"
    base = _state["roles"].get(role)
    r = Role(name=role or worker_id, system=system,
             ctx_budget=ctx_budget or (base.ctx_budget if base else orc.profile.ctx_per_slot),
             tools=base.tools if base else (),
             max_out=max_out or (base.max_out if base else 1024),
             compact_at=base.compact_at if base else 0.8)
    orc.spawn(worker_id, r)
    return f"spawned {worker_id!r} (ctx_budget {r.ctx_budget}, max_out {r.max_out})"


@mcp.tool()
async def worker_ask(worker_id: str, instruction: str, attach: str = "") -> str:
    """Give a worker one turn. It remembers everything from its previous turns.

    `attach` is material for THIS turn only -- a file, a snippet, a result. Only send a
    given piece of context once: the worker keeps it, and re-sending it both wastes its
    budget and breaks the append-only prompt that makes the caching work.

    If the worker's context outgrows its budget it is compacted automatically: retired and
    re-seeded with a summary. That summary is written by Bonsai itself, which is the weak
    link -- keep workers scoped enough that they rarely reach it.
    """
    orc = _orc()
    if worker_id not in orc.workers:
        return f"no worker {worker_id!r} -- call worker_spawn first"
    if not await orc.llama.healthy():
        return "Bonsai is not running -- call bonsai_start first."
    w = orc.workers[worker_id]
    before = w.compactions
    resp = await orc.dispatch(Task(worker_id, instruction,
                                   attach=(attach,) if attach else ()))
    note = ""
    if w.compactions > before:
        note = "\n\n[worker was compacted after this turn: its context was summarized " \
               "and reset. Re-send anything it must not forget.]"
    return (resp.content.strip() or "(empty response)") + note


@mcp.tool()
async def worker_list() -> str:
    """Show every worker: its state, context size, and how the slot pool is doing."""
    orc = _orc()
    if not orc.workers:
        return "no workers yet -- call worker_spawn"
    lines = [f"{'worker':<16} {'state':<6} {'ctx':>7} {'compactions':>12}",
             "-" * 45]
    for wid, w in orc.workers.items():
        lines.append(f"{wid:<16} {w.state:<6} {w.n_ctx_used:>7} {w.compactions:>12}")
    p = orc.pool.snapshot()
    lines += ["", f"slots: {p['resident'] or 'none resident'}  free={p['free']}",
              f"paging: {p['stats']}",
              f"re-prefilled {orc.reprefill_ratio:.1%} of prompt tokens this session "
              f"(lower is better; the rest came from cache)"]
    return "\n".join(lines)


@mcp.tool()
async def worker_retire(worker_id: str) -> str:
    """Delete a worker and free its saved state. Use when a task is finished."""
    orc = _orc()
    if worker_id not in orc.workers:
        return f"no worker {worker_id!r}"
    await orc.pool.retire(orc.workers.pop(worker_id))
    orc.guard.reset(worker_id)
    return f"retired {worker_id!r}"


if __name__ == "__main__":
    mcp.run()
