#!/usr/bin/env python3
"""m5_fanout.py -- M5 acceptance: the whole thing, unattended.

Four logical workers, one physical slot, driven from workers.yaml through the scheduler.
Each worker is handed a different real source file from this repo, then asked a series of
scoped questions about it, then a cross-referencing question. With one slot, all four
workers cannot be resident, so the pool pages them continuously.

The number that matters is the re-prefill ratio: the share of prompt tokens the server had
to re-evaluate rather than reuse. Cold seeds are unavoidable, so this only falls below the
15% bar if paging is genuinely carrying the contexts between turns.

    python experiments/m5_fanout.py [--turns 8]

Writes m5-results.md next to this file. Requires a running server.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from hotshuttle.core import manifest                        # noqa: E402
from hotshuttle.core.client import Llama                    # noqa: E402
from hotshuttle.core.orchestrator import Orchestrator       # noqa: E402
from hotshuttle.core.pool import SlotPool                   # noqa: E402
from hotshuttle.core.scheduler import Scheduler             # noqa: E402
from hotshuttle.core.worker import Task                     # noqa: E402
from hotshuttle.profiles.bonsai import BonsaiProfile        # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).resolve().parent / "m5-results.md"

ASSIGNMENTS = {
    "pool": "hotshuttle/core/pool.py",
    "compiler": "hotshuttle/core/compiler.py",
    "scheduler": "hotshuttle/core/scheduler.py",
    "manifest": "hotshuttle/core/manifest.py",
}

QUESTIONS = [
    "In one sentence, what is this module responsible for?",
    "Name the main class or functions it defines.",
    "What is the single most important invariant or rule it enforces?",
    "What would break first if this module were removed?",
    "Does it import anything model-specific? Answer yes or no and name it.",
    "Which other module in this package does it depend on most?",
    "Name one edge case its code explicitly handles.",
    "Summarize its role in five words or fewer.",
]


async def run(turns: int) -> dict:
    m = manifest.load(REPO / "workers.yaml")
    profile = BonsaiProfile(**m.model_kwargs())
    role = m.roles["code-reader"]

    llama = Llama(profile.server_url)
    if not await llama.healthy():
        raise SystemExit(f"server not up at {profile.server_url} -- bash bonsai.sh start")

    pool = SlotPool(llama, n_slots=profile.n_slots, slot_dir=profile.slot_save_path)
    orc = Orchestrator(llama, pool, profile)

    # Each worker is seeded once with its own file, then never re-sent it: every later
    # turn relies on that context still being there, paged in from a blob.
    for wid, relpath in ASSIGNMENTS.items():
        orc.spawn(wid, role)
        source = (REPO / relpath).read_text(encoding="utf-8")
        await orc.dispatch(Task(wid, f"Here is `{relpath}`. Read it and reply READY.",
                                attach=(source,)))
        print(f"  seeded {wid:<10} <- {relpath} ({len(source)} chars)")

    seed_reprefill = orc.reprefilled
    seed_total = orc.total_prompt

    # Round-robin the questions so consecutive turns hit different workers, forcing an
    # eviction every time. This is the worst case for paging, deliberately.
    tasks = [Task(wid, q)
             for q in QUESTIONS[:turns]
             for wid in ASSIGNMENTS]
    s = Scheduler(orc, extra_loops=1)
    s.submit(*tasks)
    print(f"\n  running {len(tasks)} questions across {len(ASSIGNMENTS)} workers "
          f"on {profile.n_slots} slot ...")
    stats = await s.run()

    answers = {wid: orc.workers[wid].transcript[-400:] for wid in ASSIGNMENTS}
    return {
        "turns_per_worker": turns,
        "workers": len(ASSIGNMENTS),
        "dispatched": stats.dispatched + len(ASSIGNMENTS),
        "errors": stats.errors,
        "wall_s": stats.wall_ms / 1000,
        "seed_reprefill_ratio": seed_reprefill / seed_total if seed_total else 0,
        "reprefilled": orc.reprefilled,
        "total_prompt": orc.total_prompt,
        "reprefill_ratio": orc.reprefill_ratio,
        "pool": pool.snapshot(),
        "compactions": {wid: orc.workers[wid].compactions for wid in ASSIGNMENTS},
        "n_ctx_used": {wid: orc.workers[wid].n_ctx_used for wid in ASSIGNMENTS},
        "answers": answers,
    }


def verdicts(r: dict) -> dict:
    return {
        "demo completed unattended": r["errors"] == [] and r["dispatched"] > 0,
        "re-prefilled < 15% of prompt tokens": r["reprefill_ratio"] < 0.15,
        "every worker still has its context": all(v > 0 for v in r["n_ctx_used"].values()),
        "pool bookkeeping consistent": (
            len(r["pool"]["resident"]) <= 1
            and r["pool"]["stats"].get("evictions", 0) > 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=len(QUESTIONS),
                    help="questions per worker")
    args = ap.parse_args()

    print(f"M5: {len(ASSIGNMENTS)} workers, 1 slot, driven from workers.yaml")
    r = asyncio.run(run(min(args.turns, len(QUESTIONS))))
    v = verdicts(r)

    print(f"\n  {r['dispatched']} dispatches in {r['wall_s']:.0f}s")
    print(f"  re-prefilled {r['reprefilled']} of {r['total_prompt']} prompt tokens "
          f"= {r['reprefill_ratio'] * 100:.1f}%")
    print(f"  pool: {r['pool']}\n")
    for k, ok in v.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")

    write(r, v)
    print(f"\nwrote {OUT}")
    return 0 if all(v.values()) else 1


def write(r, v):
    L = ["# M5 results -- end-to-end fan-out", "",
         f"{r['workers']} workers, 1 physical slot, {r['turns_per_worker']} questions each,",
         "driven from `workers.yaml` through the scheduler. Consecutive turns deliberately",
         "target different workers, so the pool evicts and restores on nearly every dispatch.",
         "",
         f"- dispatches: **{r['dispatched']}** in {r['wall_s']:.0f}s",
         f"- re-prefilled: **{r['reprefilled']} / {r['total_prompt']} tokens "
         f"= {r['reprefill_ratio'] * 100:.1f}%** (bar: < 15%)",
         f"- of which the four unavoidable cold seeds account for "
         f"{r['seed_reprefill_ratio'] * 100:.0f}% of their own round",
         f"- pool: `{r['pool']}`",
         f"- compactions: `{r['compactions']}`", "",
         "## Where each worker ended up", ""]
    for wid, tail in r["answers"].items():
        L += [f"**{wid}** ({ASSIGNMENTS[wid]}), {r['n_ctx_used'][wid]} tokens resident:", "",
              "```", tail.strip()[-350:], "```", ""]
    L += ["## Acceptance", ""]
    L += [f"- [{'x' if ok else ' '}] {k}" for k, ok in v.items()]
    L += ["", "```json",
          json.dumps({k: x for k, x in r.items() if k != "answers"}, indent=2), "```"]
    OUT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
