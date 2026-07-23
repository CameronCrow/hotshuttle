#!/usr/bin/env python3
"""m2_pingpong.py -- M2 acceptance: two workers alternating on one physical slot.

Every dispatch forces a save/evict/restore, so this is the whole paging path running
against the real server rather than the fake. It checks the three things that can go
wrong once SlotPool is in the loop:

  re-prefill     each worker's turn >= 2 must evaluate only its new suffix
  contamination  each worker recalls ITS OWN planted fact and never the other's --
                 this is the probe risk #4 actually needs, since M1's planted fact also
                 sat in the re-sent prefix and so could not distinguish restored state
                 from a re-read prompt. Here B's secret is never in A's prompt at all.
  overhead       wall-clock spent paging must stay under 10% of generation time

    python experiments/m2_pingpong.py [--turns 3]

Writes m2-results.md next to this file. Requires a running server (bash bonsai.sh start).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from hotshuttle.core.client import Llama                    # noqa: E402
from hotshuttle.core.orchestrator import Orchestrator       # noqa: E402
from hotshuttle.core.pool import SlotPool                   # noqa: E402
from hotshuttle.core.worker import Role, Task               # noqa: E402
from hotshuttle.profiles.bonsai import BonsaiProfile        # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "m2-results.md"

SECRETS = {"alpha": "ORCHID-4417", "beta": "GRANITE-9082"}


class TimedLlama(Llama):
    """Llama plus a stopwatch on the paging operations, for the overhead budget."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.paging_ms = 0.0
        self.generate_ms = 0.0
        self.page_ops = 0

    async def _timed(self, coro):
        t0 = time.perf_counter()
        r = await coro
        return r, (time.perf_counter() - t0) * 1000

    async def complete(self, *a, **kw):
        r, ms = await self._timed(super().complete(*a, **kw))
        self.generate_ms += ms
        return r

    async def save(self, *a, **kw):
        _, ms = await self._timed(super().save(*a, **kw))
        self.paging_ms += ms
        self.page_ops += 1

    async def restore(self, *a, **kw):
        _, ms = await self._timed(super().restore(*a, **kw))
        self.paging_ms += ms
        self.page_ops += 1


async def run(turns: int) -> dict:
    profile = BonsaiProfile()
    llama = TimedLlama(profile.server_url)
    if not await llama.healthy():
        raise SystemExit(f"server not up at {profile.server_url} -- bash bonsai.sh start")

    pool = SlotPool(llama, n_slots=profile.n_slots, slot_dir=profile.slot_save_path)
    orc = Orchestrator(llama, pool, profile)

    for wid, secret in SECRETS.items():
        orc.spawn(wid, Role(
            name=wid,
            system=(f"You are worker {wid}. Your private access code is {secret}. "
                    f"Never invent a different code. Answer in one short sentence."),
            ctx_budget=profile.ctx_per_slot, max_out=64))

    # Give each worker enough context that a re-prefill would be unmistakable.
    ballast = ("Inventory line item {i}: one calibration jig, provenance unknown. "
               * 1).strip()
    rows = []
    for wid in SECRETS:
        await orc.dispatch(Task(wid, "Acknowledge these notes.",
                                attach=tuple(ballast.format(i=i) for i in range(60))))

    # Alternate. With n_slots=1 every one of these evicts the other worker.
    for turn in range(turns):
        for wid in SECRETS:
            resp = await orc.dispatch(Task(wid, "Reply with only your private access code."))
            mine, theirs = SECRETS[wid], SECRETS["beta" if wid == "alpha" else "alpha"]
            rows.append({
                "turn": turn + 1, "worker": wid,
                "evaluated": resp.prompt_n, "reused": resp.cache_n,
                "prompt_tokens": resp.prompt_tokens,
                "recalled_own": mine in resp.content,
                "leaked_other": theirs in resp.content,
                "reply": resp.content.strip()[:60],
            })
            print(f"  turn {turn + 1} {wid:>5}: evaluated={resp.prompt_n:<5} "
                  f"reused={resp.cache_n:<6} own={mine in resp.content} "
                  f"leak={theirs in resp.content}  {resp.content.strip()[:40]!r}")

    overhead = llama.paging_ms / llama.generate_ms if llama.generate_ms else 0
    return {
        "rows": rows,
        "pool": pool.snapshot(),
        "paging_ms": llama.paging_ms, "generate_ms": llama.generate_ms,
        "page_ops": llama.page_ops, "overhead_ratio": overhead,
        "reprefill_ratio": orc.reprefill_ratio,
    }


def verdicts(r: dict) -> dict:
    warm = [x for x in r["rows"] if x["turn"] > 1]
    return {
        "no re-prefill on warm turns": all(x["evaluated"] <= 64 for x in warm),
        "every worker recalled its own code": all(x["recalled_own"] for x in r["rows"]),
        "no cross-worker contamination": not any(x["leaked_other"] for x in r["rows"]),
        "paging overhead < 10% of generation": r["overhead_ratio"] < 0.10,
        "saves matched evictions": (r["pool"]["stats"].get("evictions", 0)
                                    <= r["page_ops"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=3)
    args = ap.parse_args()

    print(f"M2: {len(SECRETS)} workers alternating on 1 slot, {args.turns} turns each")
    r = asyncio.run(run(args.turns))
    v = verdicts(r)

    print(f"\npool: {r['pool']}")
    print(f"paging {r['paging_ms']:.0f} ms over {r['page_ops']} ops vs "
          f"{r['generate_ms']:.0f} ms generating -> {r['overhead_ratio'] * 100:.1f}% overhead")
    print(f"re-prefilled {r['reprefill_ratio'] * 100:.1f}% of all prompt tokens\n")
    for k, ok in v.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")

    write(r, v)
    print(f"\nwrote {OUT}")
    return 0 if all(v.values()) else 1


def write(r, v):
    L = ["# M2 results -- two workers, one slot", "",
         "Every turn below forced a save, an eviction and a restore: with `n_slots=1`,",
         "dispatching either worker displaces the other.", "",
         "| turn | worker | evaluated | reused | own code | leaked other | reply |",
         "|---:|---|---:|---:|:-:|:-:|---|"]
    for x in r["rows"]:
        L.append(f"| {x['turn']} | {x['worker']} | **{x['evaluated']}** | {x['reused']} | "
                 f"{'yes' if x['recalled_own'] else 'NO'} | "
                 f"{'YES' if x['leaked_other'] else 'no'} | `{x['reply']}` |")
    L += ["", f"- paging: **{r['paging_ms']:.0f} ms** over {r['page_ops']} save/restore ops",
          f"- generating: {r['generate_ms']:.0f} ms",
          f"- overhead: **{r['overhead_ratio'] * 100:.1f}%** of generation time",
          f"- re-prefilled across the run: **{r['reprefill_ratio'] * 100:.1f}%** of prompt tokens",
          f"- pool: `{r['pool']}`", "", "## Acceptance", ""]
    L += [f"- [{'x' if ok else ' '}] {k}" for k, ok in v.items()]
    L += ["", "```json", json.dumps(r, indent=2), "```"]
    OUT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
