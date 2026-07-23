#!/usr/bin/env python3
"""m3_compaction.py -- M3 acceptance: a worker crosses its budget and is reset, not edited.

Drives one worker past compact_at with a fact planted in its very first turn, then asks
for that fact after the reset. The old context is gone by then, so the only path for the
answer is the summary embedded in the new seed -- which makes this a test of the
summarizer as much as of the plumbing.

Checks:
  threshold    compaction fires, transcript is cleared, worker returns to COLD
  survival     the fact planted before the reset is still recoverable after it
  shrink       the new seed is far smaller than the context it replaced
  re-warm      the fresh seed prefills once, then turns are suffix-only again
  cleanup      the superseded warm blob is deleted from --slot-save-path

Note this exercises the LOCAL summarizer -- Bonsai summarizing itself, which docs/PLAN.md
open question 1 flags as the discouraged option (small low-bit models fail at memory
curation silently). Running it is how we find out how bad that actually is here.

    python experiments/m3_compaction.py [--budget 3000]

Writes m3-results.md next to this file. Requires a running server.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from hotshuttle.core.client import Llama                    # noqa: E402
from hotshuttle.core.orchestrator import Orchestrator       # noqa: E402
from hotshuttle.core.pool import SlotPool                   # noqa: E402
from hotshuttle.core.worker import Role, Task               # noqa: E402
from hotshuttle.profiles.bonsai import BonsaiProfile        # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "m3-results.md"
FACT = "The primary coolant valve is part number XR-3390."


async def run(budget: int) -> dict:
    profile = BonsaiProfile()
    llama = Llama(profile.server_url)
    if not await llama.healthy():
        raise SystemExit(f"server not up at {profile.server_url} -- bash bonsai.sh start")

    pool = SlotPool(llama, n_slots=1, slot_dir=profile.slot_save_path)
    orc = Orchestrator(llama, pool, profile)          # None -> local summarizer
    role = Role(name="engineer", system="You are a maintenance log worker. Answer briefly.",
                ctx_budget=budget, max_out=96, compact_at=0.8)
    w = orc.spawn("engineer", role)
    other = orc.spawn("filler", role)                 # forces an eviction -> a warm blob

    r: dict = {"budget": budget, "turns": []}

    # Turn 1 plants the fact. Everything after it is padding to burn context.
    await orc.dispatch(Task("engineer", f"Record this and confirm: {FACT}"))
    await orc.dispatch(Task("filler", "Say ok."))     # evict engineer -> blob on disk
    blob_path = pathlib.Path(profile.slot_save_path) / (w.kv_file or "")
    r["blob_before"] = str(blob_path) if w.kv_file else None
    r["blob_existed"] = blob_path.exists() if w.kv_file else False

    padding = ("Log entry {i}: routine inspection completed, no anomalies, "
               "torque values within tolerance, seals intact. ")
    turn = 1
    while w.compactions == 0 and turn < 20:
        turn += 1
        resp = await orc.dispatch(Task(
            "engineer", "Acknowledge this log batch in one word.",
            attach=tuple(padding.format(i=turn * 10 + j) for j in range(12))))
        r["turns"].append({"turn": turn, "evaluated": resp.prompt_n, "reused": resp.cache_n,
                           "n_ctx_used": w.n_ctx_used, "compactions": w.compactions})
        print(f"  turn {turn}: n_ctx_used={w.n_ctx_used:<6} evaluated={resp.prompt_n:<6} "
              f"reused={resp.cache_n:<6} compactions={w.compactions}")

    r["compacted"] = w.compactions == 1
    r["seed_after"] = w.seed
    r["transcript_cleared"] = w.transcript == ""
    r["state_after"] = w.state
    r["blob_deleted"] = (not blob_path.exists()) if r["blob_existed"] else None

    if not r["compacted"]:
        r["verdicts"] = {"compaction fired": False}
        return r

    # The old context is gone. Only the summary in the new seed can carry the fact.
    probe = await orc.dispatch(Task("engineer", "What is the coolant valve part number?"),
                               auto_compact=False)
    r["fresh_prefill_tokens"] = probe.prompt_tokens
    r["fresh_cache_n"] = probe.cache_n
    r["probe_reply"] = probe.content.strip()[:200]
    r["fact_survived"] = "XR-3390" in probe.content or "XR-3390" in w.seed
    r["fact_in_summary"] = "XR-3390" in w.seed
    print(f"\n  after reset: seed={len(w.seed)} chars, prefill={probe.prompt_tokens} tok")
    print(f"  fact in summary: {r['fact_in_summary']}   reply: {r['probe_reply']!r}")

    nxt = await orc.dispatch(Task("engineer", "Reply with the word done."),
                             auto_compact=False)
    r["next_evaluated"] = nxt.prompt_n
    r["next_reused"] = nxt.cache_n
    print(f"  next turn:  evaluated={nxt.prompt_n} reused={nxt.cache_n}")

    peak = max((t["n_ctx_used"] for t in r["turns"]), default=0)
    r["peak_n_ctx_used"] = peak
    r["verdicts"] = {
        "compaction fired at the threshold": r["compacted"],
        "transcript cleared, worker COLD": r["transcript_cleared"] and r["state_after"] == "COLD",
        "planted fact survived the reset": bool(r["fact_survived"]),
        "new seed much smaller than old context": r["fresh_prefill_tokens"] < peak / 2,
        "fresh seed prefilled once (no cache)": r["fresh_cache_n"] == 0,
        "turns are suffix-only again": r["next_evaluated"] <= 64 and r["next_reused"] > 0,
        "superseded blob deleted": r["blob_deleted"] is not False,
    }
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=3000, help="worker ctx_budget in tokens")
    args = ap.parse_args()

    print(f"M3: one worker, ctx_budget={args.budget}, compact_at=0.8")
    r = asyncio.run(run(args.budget))
    print()
    for k, ok in r.get("verdicts", {}).items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    write(r)
    print(f"\nwrote {OUT}")
    return 0 if r.get("verdicts") and all(r["verdicts"].values()) else 1


def write(r):
    v = r.get("verdicts", {})
    L = ["# M3 results -- compaction as reset", "",
         f"One worker, `ctx_budget={r['budget']}`, `compact_at=0.8`, local (Bonsai) summarizer.", "",
         "| turn | n_ctx_used | evaluated | reused |", "|---:|---:|---:|---:|"]
    for t in r["turns"]:
        L.append(f"| {t['turn']} | {t['n_ctx_used']} | {t['evaluated']} | {t['reused']} |")
    L += ["", f"Peak context before reset: **{r.get('peak_n_ctx_used')}** tokens.",
          f"First turn on the new seed prefilled **{r.get('fresh_prefill_tokens')}** tokens "
          f"(cache_n={r.get('fresh_cache_n')}); the turn after evaluated "
          f"**{r.get('next_evaluated')}** and reused {r.get('next_reused')}.", "",
          "## The summary that replaced the context", "", "```",
          r.get("seed_after", "")[:1500], "```", "",
          f"Probe reply: `{r.get('probe_reply', '')}`", "", "## Acceptance", ""]
    L += [f"- [{'x' if ok else ' '}] {k}" for k, ok in v.items()]
    L += ["", "```json", json.dumps({k: x for k, x in r.items() if k != "seed_after"},
                                    indent=2), "```"]
    OUT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
