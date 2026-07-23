#!/usr/bin/env python3
"""m1_save_restore.py -- M1: does slot save/restore actually skip re-prefill?

This is the premise of the whole hotshuttle design (docs/PLAN.md M1). If restoring
a saved slot does NOT let `cache_prompt` skip the already-seen prefix, then every
worker switch costs a full prefill of that worker's context instead of a ~10 ms blob
copy, and the cost model in docs/PLAN.md changes (though the architecture survives --
see the documented fallback in the plan).

The risk is specific to hybrid/recurrent models: upstream llama.cpp #22384 (hybrid
checkpoint restore) and #19794 (hybrid prompt cache forcing full re-processing) show
exactly this failing. Fixes landed on mainline; the PrismML fork's base revision is
unknown, so it is measured here rather than assumed.

The experiment, per worker A:

  1. cold prefill of a ~2000-token prompt into slot 0     -> expect timings.prompt_n ~= all of it
  2. save the slot to a blob
  3. perturb: erase, and prefill a DIFFERENT worker B into the same slot
  4. restore A's blob
  5. re-dispatch A with [same prompt + its own reply + a short new turn]
                                                          -> expect timings.prompt_n ~= the new turn only

PASS  step 5 evaluates at most (new turn + 64) tokens AND A still recalls a fact
      planted in its seed (guards against a restore that "succeeds" with corrupt state).
FAIL  step 5 re-evaluates the whole prompt, or the planted fact is gone/garbled.

    python experiments/m1_save_restore.py             # current server config
    python experiments/m1_save_restore.py --restart   # also restore after a server restart
    python experiments/m1_save_restore.py --matrix    # q8_0 and q4_0, both legs

Writes m1-results.md next to this file.
"""
import argparse, json, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from m0_verify import _req, healthy, server, SLOT_DIR, URL  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "m1-results.md"

# Template bytes captured from the server's own Jinja render in M0
# (experiments/m0_template_ground_truth.txt). The empty <think></think> pair is how
# enable_thinking=false renders; without it Bonsai burns its budget reasoning.
ASSISTANT_OPEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
SECRET = "HOTSHUTTLE-7742"

# ~2000 tokens of ballast so a re-prefill is unmistakable next to a ~40-token suffix.
FILLER = ("Background note {i}: the shuttle bay inventory lists spare couplings, "
          "torque plates, and a calibration jig of unknown provenance. ")


def seed(system, n_filler=95):
    body = system + "\n\n" + "".join(FILLER.format(i=i) for i in range(n_filler))
    return f"<|im_start|>system\n{body}<|im_end|>\n"


def user_turn(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n" + ASSISTANT_OPEN


def n_tokens(text):
    return len(_req("POST", "/tokenize", {"content": text})["tokens"])


def complete(prompt, n_predict=32, slot=0):
    t0 = time.perf_counter()
    r = _req("POST", "/completion", {"prompt": prompt, "id_slot": slot,
                                     "n_predict": n_predict, "cache_prompt": True,
                                     "temperature": 0.0}, timeout=900)
    r["_wall_ms"] = (time.perf_counter() - t0) * 1000
    return r


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


def leg(label, restart_cfg=None, save_restore=True, perturb=True):
    """One save/restore round trip. restart_cfg=(ctx, kv) also bounces the server.

    save_restore=False, perturb=False is the CONTROL: same prompts, same append-only
    continuation, but the slot is never touched in between. It isolates "save/restore
    does not preserve the cache" from "this fork's prefix caching never reuses anything",
    which look identical at step 5 but mean completely different things.
    """
    print(f"\n--- {label} ---")
    res = {"label": label}

    # 1. cold prefill of worker A
    _req("POST", "/slots/0?action=erase", {})
    sysA = (f"You are worker A. Remember this project codename exactly: {SECRET}. "
            f"Answer in one short sentence.")
    p1 = seed(sysA) + user_turn("Name one item in the shuttle bay inventory.")
    n_p1 = n_tokens(p1)
    r1 = complete(p1)
    res["cold_prompt_tokens"] = n_p1
    res["cold_evaluated"] = r1["timings"]["prompt_n"]
    res["cold_cached"] = r1["timings"]["cache_n"]
    print(f"  1. cold prefill      : prompt={n_p1} tok, evaluated={res['cold_evaluated']}, "
          f"reused={res['cold_cached']}")

    if save_restore:
        # 2. save
        res["save_ms"] = timed(lambda: _req("POST", "/slots/0?action=save",
                                            {"filename": "m1_A.bin"}))
        blob = SLOT_DIR / "m1_A.bin"
        res["blob_mib"] = blob.stat().st_size / 2**20 if blob.exists() else 0
        print(f"  2. save              : {res['save_ms']:.0f} ms, {res['blob_mib']:.1f} MiB")
        if not res["blob_mib"]:
            res["verdict"] = "FAIL (no blob written)"
            return res

    if perturb:
        # 3. evict A the way the SlotPool will: another worker takes the slot
        _req("POST", "/slots/0?action=erase", {})
        pB = seed("You are worker B. You know nothing about any codename.") + \
            user_turn("State your worker letter.")
        complete(pB, n_predict=8)
        print(f"  3. perturb           : worker B prefilled into slot 0")

    if restart_cfg:
        ctx, kv = restart_cfg
        _, used, err = server(ctx, kv)
        if used is None:
            res["verdict"] = f"FAIL (server did not restart: {err})"
            return res
        print(f"  3b. server restarted : {used} MiB")

    if save_restore:
        # 4. restore A
        res["restore_ms"] = timed(lambda: _req("POST", "/slots/0?action=restore",
                                               {"filename": "m1_A.bin"}))
        print(f"  4. restore           : {res['restore_ms']:.0f} ms")

    # 5. A continues: identical bytes so far, plus one short new turn
    prior = p1 + r1["content"] + "<|im_end|>\n"
    p2 = prior + user_turn("What is the project codename? Reply with just the codename.")
    suffix = n_tokens(p2) - n_tokens(prior)
    r2 = complete(p2)
    res["suffix_tokens"] = suffix
    # timings.prompt_n is the tokens actually evaluated; timings.cache_n is the tokens
    # reused from the slot's cache. NOT tokens_evaluated -- that is the full prompt
    # length regardless of what was cached, so asserting on it always passes.
    res["warm_evaluated"] = r2["timings"]["prompt_n"]
    res["warm_cached"] = r2["timings"]["cache_n"]
    res["reply"] = r2["content"].strip()[:80]
    res["recalled"] = SECRET in r2["content"]
    budget = suffix + 64
    res["skipped_reprefill"] = res["warm_evaluated"] <= budget
    pct = 100 * res["warm_evaluated"] / n_tokens(p2)
    print(f"  5. warm continuation : suffix={suffix} tok, evaluated={res['warm_evaluated']}, "
          f"reused={res['warm_cached']} (budget {budget}, {pct:.1f}% of the full prompt)")
    print(f"     recalled {SECRET}: {res['recalled']}   reply={res['reply']!r}")

    res["verdict"] = ("PASS" if res["skipped_reprefill"] and res["recalled"]
                      else "FAIL (re-prefilled)" if not res["skipped_reprefill"]
                      else "FAIL (state restored but the planted fact was lost)")
    print(f"  => {res['verdict']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true", help="also test restore after a server restart")
    ap.add_argument("--matrix", action="store_true", help="run q8_0 and q4_0, both legs")
    ap.add_argument("--ctx", type=int, default=12288)
    args = ap.parse_args()

    rows = []
    if args.matrix:
        for kv in ("q8_0", "q4_0"):
            _, used, err = server(args.ctx, kv)
            if used is None:
                print(f"server did not start at {args.ctx}/{kv}: {err}")
                continue
            rows.append(leg(f"{args.ctx} / {kv} / round-trip"))
            rows.append(leg(f"{args.ctx} / {kv} / after server restart",
                            restart_cfg=(args.ctx, kv)))
    else:
        if not healthy():
            print(f"server not up at {URL} -- run: bash bonsai.sh start")
            return 1
        rows.append(leg("CONTROL: no save, no eviction", save_restore=False, perturb=False))
        rows.append(leg("current server config / round-trip"))
        if args.restart:
            rows.append(leg("current server config / after server restart",
                            restart_cfg=(args.ctx, "q8_0")))

    write_results(rows)
    print(f"\nwrote {OUT}")
    return 0 if rows and all(r.get("verdict") == "PASS" for r in rows) else 1


def write_results(rows):
    L = ["# M1 results -- does slot save/restore skip re-prefill?", "",
         "Generated by `experiments/m1_save_restore.py`. PASS means a restored worker's",
         "next turn evaluated only its new suffix (+64 tokens of template slop) *and* still",
         "recalled a fact planted in its seed before the save.", "",
         "Metric is `timings.prompt_n` (tokens actually evaluated) and `timings.cache_n`",
         "(tokens reused). **Not** `tokens_evaluated`, which is the full prompt length",
         "whether or not anything was cached -- asserting on it always passes.", "",
         "| leg | cold prefill | suffix | evaluated warm | reused warm | blob | save ms | restore ms | recall | verdict |",
         "|---|---:|---:|---:|---:|---:|---:|---:|:-:|---|"]
    for r in rows:
        L.append("| {label} | {cold_evaluated}/{cold_prompt_tokens} tok | {suffix_tokens} | "
                 "**{warm_evaluated}** | {warm_cached} | {blob_mib:.0f} MiB | {save_ms:.0f} | "
                 "{restore_ms:.0f} | {rec} | {verdict} |".format(
                     rec="yes" if r.get("recalled") else "no",
                     **{k: r.get(k, 0) for k in
                        ("label", "cold_evaluated", "cold_prompt_tokens", "suffix_tokens",
                         "warm_evaluated", "warm_cached", "blob_mib", "save_ms",
                         "restore_ms", "verdict")}))
    L += ["", "```json", json.dumps(rows, indent=2), "```"]
    OUT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
