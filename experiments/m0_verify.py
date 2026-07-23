#!/usr/bin/env python3
"""m0_verify.py -- M0 acceptance checks for the hotshuttle orchestration layer.

Answers four questions the plan (docs/PLAN.md M0) says must be answered with
measurements, not arithmetic:

  1. Does this llama.cpp fork actually expose the slot endpoints we build on?
     (GET /slots, POST /slots/{id}?action=save|restore -- the latter silently
     does nothing unless the server was started with --slot-save-path.)
  2. How much VRAM does a slot really cost? bench-results.md computed KV/token
     from all 64 layers; the Qwen3.6-27B config says only 16 of them carry a
     growing KV cache (the other 48 are DeltaNet with a fixed recurrent state).
     Those differ by ~4x and they disagree about whether 16K @ q8_0 fits.
  3. Therefore: does 16K @ q8_0 fit on this card, or must the default be lower?
  4. What exact bytes does the server's Jinja template emit with thinking
     disabled? The prompt compiler has to reproduce them byte-for-byte or the
     prefix cache silently misses on every turn.

Stdlib only, same as bonsai_client.py.

    python experiments/m0_verify.py             # checks 1 + 4 against a running server
    python experiments/m0_verify.py --measure   # checks 2 + 3: restarts the server per
                                                # config and diffs VRAM. Takes a few minutes
                                                # and stops whatever server is running.

Writes m0-results.md next to this file.
"""
import argparse, json, os, pathlib, shutil, subprocess, sys, time, urllib.error, urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
# Resolve bash by PATH rather than letting CreateProcess do it: Windows searches
# System32 before PATH, so a bare "bash" finds WSL's bash.exe, which cannot see
# C:/... paths and fails with a confusing "No such file or directory".
BASH = shutil.which("bash") or "bash"
BONSAI_DIR = pathlib.Path(os.environ.get("BONSAI_DIR", "C:/Users/Cameron/Projects/bonsai"))
SLOT_DIR = pathlib.Path(os.environ.get("BONSAI_SLOT_DIR", BONSAI_DIR / "bench-logs" / "slots"))
SERVER_LOG = BONSAI_DIR / "bench-logs" / "bonsai-server.log"
URL = os.environ.get("BONSAI_URL", "http://127.0.0.1:8080")
OUT = pathlib.Path(__file__).resolve().parent / "m0-results.md"

# Configs to measure, as (total_ctx, kv_quant). One slot each, so ctx == ctx_per_slot.
# 8K/q4_0 is today's shipping default (the baseline bench-results.md measured);
# 16K/q8_0 is the default docs/PLAN.md proposes; 16K/q4_0 is the documented fallback.
MATRIX = [(8192, "q4_0"), (16384, "q8_0"), (16384, "q4_0")]


# --- transport ---------------------------------------------------------------

def _req(method, path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(URL + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else {}


def healthy():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


# --- environment probes ------------------------------------------------------

def vram_used_mib():
    """MiB currently allocated on GPU 0, per the driver."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    return int(out.stdout.strip().splitlines()[0])


def read_log():
    """Server log, tolerating the UTF-16 the Windows console sometimes writes."""
    if not SERVER_LOG.exists():
        return ""
    raw = SERVER_LOG.read_bytes()
    if raw.count(b"\x00") > len(raw) // 4:      # UTF-16-ish
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def bonsai_sh(action, env):
    # .as_posix(): MSYS strips the backslashes out of a Windows path argument
    # ("C:\a\b.sh" -> "C:absh"), so pass forward slashes.
    return subprocess.run([BASH, (REPO / "bonsai.sh").as_posix(), action],
                          env=env, capture_output=True, text=True)


def server(ctx, kv, slots=1):
    """Restart the server at a given config; return once it answers /health."""
    env = {**os.environ, "BONSAI_CTX": str(ctx), "BONSAI_KV_QUANT": kv,
           "BONSAI_SLOTS": str(slots), "BONSAI_SLOT_DIR": str(SLOT_DIR)}
    r = bonsai_sh("stop", env)
    if r.returncode != 0:
        raise RuntimeError(f"bonsai.sh stop failed: {r.stderr.strip()[:300]}")
    for _ in range(20):                          # driver releases the allocation lazily
        if not healthy() and vram_used_mib() < 2500:
            break
        time.sleep(2)
    baseline = vram_used_mib()
    if baseline > 2500:
        # Measuring a delta against a baseline that still holds the last model would
        # report ~0 MiB per config, which reads as "free" instead of "not measured".
        raise RuntimeError(f"VRAM did not drop after stop ({baseline} MiB still used); "
                           f"another CUDA process is holding memory -- measurement aborted")
    r = bonsai_sh("start", env)
    if not healthy():
        return baseline, None, (r.stdout + r.stderr).strip()[-600:]
    time.sleep(3)                                # allocation settles after /health flips
    return baseline, vram_used_mib(), None


# --- check 1: slot endpoints exist and do something --------------------------

def check_slots():
    print("\n[1] slot endpoints")
    results = {}

    slots = _req("GET", "/slots")
    print(f"    GET /slots          -> {len(slots)} slot(s), "
          f"is_processing={[s.get('is_processing') for s in slots]}")
    results["get_slots"] = True

    # Put something in slot 0 so there is state worth saving.
    resp = _req("POST", "/completion", {
        "prompt": "The capital of France is", "id_slot": 0,
        "n_predict": 8, "cache_prompt": True, "temperature": 0.0})
    # timings.prompt_n = tokens actually evaluated. NOT tokens_evaluated, which is the
    # full prompt length whether or not it was cached (see docs/PLAN.md M1).
    print(f"    POST /completion    -> evaluated={resp['timings']['prompt_n']} "
          f"content={resp.get('content','')[:40]!r}")
    results["completion"] = True

    # A save with no --slot-save-path returns 200 and writes nothing, so the
    # only honest check is whether a file actually appeared on disk.
    before = set(SLOT_DIR.glob("*")) if SLOT_DIR.exists() else set()
    _req("POST", "/slots/0?action=save", {"filename": "m0probe.bin"})
    new = (set(SLOT_DIR.glob("*")) if SLOT_DIR.exists() else set()) - before
    blob = SLOT_DIR / "m0probe.bin"
    if blob.exists():
        mib = blob.stat().st_size / 2**20
        print(f"    POST ?action=save   -> wrote {blob.name} ({mib:.1f} MiB)")
        results["save_bytes"] = blob.stat().st_size
    else:
        print(f"    POST ?action=save   -> FAILED: no blob in {SLOT_DIR} (new files: {new})")
        results["save_bytes"] = 0

    _req("POST", "/slots/0?action=restore", {"filename": "m0probe.bin"})
    print("    POST ?action=restore-> ok")
    results["restore"] = True
    return results


# --- check 4: the exact template bytes the compiler must reproduce -----------

def check_template():
    print("\n[4] template ground truth (thinking disabled)")
    msgs = [{"role": "system", "content": "SYSTEM_MARKER"},
            {"role": "user", "content": "USER_MARKER"}]
    out = {}
    for think in (False, True):
        try:
            d = _req("POST", "/apply-template",
                     {"messages": msgs, "chat_template_kwargs": {"enable_thinking": think}})
            out["thinking_on" if think else "thinking_off"] = d.get("prompt", "")
        except urllib.error.HTTPError as e:
            print(f"    /apply-template unavailable ({e.code}); compiler must be verified "
                  f"against a logged render instead")
            return {}
    off, on = out["thinking_off"], out["thinking_on"]
    print(f"    enable_thinking=false -> {off!r}")
    print(f"    enable_thinking=true  -> {on!r}")
    print(f"    differ: {off != on}   (if False, the flag is a no-op in this template "
          f"and suppression must be done by the compiler itself)")
    (pathlib.Path(__file__).parent / "m0_template_ground_truth.txt").write_text(
        off, encoding="utf-8")
    return out


# --- checks 2+3: what a slot actually costs ---------------------------------

def throughput(n_prompt_tokens=1500, n_predict=48):
    """Prompt-ingest and decode rate. This, not memory.used, is the fit test.

    WDDM lets a process oversubscribe VRAM and silently pages the overflow to host
    memory, so a config that does not fit still reports ~the same memory.used as one
    that does -- it just gets slow. bench-results.md saw healthy decode at ~28 t/s and
    thrashing at ~3-20 t/s, so decode rate separates resident from spilled.
    """
    prompt = "The quick brown fox jumps over the lazy dog. " * (n_prompt_tokens // 10)
    r = _req("POST", "/completion", {"prompt": prompt, "id_slot": 0, "n_predict": n_predict,
                                     "cache_prompt": False, "temperature": 0.0}, timeout=900)
    t = r.get("timings", {})
    return t.get("prompt_per_second"), t.get("predicted_per_second"), r.get("tokens_evaluated")


def check_vram(matrix=MATRIX):
    print("\n[2/3] VRAM + throughput per config (server restarted for each; a few minutes)")
    rows = []
    for ctx, kv in matrix:
        baseline, used, err = server(ctx, kv)
        if used is None:
            print(f"    -c {ctx:<6} kv={kv}  -> DID NOT START (baseline {baseline} MiB)")
            rows.append((ctx, kv, baseline, None, None, None, err))
            continue
        pp, tg, nev = throughput()
        print(f"    -c {ctx:<6} kv={kv}  -> {used} MiB used (+{used - baseline} over a "
              f"{baseline} MiB desktop baseline) | ingest {pp:.0f} t/s, decode {tg:.1f} t/s "
              f"({nev} tok prompt)")
        rows.append((ctx, kv, baseline, used, used - baseline, (pp, tg, nev), ""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true",
                    help="restart the server per config and diff VRAM (stops any running server)")
    ap.add_argument("--configs", help='override the matrix, e.g. "8192:q8_0,12288:q4_0"')
    args = ap.parse_args()

    matrix = MATRIX
    if args.configs:
        matrix = [(int(c.split(":")[0]), c.split(":")[1]) for c in args.configs.split(",")]
    vram_rows = check_vram(matrix) if args.measure else []

    if not healthy():
        print(f"\nserver not up at {URL} -- start it with:\n"
              f"  BONSAI_CTX=16384 BONSAI_KV_QUANT=q8_0 bash bonsai.sh start")
        return 1
    slots = check_slots()
    tmpl = check_template()

    total = vram_used_mib()
    print(f"\nGPU now: {total} MiB used")
    write_results(vram_rows, slots, tmpl, total)
    print(f"wrote {OUT}")
    return 0 if slots.get("save_bytes") else 1


def write_results(vram_rows, slots, tmpl, total):
    L = ["# M0 results -- measured, not computed", "",
         f"Generated by `experiments/m0_verify.py`. GPU at time of run: {total} MiB used.", ""]
    if vram_rows:
        L += ["## VRAM + throughput per config", "",
              "Decode rate is the fit test, not `memory.used`: WDDM lets the process",
              "oversubscribe and pages the overflow to host memory, so an over-budget",
              "config reports similar VRAM and simply runs slow.", "",
              "| total ctx | KV quant | desktop baseline | with server | server cost | ingest t/s | decode t/s |",
              "|---:|---|---:|---:|---:|---:|---:|"]
        for ctx, kv, base, used, delta, perf, note in vram_rows:
            u = f"{used} MiB" if used else "**did not start**"
            d = f"{delta} MiB" if delta is not None else "-"
            pp = f"{perf[0]:.0f}" if perf else "-"
            tg = f"{perf[1]:.1f}" if perf else "-"
            L.append(f"| {ctx} | {kv} | {base} MiB | {u} | {d} | {pp} | {tg} |")
        L.append("")
    if slots:
        L += ["## Slot endpoints", "",
              f"- `GET /slots`: ok",
              f"- `POST /slots/0?action=save`: "
              f"{'wrote %.1f MiB' % (slots['save_bytes']/2**20) if slots.get('save_bytes') else 'NO BLOB WRITTEN'}",
              f"- `POST /slots/0?action=restore`: ok", ""]
    if tmpl:
        L += ["## Template ground truth", "",
              "Bytes the prompt compiler must reproduce exactly (`enable_thinking=false`):", "",
              "```", repr(tmpl.get("thinking_off", "")), "```", "",
              f"Differs from `enable_thinking=true`: {tmpl.get('thinking_off') != tmpl.get('thinking_on')}", ""]
    OUT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
