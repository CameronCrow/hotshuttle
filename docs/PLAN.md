# hotshuttle — Implementation Plan

**Status:** M0 ✅ · M1 ✅ · M2 next · 2026-07-22
**Design source of record:** `brainstorm-vault/Ideas/Moving Off Claude To Open Weights.md`
(private repo; sections "KV-cache tiering on one 8GB card", "Tuning the knobs", and
"Orchestration layer: how Fable actually drives the workers", all dated 2026-07-22).
This document is **self-contained** — every number and decision the plan depends on is
embedded here; the brainstorm doc is the provenance, not a prerequisite.

Companion: [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — component-level specs (client,
SlotPool, Orchestrator, scheduler, prompt compiler) and the correctness invariants.

---

## 1. Vision and scope

**hotshuttle** lets one capable orchestrator model (Claude Fable today; anything with an
HTTP client tomorrow) drive **many small worker agents off a single loaded copy of a local
model on one 8 GB GPU**, by paging each worker's context state (attention KV cache +
recurrent state) between VRAM (*hot*) and system RAM (*warm*) through a small number of
llama-server slots.

The framing that makes the whole design click: **register allocation over KV slots.**

- Physical llama-server slots (1–2 on 8 GB) are *registers*.
- Logical workers are *variables* — most of them spilled to RAM as saved state blobs.
- The orchestrator is the *allocator*: it decides which worker occupies a slot, when to
  spill (save) the current occupant, and when to fill (restore) the next.

llama-server supplies 100 % of the **mechanism** (prefix cache, slot save/restore,
generation); the orchestrator owns 100 % of the **policy** (which worker is hot, LRU
eviction, when to compact, what each worker sees) as ordinary Python. There is nothing to
build in the inference layer — the work is a cache policy plus a prompt contract.

### Scope: Bonsai-first, multi-model deferred

The first (only, for now) target is **PrismML Ternary-Bonsai-27B** served by PrismML's
llama.cpp fork — the model this repo already builds and serves. Generalizing to other
locally-hosted models is **explicitly deferred** — see
[issue #7](https://github.com/CameronCrow/hotshuttle/issues/7), which records the
analysis: the orchestration core is already ~80 % model-agnostic, and everything
Bonsai-specific is *data* (sizing constants + launch flags), not logic. The obligation
this plan takes on **now** is to keep the package boundary clean (`core/` vs
`profiles/bonsai/`, §5) so extraction later is a move, not surgery. We do **not** build a
second profile — don't abstract over a sample size of one.

**Orchestrator-agnostic:** hotshuttle never hardcodes Fable. The orchestrator side is
just an HTTP client plus a policy loop; anything that can call the hotshuttle API (or
import the Python package) can drive it. Related: issue #6 (MCP wrapper) is a natural
future front-end for this, and issue #5 (context paging) is subsumed by this plan.

### What Bonsai is (and why it changes the math)

Bonsai 27B is **not a dense transformer**. It is PrismML's ternary low-bit build of
**Qwen3.6-27B** — a *hybrid* architecture (numbers from the Qwen3.6-27B config):

| Property | Value |
|---|---|
| Weights | ternary, ~1.71 bpw true, **2.125 bpw deployed** (`Q2_0_g128`), ~7.2 GB file / ~6.7–7.1 GiB resident |
| Layers | **64** total, `full_attention_interval: 4` |
| Full-attention layers | **16** (24 Q heads / **4 KV heads**, head_dim **256**) — these grow a normal KV cache |
| Gated DeltaNet linear-attention layers | **48** (48 value heads, 128×128 state, conv kernel 4) — **fixed-size recurrent state, does not grow with context** |
| Native context | 262,144 tokens (nominal — see §7) |
| Serving | **requires PrismML's llama.cpp fork** (`Q2_0_g128` hybrid kernels); mainline cannot load it |

Three-quarters of the layers have *flat* memory. That inverts the usual sizing story —
see §6.

---

## 2. How hotshuttle layers on the existing repo

This repo (`bonsai-backend` → renamed `hotshuttle`) is currently a **serving layer**:
build scripts for the PrismML fork (`build.bat`/`build-cpu.bat`), a lifecycle launcher
(`bonsai.sh`, `serve.bat`, `run.bat`), a minimal client (`bonsai_client.py`), and
benchmarks (`bench.sh`, `bench-results.md`, `run_paper.sh`). None of that is discarded:

```
┌────────────────────────────────────────────────────────┐
│  Orchestrator (Fable / any agent)      ← policy        │
│    hotshuttle core: SlotPool, Orchestrator, scheduler  │  ← NEW (this plan)
├────────────────────────────────────────────────────────┤
│  llama-server HTTP API                 ← mechanism     │
│    /completion (id_slot, cache_prompt)                 │
│    POST /slots/{id}?action=save|restore|erase          │
├────────────────────────────────────────────────────────┤
│  Existing serving layer (this repo today)              │
│    bonsai.sh / serve.bat  → llama-server (PrismML fork)│  ← flag changes only (§3)
│    build.bat, bench.sh, bonsai_client.py               │
└────────────────────────────────────────────────────────┘
```

The existing `bonsai_client.py` (chat-completions convenience client, thinking-mode
suppression) stays as-is for one-shot callers. The orchestration layer gets its own thin
client because it needs the **raw `/completion` endpoint** with `id_slot` +
`cache_prompt`, which the OpenAI-style surface doesn't expose (see §4 and
ARCHITECTURE.md §3).

### Confirmed llama-server API surface (what we build on)

- `POST /completion` with `id_slot` (default `-1` = any idle slot) and `cache_prompt`
  (default `true`): the server **diffs the prompt against the slot's cached tokens and
  evaluates only the unseen suffix**. This is the within-life re-prefill avoidance.
- `POST /slots/{id_slot}?action=save|restore|erase`, body `{"filename": "..."}` —
  serializes/restores a slot's state to/from `--slot-save-path`. This is the
  across-spill re-prefill avoidance. Requires the server started with
  `--slot-save-path`.
- `GET /slots` — lists slot state (`is_processing`, cached token counts). Requires slot
  endpoints enabled.
- `--parallel N` — N slots sharing the one loaded model; `-c` is the **total** context,
  split across slots (`ctx_per_slot = c / parallel`).

---

## 3. Exact serving-layer flag changes

**What `bonsai.sh` and `serve.bat` pass today** (both identical in substance):

```
-ngl 99 -c 8192 --parallel 1 -fa 1
--cache-type-k q4_0 --cache-type-v q4_0
--jinja --temp 0.7 --top-p 0.95 --top-k 20
```

Assessment against what orchestration needs:

| Flag | Today | Needed | Verdict |
|---|---|---|---|
| flash-attention | `-fa 1` | on (required for quantized KV) | **keep** |
| `--parallel` | `1` | `1` (1 hot slot on 8 GB; §6) | **keep**, make env-tunable (`BONSAI_SLOTS`) for the 2-slot experiment |
| `-c` (total ctx) | `8192` | **`12288`** (12K per-slot default; §7 — revised down from 16K by M0 measurement) | **change** |
| KV quant | `q4_0` / `q4_0` | **`q8_0` / `q8_0`** default (§7) | **change** |
| `--slot-save-path` | **absent** | **required** — without it the save/restore endpoints do not function | **add** |
| slot endpoints | not enabled | `GET /slots` needed by the pool for `is_processing` checks | **add `--slots`** (and never `--no-slots`) |
| `-ngl` | `99` | `99` (full offload; drop only to free VRAM, then re-test `-fa` output) | **keep** |
| `--jinja`, sampling | present | harmless for chat-completions callers; orchestrator sets sampling per-request and doesn't use the chat template server-side | **keep** |

Concrete new invocation (in `bonsai.sh` terms; `serve.bat` mirrors it):

```bash
"$SERVER" -m "$MODEL" --alias bonsai-27b --host 127.0.0.1 --port "$PORT" \
  -ngl 99 -c 12288 --parallel 1 \
  -fa 1 --cache-type-k q8_0 --cache-type-v q8_0 \
  --slots --slot-save-path "$SLOT_DIR" \
  --jinja --temp 0.7 --top-p 0.95 --top-k 20
```

Long-input mode is `BONSAI_CTX=32768 BONSAI_KV_QUANT=q4_0` — 32K *forces* q4 KV on this
card and degrades long-retrieval; see §7 for why it is opt-in, not default. (The plan
originally specified a `BONSAI_LONGCTX=1` shorthand for this; it was dropped as an alias
for two env vars that already exist. The coupling it encoded — 32K forces q4 — is a
comment in `bonsai.sh` and `serve.bat`.)

**`$SLOT_DIR` on Windows:** `--slot-save-path` writes slot blobs to disk files and
Windows has no tmpfs. Route 1 (start here): a normal directory (e.g.
`bench-logs/slots/` or `%LOCALAPPDATA%`), letting the **OS page cache** serve
just-written files from RAM — fine for a working set of a few hundred MB. Route 2 (only
if disk churn actually shows up): hold blobs in process RAM via the C API
(`llama_state_seq_get_data`/`set_data`). Do not buy a RAM-disk driver up front.

**These changes are not made in this pass** — this document specifies them; M0 (§9)
lands them behind env toggles so the existing single-shot workflow (`bonsai` skill,
`run_paper.sh`) is not disturbed.

> **✅ Sizing contradiction RESOLVED in M0 (2026-07-22)** — the config-derived reading was
> right and `bench-results.md`'s 64-layer formula was wrong by ~3.4×; that file now carries
> a correction note. Measured by `experiments/m0_verify.py`:
>
> | quantity | plan predicted (config) | bench computed | **measured** |
> |---|---:|---:|---:|
> | KV/token @ q8_0 | ~34 KiB | 128 KiB | **34.1 KiB** |
> | KV/token @ q4_0 | ~18 KiB | 64 KiB | **18.1 KiB** |
> | recurrent state / slot | ~150 MiB | not modeled | **149.626 MiB** |
>
> Every config-derived number in this plan was right. KV/token is measured by differencing
> saved slot blobs (M1: a 2611-token worker saves at 236.5 MiB q8_0 / 195.7 MiB q4_0, less
> the 149.6 MiB floor). A blob for a 5-token prompt weighs 149.8 MiB — i.e. **the blob is
> mostly the recurrent state**, so per-worker spill cost is nearly flat in context length,
> as §6 assumed.
>
> **But the operative constraint turned out to be neither formula.** 16K @ q8_0 does fit
> and runs at full speed (29.8 t/s) — *when the Windows desktop holds ≲680 MiB of VRAM*.
> At ~800 MiB of desktop it thrashes to 6.4 t/s. Same binary, same flags, 4× slowdown,
> and WDDM makes it **silent**: the server still starts, still answers, and still reports
> a similar `nvidia-smi memory.used`. This is risk #6 in §11, which was rated Low-Medium
> and is in fact the binding constraint. Hence the 12K default (§7) — it holds full speed
> with ~845 MiB of desktop budget. Decode rate, not `memory.used`, is the fit test.

---

## 4. Two API surfaces, one server

| Surface | Endpoint | Used by | Template handling |
|---|---|---|---|
| Chat (existing) | `/v1/chat/completions` | `bonsai_client.py`, harnesses, `bonsai` skill | server-side (`--jinja`), `chat_template_kwargs.enable_thinking:false` per request |
| Raw (new) | `/completion` + `/slots/...` | hotshuttle core | **client-side**: the prompt compiler renders Qwen chat-format text itself (ARCHITECTURE.md §8) |

The orchestrator must use `/completion` because `id_slot` + byte-stable append-only
prompts are the whole re-prefill-avoidance mechanism; a server-side template re-render
per request cannot be trusted to be byte-identical-prefix-stable. Consequence: the
prompt compiler owns the Qwen template tokens **and the thinking-mode suppression** (the
known Bonsai gotcha — it otherwise burns its whole budget in `<think>`). That means
emitting the empty-think form (`<think>\n\n</think>`) in the assistant-start position,
exactly as the Jinja template does when `enable_thinking=false`. Verified byte-for-byte
in M0 against a template render.

---

## 5. Package layout — the `core/` vs `profiles/` seam

```
hotshuttle/
├─ core/                        # model-agnostic; NO Bonsai constants anywhere in here
│  ├─ client.py                 #   Llama — thin async HTTP client (completion/save/restore/erase/slots)
│  ├─ manifest.py               #   workers.yaml load + validation (schema in ARCHITECTURE.md §2)
│  ├─ worker.py                 #   Worker + Task dataclasses
│  ├─ pool.py                   #   SlotPool — LRU allocator over physical slots
│  ├─ orchestrator.py           #   dispatch / compact
│  ├─ scheduler.py              #   asyncio queue + GPU semaphore, idle-fill
│  ├─ compiler.py               #   prompt compiler: seed + append-only turns, template rendering
│  └─ profile.py                #   ModelProfile protocol (the seam)
├─ profiles/
│  └─ bonsai/
│     ├─ profile.py             #   BonsaiProfile: sizing constants, launch config, template quirks
│     └─ launch.py              #   wraps bonsai.sh / builds the llama-server argv from the profile
├─ experiments/
│  ├─ m1_save_restore.py        #   Milestone 1 verification (§9)
│  └─ m0_vram_budget.py         #   sizing-contradiction measurement (§3)
├─ tests/                       #   unit tests against a fake server; opt-in integration tests
├─ workers.yaml                 #   the live manifest
└─ (existing serving files unchanged at repo root)
```

**The `ModelProfile` seam** (per issue #7 — data, not logic):

```python
class ModelProfile(Protocol):
    name: str                       # "bonsai-27b"
    server_url: str                 # http://127.0.0.1:8080
    n_slots: int                    # 1 on 8 GB
    ctx_per_slot: int               # 16384
    slot_save_path: str
    per_slot_floor_bytes: int       # ~150 MiB recurrent state (0 for dense models)
    kv_bytes_per_token: int         # ~34 KiB @ q8_0
    sampling_defaults: dict         # {"temperature": 0.2, "top_p": 0.9} for deterministic worker output
    def render_seed(self, system: str, tools: list) -> str: ...   # template + thinking suppression
    def render_turn(self, instruction: str, attach: list[str]) -> str: ...
    def launch_argv(self) -> list[str]: ...
```

Rules of the seam: `core/` imports `profile.py`'s protocol only, never
`profiles.bonsai`; every Bonsai number, path, port, template quirk, and launch flag
lives in `profiles/bonsai/`. Enforced by a trivial test that greps `core/` for
"bonsai"/"prism"/sizing literals. Adding a second model later = write a new profile
(issue #7 recommends a **dense** mainline-llama.cpp model first, since dense
save/restore is better-supported).

Note: the manifest sketch in the brainstorm doc shows `server: http://127.0.0.1:7834`;
that port belongs to another local service — the actual Bonsai server is **:8080** (per
`bonsai.sh`), which is what the profile uses.

---

## 6. VRAM budget and the 1–2-hot-slot conclusion

All numbers config-derived (Qwen3.6-27B config) or measured, per the design source.

**Fixed costs on the 8 GB card:**

- Weights: ~6.7–7.1 GiB resident (2.125 bpw × 27.3B + higher-precision embeddings/norms).
  (`bench-results.md` measured 6.66 GiB for the file's load.)
- CUDA compute/context buffers: ~0.4–0.6 GiB.
- Windows desktop compositor: more if the card drives a display (can approach the whole
  remainder).

→ **Free for all resident slot state: ~0.4–0.8 GiB headless; less with a monitor.**

**Per-slot (per hot worker) cost — the surprise:**

- **DeltaNet recurrent state: ~150 MiB, fixed, f32, NOT quantizable.** Derived:
  `48 layers × 48 value heads × 128 × 128 × 4 B ≈ 144 MiB` + ~6 MiB conv state.
  llama.cpp keeps recurrent state in f32 deliberately — decoding is recurrent, so
  quantization error feeds into the next step and **accumulates over the sequence**
  (unlike attention-KV error, which stays local to its token). Treat as fixed. This
  floor, not the KV cache, is the dominant per-slot cost.
- **Attention KV: ~34 KiB/token @ q8_0, ~18 KiB/token @ q4_0** (16 layers × 4 KV heads ×
  256 head_dim × 2 (K+V) = 32,768 elements/token; the 256 head_dim makes it heavier per
  token than typical).

Per-worker cost using the **measured** rates (floor 149.6 MiB, q8_0 34.1 KiB/token,
q4_0 18.1 KiB/token — the config-derived predictions in the bullets above were correct):

| Per worker (floor + KV) | KV quant | 8K ctx | **12K ctx** | 16K ctx | 32K ctx |
|---|---|---|---|---|---|
| 149.6 MiB + KV | q8_0 | ~423 MiB | **~559 MiB** | ~696 MiB | ~1242 MiB |
| 149.6 MiB + KV | q4_0 | ~295 MiB | ~367 MiB | ~440 MiB | ~730 MiB |

Against a ~0.4–0.8 GiB budget: **1–2 hot slots, and only with modest context.**
16K@q8_0 fits one slot headless. True double-buffering (two workers simultaneously
resident) only fits at small (≤8K, q4) contexts — above that, worker switching is a RAM
ping-pong, not simultaneous residency.

**Slots are expensive; context is cheap** — the inverse of a dense transformer. So the
design leans on paging, not residency: everything beyond 1–2 hot workers lives in system
RAM as saved blobs (~150 MiB + context KV each; a dozen warm workers ≈ 2–3 GB of RAM —
trivial; 16 GB system RAM per `bench-results.md`).

**Swap cost is cheap and roughly constant** — the conclusion holds, but the estimate below
was optimistic by ~50×; see the measured note.

The 4060 Ti runs PCIe 4.0 **x8** (~12–13 GB/s real host↔device), and a worker payload at
working context lengths is tens-to-low-hundreds of MB → single-digit to low-double-digit ms
each way, barely growing with the worker's context. It hides completely inside the
orchestrator's own turn-processing plus Bonsai's seconds-long generation (~20–28 tok/s
decode measured; weight-bandwidth-bound at ~288 GB/s, so KV quant is a **capacity** lever
here, not a speed lever).

> **Measured (M1, 2026-07-22).** A 236.5 MiB q8_0 payload costs **170–1185 ms to save** and
> **123–531 ms to restore** — not the ~10 ms the PCIe arithmetic predicts, because
> `--slot-save-path` round-trips through the Windows filesystem, not straight over PCIe.
> The variance is OS page-cache behaviour. q4_0 payloads (195.7 MiB) are consistently at
> the fast end (~170 ms save, ~125 ms restore).
>
> **The design conclusion survives comfortably:** re-prefilling that same worker costs
> 2611 tokens ÷ ~686 tok/s ≈ **3.8 s**, so restore is still ~10–30× cheaper than the thing
> it replaces, and the gap *widens* with context (prefill is linear in tokens; the blob is
> ~flat). It no longer disappears entirely inside one orchestrator turn, though, so the
> idle-fill overlap in §7/M4 is doing real work rather than hiding a rounding error.
> If save latency ever dominates, that is the trigger for the C-API route
> (`llama_state_seq_get_data`/`set_data`, §3 Route 2) — which is exactly the ~10 ms path.

**Levers for more concurrency, in order of preference:** page to RAM (cheap) → shorten
context (cheap) → q4 the **K**-cache only, V stays q8 (small quality risk) → offload
weight layers to CPU (~15–30 % slower decode per the estimate; llama.cpp's
flash-attention + *partial*-offload path has known bugs — re-verify output quality if
used).

---

## 7. Context length: 12K default, 16K and 32K opt-in

> **Revised by M0 (2026-07-22): the default is 12288, not 16384.** 16K @ q8_0 measured
> at full speed and remains available via `BONSAI_CTX=16384`, but it needs the Windows
> desktop under ~680 MiB of VRAM and degrades silently past that (§3). 12K @ q8_0 holds
> 30.2 t/s with ~845 MiB of desktop budget. Force 2 and 3 below are unchanged and still
> argue for the shorter window; force 1 is what moved.

The live tuning decision, locked as follows (three independent forces, same direction):

1. **VRAM.** 12K@q8_0 (~7345 MiB total server footprint, measured) leaves enough headroom
   for a normal desktop; 16K@q8_0 (~7507 MiB) fits only a quiet one; 32K@q8_0 does not fit
   at all — running 32K *forces* q4 KV. So the choice of length is also a choice of KV
   precision.
2. **Low-bit quant wrecks long context specifically.** Measured elsewhere
   (RULER/ONERULER): 8-bit KV holds ~99 % accuracy to 64K, but **4-bit KV drops
   long-context retrieval by ~16–23 %, worsening as input grows**. Bonsai's weights are
   already 2-bit (already −7.4 % on tool-calling vs FP16, per PrismML's own card) — 32K
   stacks the most quant-degraded task on the forced-q4 KV on already-2-bit weights.
3. **The orchestrator makes long context unnecessary.** By design, each worker receives
   a compact, purpose-built prompt (compaction-as-reset, §8/ARCHITECTURE.md); a scoped
   worker rarely needs even 16K. The 262K native window is nominal; the usable window of
   a 2-bit 27B is much shorter. Also: prefill is compute-bound and this card is slow —
   a 32K prompt is real seconds before the first token, paid on every cold worker.

**KV precision default: q8_0/q8_0** (near-lossless: +0.002–0.05 ppl on normal models;
`bench-results.md` measured q8_0 at +0.04 % on this exact build). q4_0 measured equally
clean **on perplexity** (+0.03 %) — but perplexity is not the metric that matters for a
workforce; **tool-call success rate** is, and that is the axis this model is already
weakest on, so bias conservative. If a second slot must be squeezed: drop **K** to q4_0,
keep **V** at q8_0 (K tolerates quantization far better; q4 V is ~3–4× worse than q4 K).

**32K "long-input" mode** stays available behind `BONSAI_LONGCTX=1` for the rare task
that must ingest a long document, accepting forced q4 KV and degraded retrieval. If a
task truly needs >16K, prefer decomposing it or handing it to the orchestrator's own
(frontier) model over stretching a 2-bit worker.

---

## 8. Orchestration design summary

Full component specs, code-level signatures, the `workers.yaml` schema, and the
invariants are in [`docs/ARCHITECTURE.md`](ARCHITECTURE.md). The load-bearing rules:

1. **One writer per slot.** Never dispatch to, save, or restore a slot that is
   `is_processing`. A slot is locked from dispatch until its completion returns.
2. **Save before evict, always.** Reusing a slot without saving the occupant discards
   its state → silent re-prefill on that worker's next turn.
3. **Stable prefix, append-only body.** `cache_prompt` reuses only the *common leading
   prefix*. A worker's prompt is `[seed: system+tools(+summary)] + [turn 1] + [turn 2] +
   …`, grown strictly by appending. Rewriting anything early invalidates the cache from
   that point.
4. **Compaction = reset, not edit.** You cannot shrink an append-only context in place.
   At `compact_at × ctx_budget`, the orchestrator retires the worker's state and
   re-seeds a fresh instance whose new stable prefix embeds a summary. "Hand the worker
   a compact prompt" happens at construction and at each reset — never mid-life.

Components: `Llama` client → `Worker`/`Task` → `SlotPool` (LRU; save-on-evict,
restore-on-fill) → `Orchestrator` (dispatch, compact) → async scheduler whose
concurrency is a GPU semaphore sized to `n_slots` (idle-fill falls out for free: while
the orchestrator processes worker A's response off-GPU, task B grabs the slot).

---

## 9. Milestones and build order

Smallest testable slice first. Correctness of paging + compaction is proven before any
concurrency; everything above `n_slots=1` is an optimization.

### M0 — Serving config + budget verification ✅ DONE 2026-07-22

Land the §3 flag changes behind env toggles (`BONSAI_SLOTS`, `BONSAI_SLOT_DIR`,
`BONSAI_KV_QUANT`); default behavior for existing callers unchanged until verified.
Then measure. Verifier: `experiments/m0_verify.py` (`--measure` for the VRAM matrix).

**Acceptance — all met:**
- ✅ Server starts with `--slots --slot-save-path`; `GET /slots` returns slot state with
  `is_processing`; `POST /slots/0?action=save` writes a 149.8 MiB blob;
  `?action=restore` returns ok.
- ✅ Sizing contradiction resolved (§3): KV ~37–40 KiB/token @ q8_0 (bench said 128),
  recurrent state 149.626 MiB/slot. `bench-results.md` corrected.
- ✅ 16K@q8_0 confirmed to fit **and** found to be desktop-VRAM-marginal; default revised
  to **12288 @ q8_0** with the measured numbers in §7 and `bench-results.md`.
- ✅ Template ground truth captured from `/apply-template`:
  `enable_thinking=false` appends `<|im_start|>assistant\n<think>\n\n</think>\n\n` — the
  empty-think form §4 predicted. Saved to `experiments/m0_template_ground_truth.txt`;
  the compiler is asserted against it in M2.

**Also found (not in the plan):** `bash` invoked from Python resolves to WSL's
`System32\bash.exe`, not git-bash — Windows searches System32 before `PATH`. WSL bash
cannot see `C:/…` paths, so `bonsai_client.ensure_up()` was silently failing with
rc=127. Fixed by resolving the interpreter via `shutil.which`, plus `*.sh text eol=lf`
in `.gitattributes` (WSL bash also chokes on a CRLF shebang).

### M1 — THE open risk: does save/restore skip re-prefill on the PrismML fork? ✅ PASS 2026-07-22

> **Answer: yes, on every leg tested.** `experiments/m1_save_restore.py`, 7 legs
> (round-trip and after-server-restart × q8_0 and q4_0, plus a control). A 2611-token
> worker, saved, evicted by prefilling a *different* worker into its slot, restored, then
> continued with a 26-token new turn:
>
> | | cold prefill | warm continuation |
> |---|---:|---:|
> | tokens evaluated | 2611 | **28** |
> | tokens reused from cache | 0 | **2615** |
>
> 1.1 % of the full prompt — well inside the ≤ suffix+64 pass criterion. The planted fact
> came back verbatim, and it survives a **full server restart**, so warm state genuinely
> outlives the process. Save 170–1185 ms, restore 123–531 ms (see the §6 note: slower than
> predicted, still ~10–30× cheaper than the 3.8 s re-prefill it replaces).
>
> **The metric in this section and in ARCHITECTURE.md §3 was wrong and has been corrected.**
> Both said to assert on the response's `tokens_evaluated`. That field is the *full prompt
> length* whether or not anything was cached — it read 2643 on a run that actually
> evaluated 28 tokens. The real telemetry is `timings.prompt_n` (evaluated) and
> `timings.cache_n` (reused). Asserting on `tokens_evaluated`, as originally specified,
> would have passed unconditionally and made M2's re-prefill regression test decorative.
> This cost a false FAIL here before the control leg caught it, which is why the control
> (same continuation, no save/restore, no eviction) is now a permanent part of the
> experiment: it separates "paging is broken" from "the measurement is broken".

**Original analysis, retained for provenance:**

Whether slot save/restore actually avoids re-prefill for this **hybrid/recurrent** model
on the fork is **unverified**, and it is the premise of the whole design. Upstream
history is exactly why: llama.cpp **#22384** (hybrid/recurrent checkpoint restore fixed)
and **#19794** (hybrid prompt cache forcing full re-processing) show hybrid checkpoints
getting invalidated and forcing full re-prefill. Fixes have landed on mainline and
Qwen3-Next slot restore is reported working — but the PrismML fork's base revision is
unknown, so it may predate them.

**Experiment** (`experiments/m1_save_restore.py`, runnable standalone):

1. Dispatch worker A: `/completion` with `id_slot=0`, `cache_prompt=true`, a ~2,000-token
   seed+turn prompt. Record `timings.prompt_n` (expect ≈ full prompt — cold prefill).
2. `POST /slots/0?action=save` `{"filename":"A.bin"}`.
3. Perturb the slot: erase it, or prefill a different worker B into it.
4. `POST /slots/0?action=restore` `{"filename":"A.bin"}`.
5. Dispatch A's follow-up: the identical prior prompt + a ~50-token appended turn, same
   slot, `cache_prompt=true`. Record `timings.prompt_n`.

**Pass:** step-5 `timings.prompt_n` ≈ the appended suffix only — operationally
`prompt_n ≤ suffix_tokens + 64` (allowing template-boundary slop), i.e. **< 10 % of the
full prompt length**. Also: A's continuation is coherent with its pre-save context (a
planted fact from the seed is recalled), guarding against a restore that "succeeds" with
corrupt state.
**Fail:** `prompt_n` ≈ the whole prompt (re-prefill), or the recalled fact is
lost/garbled.

> Use `timings.prompt_n` / `timings.cache_n`, **not** `tokens_evaluated` — the latter is
> the full prompt length regardless of caching and always "passes".

Run the matrix: q8_0 and q4_0 KV × (save/restore round-trip, restore-after-server-
restart). Record ms per save/restore (informs §6's swap-cost claim).

**If it fails — documented fallback:** the architecture stands; the **cost model**
changes. Idle-fill and slot multiplexing still work, but every worker switch pays a
prefill of that worker's full context (~seconds at 16K on this card's ~650–790 tok/s
ingest) instead of a ~10 ms blob copy. Mitigations, in order: keep contexts shorter
(prefill cost is linear); prefer scheduling consecutive turns of the same worker
(affinity beats LRU); check whether the fork can rebase onto a mainline revision
containing the #22384 fix; escalate to the C-API route (`llama_state_seq_*`) which the
fixes target more directly. Record the outcome in this file either way.

### M2 — SlotPool ping-pong: two workers, one slot

`Llama` client + `Worker` + `SlotPool` with `n_slots=1`; workers A and B alternate turns
(A1, B1, A2, B2, …), forcing save/evict/restore on every dispatch.

**Acceptance:** each worker's turn ≥2 evaluates only its new suffix (per-turn
`timings.prompt_n` assertions as in M1); both conversations stay coherent and
uncontaminated (each recalls its own planted fact, never the other's); pool invariants
hold under an interleaving stress test (no dispatch to a processing slot, no evict
without save); total wall-clock overhead of switching < 10 % of generation time.

### M3 — Compaction as reset

Add `Orchestrator.compact()`: at `compact_at × ctx_budget`, summarize (via the
orchestrator's own model or a scoped Bonsai call), erase the slot state, re-seed with
`system + "Prior progress:\n" + summary`, transcript reset to empty.

**Acceptance:** a long-running worker crosses the threshold and continues correctly
post-reset (planted early fact survives *via the summary*); post-compaction
`n_ctx_used` ≈ seed size; the fresh seed prefills once and subsequent turns are
suffix-only again; the retired blob file is deleted.

### M4 — Async scheduler + idle-fill

The `asyncio` queue + GPU semaphore (`n_slots`) loop; `n_slots + 1` worker loops so one
task is always queued behind the semaphore.

**Acceptance:** with two logical workers and orchestrator-side processing simulated at
~2 s/turn, GPU busy-fraction measurably exceeds the serial baseline (target: ≥ 25 %
wall-clock reduction on a scripted 10-turn workload — my estimate, tune after M2 gives
real switch timings); invariants still hold under concurrency (fuzzed interleavings);
graceful behavior when a task targets a worker mid-compaction.

### M5 — Manifest + profile hardening (wrap-up)

`workers.yaml` loading/validation, `BonsaiProfile` extracted per §5, the core/profile
grep test, end-to-end demo: an orchestrator script drives 4 logical workers through a
realistic fan-out (e.g. 4 files summarized then cross-referenced) on 1 hot slot.

**Acceptance:** demo completes unattended; total re-prefilled tokens across the run
< 15 % of total prompt tokens (i.e., paging is actually paying for itself); README
updated with orchestration quick-start.

---

## 10. Testing strategy

- **Unit tests (no GPU):** a fake llama-server (`aiohttp`/`FastAPI` stub) that models
  slot state, `cache_prompt` prefix-diffing, and save/restore files. SlotPool LRU,
  save-before-evict, lock discipline, compaction bookkeeping, and manifest validation
  all testable here. A **prefix-stability checker** asserts every prompt the compiler
  emits for a worker `startswith` the previous one — the append-only contract as a test.
- **Invariant fuzzing:** randomized task interleavings against the fake server; assert
  no dispatch-while-processing, no evict-without-save, no two workers claiming one slot.
- **Integration tests (opt-in, real server):** marked `@pytest.mark.gpu`; M1's
  experiment doubles as the canary. `timings.prompt_n` is tracked on every dispatch and
  asserted — silent re-prefill regressions become test failures, not vibes. (**Not**
  `tokens_evaluated`; see M1 §9 — it is the full prompt length and always passes.)
  Every paging test carries a **control leg** that exercises the same continuation with
  no save/restore and no eviction, so a broken measurement is distinguishable from
  broken paging.
- **Quality guard:** a small tool-call-success probe (N scoped tool-call prompts, parse
  rate measured) run at q8_0 vs q4_0 KV — because perplexity already told us nothing
  (§7), and tool-calling is the metric the workforce lives on.
- Test layout: `tests/` in-repo, pytest, same conventions as the rest of Cameron's
  repos.

---

## 11. Open risks and verification experiments

| # | Risk | Severity | Verification |
|---|---|---|---|
| 1 | ~~**Hybrid save/restore re-prefills on the PrismML fork** (upstream #22384/#19794 class)~~ | **CLOSED (M1) — does not occur** | 7/7 legs skip re-prefill (28 of 2643 tokens evaluated), including across a server restart. Fork post-dates the fixes |
| 2 | ~~Fork lacks slot endpoints entirely, or predates recurrent-state serialization~~ | **CLOSED (M0)** | Endpoints present and functional; the fork also logs hybrid context checkpoints, so it post-dates that work |
| 3 | ~~Sizing contradiction: bench's 64-layer KV formula vs config's 16-layer + 150 MiB floor~~ | **CLOSED (M0)** | Config reading confirmed; `bench-results.md` corrected |
| 4 | Recurrent state restore is *lossy/approximate* (coherent-looking but degraded) | Medium — **partially checked** | M1's planted-fact recall passes on all legs, but that is a weak probe: the fact also sits in the re-sent prefix, so it cannot distinguish "restored correctly" from "re-read from the prompt". The real check is M2's longer semantic probe + cross-worker contamination test |
| 5 | Byte-stable client-side template drifts from server Jinja (breaks prefix cache silently) | Medium | M0 byte-compare; unit prefix-stability checker |
| 6 | WDDM/display VRAM pressure makes 16K@q8 marginal on a monitor-driving card | **CONFIRMED (M0) — upgraded to High; this is the binding constraint, not KV size** | Measured: identical config runs 29.8 t/s at a 346 MiB desktop and 6.4 t/s at 845 MiB, silently. Mitigated by the 12K default; `--measure` re-runs the matrix when the desktop changes |
| 7 | q4-K squeeze (for slot #2) hurts tool-call rate more than perplexity suggested | Low | §10 quality guard probe |
| 8 | Save/restore blob I/O churn on spinning OS page cache (Windows, no tmpfs) | Low | measure in M2; escalate to C-API route only if real |
| 9 | `--parallel 2` interactions: `-c` splits across slots; continuous batching + hybrid | Low (post-M4) | 2-slot experiment after M4, headless |

Open questions needing a human decision:

1. **Where does compaction summarization run?** Options: the orchestrator's own model
   (best quality, costs API tokens), or a scoped Bonsai self-summarize (free, but the
   design source's own research warns weak models corrupt memory — "Silent Failure").
   Plan default: orchestrator-side; revisit after M3 measurements.
2. **Fork maintenance posture** if risk #2 materializes: rebase PrismML's fork ourselves
   vs. wait for PrismML. Not decidable until M0/M1 produce facts.

---

## 12. Relationship to existing issues

- **#7 (ModelProfile / multi-model)** — deferred scope; this plan implements the clean
  seam it requires (§5) and nothing more.
- **#5 (context paging tiers + paging client)** — subsumed by this plan (M0–M2 deliver
  a strictly stronger version); close it against this plan when M2 lands. Note #5's
  suggested flags (`--kv-unified --cache-ram --cache-idle-slots --ctx-checkpoints`) are
  from a different (mainline) feature-set exploration; §3's flag set supersedes them.
- **#6 (MCP wrapper)** — complementary future front-end for the orchestrator surface.
- **#1 (hardcoded paths)** — `profiles/bonsai/launch.py` should resolve paths via env
  (`BONSAI_DIR`) from day one and not add new hardcodes.
- **#2/#3/#4** — unaffected.
