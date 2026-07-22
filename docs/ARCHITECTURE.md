# hotshuttle — Architecture

Component-level specification for the orchestration layer. Read
[`docs/PLAN.md`](PLAN.md) first for vision, sizing, and milestones; this file pins down
the contracts each component must honor. Code sketches below are grounded in the design
source of record (`brainstorm-vault/Ideas/Moving Off Claude To Open Weights.md`,
"Orchestration layer", 2026-07-22) — they are the intended shape, not final code.

---

## 1. The invariants (normative)

Violate any of these and you either corrupt worker state or silently re-prefill.
Every component below exists to uphold one or more of them; the test suite asserts them
directly (PLAN.md §10).

**I1 — One writer per slot.** Never submit a completion to a slot that is
`is_processing`, and never save/restore/erase a slot mid-generation. A slot is locked
from dispatch until its completion returns. Mechanism: one `asyncio.Lock` per slot,
held across the entire complete() call.

**I2 — Save before evict, always.** Reusing a physical slot for another worker without
first saving the occupant discards its attention KV *and* DeltaNet recurrent state —
a silent full re-prefill the next time that worker runs (and for the recurrent state,
unrecoverable exact state). The pool's evict path is the only code allowed to reuse an
occupied slot, and it saves unconditionally first.

**I3 — Stable prefix, append-only body.** `cache_prompt=true` reuses only the *common
leading prefix* between the incoming prompt and the slot's cached tokens. Therefore a
worker's prompt is always:

```
[seed: system + tools (+ "Prior progress: <summary>")]  ← byte-stable for the worker's life
[turn 1: rendered instruction + attachments]            ← appended, never edited
[turn 1: model output]                                  ← appended verbatim
[turn 2: ...]                                           ← append-only, forever
```

If anything *early* in the prompt is rewritten — re-topping a rolling summary, editing a
prior turn, even a whitespace change in the seed — the cache invalidates from that byte
onward and the worker re-prefills. This is the single most important rule; it dictates
the compaction design (I4) and the prompt compiler (§8).

**I4 — Compaction = reset, not edit.** Because of I3 you cannot shrink a worker's
context in place. When a worker approaches its budget, the orchestrator *retires* its
state (erase slot / delete blob) and re-seeds a **fresh instance** whose new stable
prefix embeds a summary of prior progress. "Hand the worker a compact prompt" happens at
construction and at each reset — never mid-life.

**I5 — Curated context only.** The orchestrator never dumps its own conversation
history into a worker. A worker sees its seed + its own transcript + per-turn
attachments the orchestrator deliberately selected (`Task.attach`). This is what keeps
workers viable at 16K on a 2-bit model.

---

## 2. `workers.yaml` — the manifest

Declarative worker roles + runtime binding. Loaded and validated by `core/manifest.py`;
the `model:` block is populated from the active `ModelProfile` (values below are the
Bonsai profile's).

```yaml
model:
  server: http://127.0.0.1:8080       # llama-server hosting Bonsai (PrismML fork)
  n_slots: 1                          # physical HOT slots — 8 GB ⇒ 1 (PLAN.md §6)
  slot_save_path: C:/…/slots          # OS-page-cache-backed dir; warm workers spill here
  ctx_per_slot: 16384                 # 16K @ q8_0 KV — the standing default (PLAN.md §7)
  sampling_defaults: {temperature: 0.2, top_p: 0.9}   # deterministic worker output

roles:
  - name: code-reader
    system: |
      You are a focused code-reading worker. Answer ONE scoped question about the
      provided material. Output only the answer — no preamble, no restated question.
    tools: [read_file, grep]
    ctx_budget: 16384        # tokens; ≤ ctx_per_slot
    max_out: 1024            # n_predict cap per turn
    compact_at: 0.8          # fraction of ctx_budget that triggers reset-with-summary
  - name: test-writer
    system: |
      You write one pytest test for the provided function. Return only the test code.
    tools: [read_file]
    ctx_budget: 16384
    max_out: 1536
    compact_at: 0.8
```

Validation rules: `ctx_budget ≤ ctx_per_slot`; `0 < compact_at < 1`;
`max_out < ctx_budget`; role names unique; unknown keys rejected (fail loud — the
manifest is the operator's contract).

A **Task** is what the orchestrator enqueues — a worker instance, its role, the new
instruction, and the curated context for this turn (I5):

```python
@dataclass
class Task:
    worker_id: str            # which logical worker instance
    role: str                 # -> manifest role
    instruction: str          # the new turn
    attach: list[str] = ()    # snippets the orchestrator selected for THIS turn only
```

---

## 3. `Llama` — the mechanism client (`core/client.py`)

A thin async wrapper over the confirmed llama-server API. No policy, no retries beyond
transport-level, no model knowledge.

```python
class Llama:
    def __init__(self, base: str): self.base = base

    async def complete(self, prompt, id_slot, n_predict, cache_prompt=True, **sampling):
        # cache_prompt=True ⇒ server diffs `prompt` vs the slot's cached tokens and
        # evaluates ONLY the unseen suffix. This is the re-prefill avoidance.
        return await POST(f"{self.base}/completion", json={
            "prompt": prompt, "id_slot": id_slot, "n_predict": n_predict,
            "cache_prompt": cache_prompt, **sampling})

    async def save(self, id_slot, filename):     # attention KV + recurrent state -> slot_save_path/filename
        await POST(f"{self.base}/slots/{id_slot}?action=save",    json={"filename": filename})
    async def restore(self, id_slot, filename):
        await POST(f"{self.base}/slots/{id_slot}?action=restore", json={"filename": filename})
    async def erase(self, id_slot):
        await POST(f"{self.base}/slots/{id_slot}?action=erase", json={})
    async def slots(self):                        # GET /slots — state, is_processing, cached tokens
        return await GET(f"{self.base}/slots")
```

Response fields the layer above depends on: `content`, `tokens_evaluated` (prompt
tokens actually processed — the re-prefill telemetry), `tokens_predicted`, `timings`.
The client surfaces these verbatim; `tokens_evaluated` is asserted in tests on every
dispatch (a silent re-prefill regression becomes a test failure).

Uses the raw `/completion` endpoint, **not** `/v1/chat/completions` — `id_slot` plus
byte-stable prompts are the mechanism, and a server-side template re-render cannot be
trusted to preserve I3. Template rendering is the compiler's job (§8).

---

## 4. `Worker` — bookkeeping (`core/worker.py`)

A worker is (mostly) a pointer to a saved blob plus orchestrator-side bookkeeping:

```python
@dataclass
class Worker:
    id: str
    role: Role
    seed: str                     # stable prefix (I3): system + tools + optional "Prior progress: <summary>"
    transcript: str = ""          # append-only body (the worker's curated context, NOT the orchestrator's)
    kv_file: str | None = None    # set once spilled to RAM (WARM); None ⇒ COLD (never prefilled)
    slot: int | None = None       # physical slot if HOT
    n_ctx_used: int = 0
    last_used: float = 0.0
```

Lifecycle states: **COLD** (`kv_file is None`, `slot is None` — first dispatch prefills
the seed), **HOT** (`slot is not None` — state resident in VRAM), **WARM**
(`kv_file` set, `slot is None` — state saved under `slot_save_path`, restore skips
re-prefill). Compaction (I4) returns a worker to COLD with a new seed.

---

## 5. `SlotPool` — the LRU allocator (`core/pool.py`)

The register allocator. Owns the free list, residency map, and per-slot locks (I1);
its evict path is the sole implementation of I2.

```python
class SlotPool:
    def __init__(self, llama, n_slots):
        self.llama = llama
        self.free = list(range(n_slots))
        self.resident: dict[int, Worker] = {}     # slot -> worker currently in it
        self.locks = {i: asyncio.Lock() for i in range(n_slots)}

    async def acquire(self, w: Worker) -> int:
        if w.slot is not None:                    # already HOT
            w.last_used = clock(); return w.slot
        slot = self.free.pop() if self.free else await self._evict_lru()
        if w.kv_file:                             # WARM -> restore saved state (no re-prefill)
            await self.llama.restore(slot, w.kv_file)
        else:                                     # COLD -> clean slot; first complete() prefills the seed
            await self.llama.erase(slot)
        w.slot = slot; self.resident[slot] = w
        return slot

    async def _evict_lru(self) -> int:
        slot, victim = min(self.resident.items(), key=lambda kv: kv[1].last_used)
        victim.kv_file = f"{victim.id}.bin"
        await self.llama.save(slot, victim.kv_file)   # SPILL before reuse (I2)
        victim.slot = None; del self.resident[slot]
        return slot
```

Policy notes:
- **Treat it as a cache, not per-turn choreography.** If the resident slots already
  hold every live worker, no save/restore happens at all — requests just route and
  continuous batching interleaves. Explicit paging only kicks in once logical workers
  exceed the physical budget.
- Eviction is LRU by `last_used`. If M1's re-prefill verification *fails* (PLAN.md §9),
  the policy shifts from LRU toward **affinity** (prefer consecutive turns of the same
  worker) because switches become expensive.
- Eviction must only select victims whose slot lock is free (I1) — an `is_processing`
  occupant is never a victim.

---

## 6. `Orchestrator` — dispatch and compact (`core/orchestrator.py`)

```python
class Orchestrator:
    def __init__(self, llama, pool, cfg):
        self.llama, self.pool, self.cfg = llama, pool, cfg
        self.workers: dict[str, Worker] = {}

    async def dispatch(self, task: Task):
        w = self.workers[task.worker_id]
        turn = compiler.render_turn(task.instruction, task.attach)   # append-only (I3)
        w.transcript += turn
        slot = await self.pool.acquire(w)
        async with self.pool.locks[slot]:                            # I1: one writer per slot
            prompt = w.seed + w.transcript                           # stable prefix + append-only body
            resp = await self.llama.complete(
                prompt, id_slot=slot, n_predict=w.role.max_out,
                cache_prompt=True, **w.role.sampling)                # server evaluates only the new suffix
        w.transcript += resp.content
        w.n_ctx_used = resp.tokens_evaluated + resp.tokens_predicted
        w.last_used = clock()
        if w.n_ctx_used > w.role.ctx_budget * w.role.compact_at:
            await self.compact(w)                                    # I4: retire + re-seed
        return resp

    async def compact(self, w: Worker):
        summary = await self.summarize(w)          # orchestrator-side condensation of w.transcript
        if w.slot is not None:                     # drop the old state; it's superseded
            await self.llama.erase(w.slot)
            self.pool.resident.pop(w.slot, None); self.pool.free.append(w.slot); w.slot = None
        w.kv_file = None                           # COLD again; next dispatch re-prefills the new seed
        w.seed = compiler.render_seed(w.role.system, w.role.tools, prior=summary)
        w.transcript = ""                          # fresh body; summary rode into the stable seed
```

`summarize()` runs **orchestrator-side by default** (the driving model, or a scoped
one-shot call) — *not* as a self-managed worker memory operation. Rationale, from the
design source's research review: MemGPT-style self-managed paging is
worker-capability-bound, and small/low-bit models exhibit "Silent Failure" (fluent
output, corrupted memory). Bonsai is already −7.4 % on tool-calling vs FP16; handing it
its own memory curation asks it to do reliably the thing it is worst at. Open question
tracked in PLAN.md §11.

---

## 7. Scheduler + idle-fill (`core/scheduler.py`)

A GPU semaphore sized to `n_slots` is the whole concurrency model. While the
orchestrator does its *own* processing on worker A's response (its model turn, a tool
call, a retrieval), the next runnable task B grabs the slot and fills the GPU. With one
hot slot, dispatching B evicts A — fine, because A has already finished generating, so
A's state spills warm for its next turn.

```python
async def run(self):
    gpu = asyncio.Semaphore(self.cfg.n_slots)      # == physical hot slots
    async def worker_loop():
        while True:
            task = await self.queue.get()
            async with gpu:                        # serialize GPU access; paging happens inside
                resp = await self.dispatch(task)
            await self.handle(resp)                # orchestrator-side work runs OFF the GPU -> overlap
    await asyncio.gather(*[worker_loop() for _ in range(self.cfg.n_slots + 1)])  # +1 keeps one task queued
```

The `+1` loop keeps exactly one task queued behind the semaphore so the GPU never idles
waiting for the orchestrator; more than +1 buys nothing at `n_slots=1` and just grows
latency for re-prioritization.

---

## 8. Prompt compiler (`core/compiler.py` + profile hooks)

Owns everything about prompt *bytes*, because I3 makes bytes load-bearing.

Responsibilities:
1. **Seed rendering** — `render_seed(system, tools, prior=None)`: the model's chat
   template applied to the system block (+ tool definitions + optional
   "Prior progress" summary), producing the byte-stable prefix. Template specifics come
   from the `ModelProfile` (for Bonsai: Qwen chat format, **with thinking suppressed** —
   the empty `<think>\n\n</think>` form in the assistant-start position, byte-identical
   to what the server's Jinja template emits under `enable_thinking=false`; verified in
   M0). This is the raw-`/completion` equivalent of `bonsai_client.py`'s
   `chat_template_kwargs` trick.
2. **Turn rendering** — `render_turn(instruction, attach)`: user-turn template around
   the instruction plus the orchestrator-curated attachments (I5). Deterministic:
   same inputs ⇒ same bytes.
3. **Prefix-stability guarantee** — the compiler never emits a prompt for a worker that
   is not a byte-extension of the previous one (asserted in tests: every emitted prompt
   `startswith` its predecessor for the worker's current life).
4. **Token accounting** — estimates turn size pre-dispatch so the orchestrator can
   compact *before* blowing the budget rather than after a truncated generation.

Non-goals: no retrieval, no ranking, no summarization — those are orchestrator policy.
The compiler is a pure function from (role, history, task) to bytes.

---

## 9. `ModelProfile` seam (`core/profile.py`, `profiles/bonsai/`)

See PLAN.md §5 for the protocol and rules. Summary of the boundary:

| Lives in `core/` | Lives in `profiles/bonsai/` |
|---|---|
| SlotPool, Orchestrator, scheduler, manifest, Task/Worker | 150 MiB recurrent floor, 34 KiB/token KV, 1-slot/16K defaults |
| `/completion` + `/slots` client | server URL/port, launch argv, `bonsai.sh` wrapping |
| Append-only contract, compaction-as-reset | Qwen template + thinking suppression |
| ModelProfile *protocol* | ModelProfile *implementation* |

Enforcement: a unit test greps `core/` for Bonsai/PrismML identifiers and sizing
literals; CI-fails on leakage. Second profile deferred per
[issue #7](https://github.com/CameronCrow/hotshuttle/issues/7).
