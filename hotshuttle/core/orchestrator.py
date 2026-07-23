"""Orchestrator -- dispatch (and, from M3, compaction).

Policy lives here; mechanism lives in client.py. The orchestrator decides what each
worker sees (I5), when a worker's context has grown enough to reset (I4), and hands the
paging decision to SlotPool.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict

from . import compiler
from .pool import SlotPool
from .worker import Role, Task, Worker


SUMMARIZER_SYSTEM = (
    "You compress a worker's progress log so the worker can continue after its context "
    "is reset. Preserve concrete facts, identifiers, file paths, decisions taken, and "
    "anything the worker was told to remember. Drop restated instructions and pleasantries. "
    "Write terse notes, not prose. Output only the notes.")


class Orchestrator:
    def __init__(self, llama, pool: SlotPool, profile, summarize=None):
        """summarize: async callable (Worker) -> str, used when a worker hits its budget.

        Supply one that runs on YOUR model. The design source's research is explicit that
        MemGPT-style self-managed paging is worker-capability-bound, and that small
        low-bit models fail at it silently -- fluent output over corrupted memory. Bonsai
        is already -7.4% on tool-calling versus FP16, so asking it to curate its own memory
        asks it to do reliably the thing it is worst at.

        Left as None it falls back to `_local_summarize`, which does exactly that
        discouraged thing on the local model. That keeps the package runnable standalone
        (and is what the M3 experiment exercises), but an orchestrator driving this from a
        frontier model should pass its own. Tracked as open question 1 in docs/PLAN.md.
        """
        self.llama = llama
        self.pool = pool
        self.profile = profile
        self.summarize = summarize or self._local_summarize
        self.workers: dict[str, Worker] = {}
        self.guard = compiler.PrefixGuard()
        self.reprefilled = 0        # tokens re-evaluated across the run
        self.total_prompt = 0       # tokens prompted across the run
        self.server_ms = 0.0        # time the server spent computing
        # One turn at a time per worker. A worker's transcript is append-only and its
        # compaction rewrites the seed, so two concurrent turns for the same worker would
        # interleave into each other's context. The orchestrator is the serialization
        # point; the pool serializes slots, which is a different thing.
        self._turn_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # --- worker lifecycle --------------------------------------------------

    def spawn(self, worker_id: str, role: Role, prior: str | None = None) -> Worker:
        if worker_id in self.workers:
            raise ValueError(f"worker {worker_id} already exists")
        seed = compiler.render_seed(self.profile.template, role.system,
                                    list(role.tools), prior=prior)
        w = Worker(id=worker_id, role=role, seed=seed)
        self.workers[worker_id] = w
        return w

    # --- the hot path ------------------------------------------------------

    async def dispatch(self, task: Task, auto_compact: bool = True):
        """Run one turn for one worker, paging its state in if it is not already hot.

        Concurrent dispatches for *different* workers overlap freely (that is the whole
        point of the scheduler); concurrent dispatches for the *same* worker queue behind
        each other, including behind a compaction it triggered.
        """
        w = self.workers[task.worker_id]
        tmpl = self.profile.template

        async with self._turn_locks[w.id]:
            # Append-only (I3): the turn is added before dispatch, never rewritten after.
            w.transcript += compiler.render_turn(tmpl, task.instruction, list(task.attach))
            prompt = w.prompt
            self.guard.check(w.id, prompt)

            sampling = {**self.profile.sampling_defaults, **w.role.sampling}
            async with self.pool.slot_for(w) as slot:
                resp = await self.llama.complete(prompt, id_slot=slot,
                                                 n_predict=w.role.max_out,
                                                 cache_prompt=True, **sampling)

            w.transcript += compiler.close_turn(tmpl, resp.content)
            w.n_ctx_used = resp.n_ctx_used
            w.touch()
            self.reprefilled += resp.prompt_n
            self.total_prompt += resp.prompt_tokens
            self.server_ms += resp.server_ms

            if auto_compact and w.needs_compaction:
                await self.compact(w)
            return resp

    # --- compaction (I4) ---------------------------------------------------

    async def compact(self, w: Worker) -> str:
        """Retire a worker's state and re-seed it with a summary of what it had done.

        You cannot shrink an append-only context in place: rewriting anything early
        invalidates the prefix cache from that byte on (I3), so "compaction" that edited
        the transcript would silently re-prefill everything after the edit. Instead the
        old state is destroyed and the worker starts a fresh life whose new stable prefix
        embeds the summary.
        """
        summary = await self.summarize(w)
        await self.pool.retire(w)                # erase the slot, delete the warm blob
        w.seed = compiler.render_seed(self.profile.template, w.role.system,
                                      list(w.role.tools), prior=summary)
        w.transcript = ""
        w.n_ctx_used = 0
        w.compactions += 1
        self.guard.reset(w.id)                   # the new seed legitimately differs
        return summary

    async def _local_summarize(self, w: Worker) -> str:
        """Fallback summarizer: a scoped one-shot on the local model. See __init__.

        Runs as an ephemeral worker rather than a bare completion so the pool still owns
        every slot transition -- a raw call would displace whatever is resident without
        saving it, which is exactly the I2 violation the pool exists to prevent.
        """
        sid = f"__summarize__{w.id}__{w.compactions}"
        # The summary has to fit in the worker's next seed, so cap it against that budget
        # rather than a fixed size -- a quarter of the budget leaves room for the role and
        # several turns before the next compaction.
        budget = w.role.ctx_budget
        role = Role(name="summarizer", system=SUMMARIZER_SYSTEM,
                    ctx_budget=budget, max_out=min(384, max(32, budget // 4)),
                    sampling={"temperature": 0.0})
        s = self.spawn(sid, role)
        try:
            # to_plain, not the raw transcript: a transcript is rendered template bytes,
            # and handing those back as content nests one conversation inside another.
            resp = await self.dispatch(
                Task(sid, "Below is a worker's log. Write its progress notes.",
                     attach=(self.profile.template.to_plain(w.transcript),)),
                auto_compact=False)              # never recurse into compaction
            return resp.content.strip()
        finally:
            await self.pool.retire(s)
            self.workers.pop(sid, None)
            self.guard.reset(sid)

    @property
    def reprefill_ratio(self) -> float:
        """Share of prompt tokens that had to be re-evaluated. M5's acceptance bar is
        < 15% across a realistic run -- i.e. paging is actually paying for itself."""
        return self.reprefilled / self.total_prompt if self.total_prompt else 0.0
