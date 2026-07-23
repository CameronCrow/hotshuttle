"""Orchestrator -- dispatch (and, from M3, compaction).

Policy lives here; mechanism lives in client.py. The orchestrator decides what each
worker sees (I5), when a worker's context has grown enough to reset (I4), and hands the
paging decision to SlotPool.
"""
from __future__ import annotations

from . import compiler
from .pool import SlotPool
from .worker import Role, Task, Worker


class Orchestrator:
    def __init__(self, llama, pool: SlotPool, profile):
        self.llama = llama
        self.pool = pool
        self.profile = profile
        self.workers: dict[str, Worker] = {}
        self.guard = compiler.PrefixGuard()
        self.reprefilled = 0        # tokens re-evaluated across the run
        self.total_prompt = 0       # tokens prompted across the run

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

    async def dispatch(self, task: Task):
        """Run one turn for one worker, paging its state in if it is not already hot."""
        w = self.workers[task.worker_id]
        tmpl = self.profile.template

        # Append-only (I3): the turn is added before dispatch and never rewritten after.
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
        return resp

    @property
    def reprefill_ratio(self) -> float:
        """Share of prompt tokens that had to be re-evaluated. M5's acceptance bar is
        < 15% across a realistic run -- i.e. paging is actually paying for itself."""
        return self.reprefilled / self.total_prompt if self.total_prompt else 0.0
