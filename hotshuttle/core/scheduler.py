"""Scheduler -- keep the GPU busy while the orchestrator thinks.

The whole idea in one sentence: while the orchestrator does its *own* work on worker A's
reply (its model turn, a tool call, a retrieval), the next runnable task should already be
generating on the GPU. With one hot slot, dispatching B evicts A -- which is fine, because
A has finished generating, so its state spills warm for its next turn.

There is no semaphore here. Concurrency is bounded inside SlotPool, where it is a
correctness property (it is what guarantees eviction always finds an unlocked victim)
rather than something a caller has to remember. This module only decides *what runs next*.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .worker import Task


@dataclass
class SchedulerStats:
    dispatched: int = 0
    wall_ms: float = 0.0
    dispatch_ms: float = 0.0        # time inside dispatch, summed across loops. Includes
                                    # queueing for a slot, so it can exceed wall_ms.
    server_ms: float = 0.0          # time the server actually spent computing
    handle_ms: float = 0.0          # time in orchestrator-side processing
    errors: list[tuple] = field(default_factory=list)

    @property
    def gpu_busy_fraction(self) -> float:
        """Share of wall clock the server was computing. Bounded by n_slots, so with one
        slot this is a genuine 0-100% utilisation figure."""
        return self.server_ms / self.wall_ms if self.wall_ms else 0.0


class Scheduler:
    def __init__(self, orchestrator, handle=None, extra_loops: int = 1):
        """handle: async callable (Task, Completion) -> None, the orchestrator's own work.

        It runs OFF the GPU, which is exactly what creates the overlap worth scheduling
        for. extra_loops is how many tasks stay queued behind the slots so the GPU never
        waits on the orchestrator.

        docs/PLAN.md §7 says +1 is enough and more "buys nothing". M4 measured otherwise:
        +1 is enough only while orchestrator-side work is *shorter* than generation. Once
        it is longer, every loop can be sitting in handle() simultaneously and the slot
        goes idle no matter how deep the queue. Roughly, you want
        `ceil(handle_time / generation_time)` loops. Measured on an 8-turn run with 2 s of
        orchestrator work against ~1.4 s generations: +1 gave 15.7 s, +2 gave 12.7 s,
        +3 gave 13.2 s. Left at 1 because that is right for the common case (a frontier
        orchestrator turn is fast next to a local 27B generation); raise it when your
        handler is slow.
        """
        self.orc = orchestrator
        self.handle = handle
        self.queue: asyncio.Queue[Task | None] = asyncio.Queue()
        self.n_loops = orchestrator.pool.n_slots + extra_loops
        self.stats = SchedulerStats()

    def submit(self, *tasks: Task) -> None:
        for t in tasks:
            self.queue.put_nowait(t)

    async def run(self) -> SchedulerStats:
        """Process every queued task, then return. Tasks submitted while running are
        picked up; the loops exit once the queue is drained."""
        t0 = time.perf_counter()
        server0 = self.orc.server_ms
        loops = [asyncio.create_task(self._loop()) for _ in range(self.n_loops)]
        await self.queue.join()
        for task in loops:
            task.cancel()
        await asyncio.gather(*loops, return_exceptions=True)
        self.stats.wall_ms = (time.perf_counter() - t0) * 1000
        self.stats.server_ms = self.orc.server_ms - server0
        return self.stats

    async def _loop(self) -> None:
        while True:
            # Cancellation lands here, between tasks, so nothing is dropped mid-turn.
            task = await self.queue.get()
            try:
                t0 = time.perf_counter()
                resp = await self.orc.dispatch(task)
                self.stats.dispatch_ms += (time.perf_counter() - t0) * 1000
                self.stats.dispatched += 1
                if self.handle is not None:
                    t1 = time.perf_counter()
                    await self.handle(task, resp)      # off-GPU: another task fills the slot
                    self.stats.handle_ms += (time.perf_counter() - t1) * 1000
            except Exception as e:                      # one bad task must not stall the run
                self.stats.errors.append((task.worker_id, repr(e)))
            finally:
                self.queue.task_done()                  # or join() never returns


async def run_serial(orchestrator, tasks, handle=None) -> SchedulerStats:
    """The no-overlap baseline: dispatch, then handle, then the next task.

    This is what the scheduler has to beat, and by how much is bounded by the workload --
    overlap can only ever hide the smaller of (generation time, orchestrator time).
    """
    stats = SchedulerStats()
    t0 = time.perf_counter()
    server0 = orchestrator.server_ms
    for task in tasks:
        t1 = time.perf_counter()
        resp = await orchestrator.dispatch(task)
        stats.dispatch_ms += (time.perf_counter() - t1) * 1000
        stats.dispatched += 1
        if handle is not None:
            t2 = time.perf_counter()
            await handle(task, resp)
            stats.handle_ms += (time.perf_counter() - t2) * 1000
    stats.wall_ms = (time.perf_counter() - t0) * 1000
    stats.server_ms = orchestrator.server_ms - server0
    return stats
