"""SlotPool -- the register allocator.

Physical llama-server slots are registers; logical workers are variables, most of them
spilled to disk as saved state blobs. This class owns the free list, the residency map,
and the per-slot locks, and its evict path is the only code allowed to reuse an occupied
slot.

It upholds two of the invariants directly (docs/ARCHITECTURE.md section 1):

  I1  one writer per slot -- the slot's lock is held across the whole completion, and a
      slot that is locked is never chosen as an eviction victim.
  I2  save before evict, always -- reusing a slot without saving discards the occupant's
      attention KV *and* its recurrent state, which is a silent full re-prefill next time
      that worker runs.
"""
from __future__ import annotations

import asyncio
import os
from collections import Counter
from contextlib import asynccontextmanager

from .worker import Worker


class SlotPool:
    def __init__(self, llama, n_slots: int, slot_dir: str | None = None):
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}")
        self.llama = llama
        self.n_slots = n_slots
        self.slot_dir = slot_dir
        self.free: list[int] = list(range(n_slots))
        self.resident: dict[int, Worker] = {}
        self.locks = {i: asyncio.Lock() for i in range(n_slots)}
        self.stats: Counter[str] = Counter()
        # Allocation decisions are serialized; concurrent dispatches are bounded by the
        # slot count. Together these guarantee _evict_lru always finds an unlocked victim,
        # so it never has to block or spin. This is also the GPU semaphore the scheduler
        # would otherwise need -- concurrency is a property of the pool, not of its caller.
        self._alloc = asyncio.Lock()
        self._capacity = asyncio.Semaphore(n_slots)

    # --- the only public entry point ---------------------------------------

    @asynccontextmanager
    async def slot_for(self, w: Worker):
        """Yield a slot with this worker's state in it, holding the slot lock throughout.

        Paging (save the occupant, restore this worker) happens on the way in. Returning
        the slot to the pool is NOT the same as evicting the worker from it: the worker
        stays resident and hot until something else actually needs the slot, so a repeat
        turn for the same worker pages nothing at all.
        """
        await self._capacity.acquire()
        try:
            async with self._alloc:
                slot = await self._place(w)
                lock = self.locks[slot]
                await lock.acquire()
            try:
                yield slot
            finally:
                lock.release()
        finally:
            self._capacity.release()

    # --- placement ---------------------------------------------------------

    async def _place(self, w: Worker) -> int:
        if w.slot is not None:                       # already HOT -- nothing to page
            self.stats["hits"] += 1
            w.touch()
            return w.slot

        slot = self.free.pop() if self.free else await self._evict_lru()
        if w.kv_file:                                # WARM -> restore, skipping re-prefill
            await self.llama.restore(slot, w.kv_file)
            self.stats["restores"] += 1
        else:                                        # COLD -> clean slot; seed prefills next
            await self.llama.erase(slot)
            self.stats["cold_fills"] += 1
        w.slot = slot
        self.resident[slot] = w
        w.touch()
        return slot

    async def _evict_lru(self) -> int:
        # I1: an in-flight occupant is never a victim. The capacity semaphore means at
        # least one resident slot is always unlocked here.
        idle = {s: v for s, v in self.resident.items() if not self.locks[s].locked()}
        if not idle:
            raise RuntimeError(
                f"no evictable slot: all {self.n_slots} are mid-generation. This should be "
                f"impossible under the capacity semaphore -- did something bypass slot_for()?")
        slot, victim = min(idle.items(), key=lambda kv: kv[1].last_used)
        victim.kv_file = f"{victim.id}.bin"
        await self.llama.save(slot, victim.kv_file)  # I2 -- unconditional, before reuse
        victim.slot = None
        del self.resident[slot]
        self.stats["evictions"] += 1
        return slot

    # --- compaction support (I4) -------------------------------------------

    async def retire(self, w: Worker) -> None:
        """Drop a worker's state entirely: it is superseded by a fresh seed.

        Returns the worker to COLD. Safe to call on a worker in any state.
        """
        if w.slot is not None:
            async with self.locks[w.slot]:
                await self.llama.erase(w.slot)
            self.resident.pop(w.slot, None)
            if w.slot not in self.free:
                self.free.append(w.slot)
            w.slot = None
        if w.kv_file:
            self._delete_blob(w.kv_file)
            w.kv_file = None
        self.stats["retires"] += 1

    def _delete_blob(self, filename: str) -> None:
        if not self.slot_dir:
            return                                   # server-side path unknown; leave it
        try:
            os.remove(os.path.join(self.slot_dir, filename))
        except FileNotFoundError:
            pass

    # --- introspection -----------------------------------------------------

    def snapshot(self) -> dict:
        return {"free": sorted(self.free),
                "resident": {s: w.id for s, w in sorted(self.resident.items())},
                "stats": dict(self.stats)}
