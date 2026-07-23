"""FakeLlama -- an in-process stand-in for llama-server.

docs/PLAN.md section 10 called for an aiohttp/FastAPI stub. This is the same coverage
without the dependency or the port: SlotPool and the compiler only ever touch the five
client methods, so modelling those directly exercises every policy path. The real HTTP
path is covered by the gpu-marked experiments (m1_save_restore.py, m2_pingpong.py).

What it models, because these are the behaviours the policy depends on:
  * per-slot cached token lists, and cache_prompt's leading-prefix diff
  * save/restore round-tripping that cached state through a named blob
  * a slot being busy mid-generation, so I1 violations raise instead of passing quietly
"""
from __future__ import annotations

import asyncio

from hotshuttle.core.client import Completion


def tokenize(text: str) -> list[str]:
    """Whitespace tokens. Not the model's tokenizer -- prefix-diff *semantics* are what
    the policy depends on, and those are identical at any granularity."""
    return text.split()


def common_prefix_len(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class SlotBusy(RuntimeError):
    """Raised on any operation that would violate I1 (one writer per slot)."""


class FakeLlama:
    def __init__(self, n_slots: int = 1, reply=None, latency: float = 0.0):
        self.n_slots = n_slots
        self.cached: dict[int, list[str]] = {i: [] for i in range(n_slots)}
        self.blobs: dict[str, list[str]] = {}
        self.busy: set[int] = set()
        self.log: list[tuple] = []
        self.latency = latency
        self._reply = reply or (lambda prompt: "ack")

    def _guard(self, id_slot: int, op: str) -> None:
        if id_slot in self.busy:
            raise SlotBusy(f"{op} on slot {id_slot} while it is generating (I1 violation)")

    async def complete(self, prompt, id_slot, n_predict, cache_prompt=True, **sampling):
        self._guard(id_slot, "complete")
        toks = tokenize(prompt)
        cached = self.cached[id_slot]
        n_cached = common_prefix_len(cached, toks) if cache_prompt else 0
        self.busy.add(id_slot)
        try:
            self.log.append(("complete", id_slot, len(toks) - n_cached, n_cached))
            await asyncio.sleep(self.latency)          # a real generation yields; so must this
            content = self._reply(prompt)
            out = tokenize(content)
            self.cached[id_slot] = toks + out
        finally:
            self.busy.discard(id_slot)
        # Report the simulated latency as server compute so scheduler stats are exercised.
        return Completion(content=content, prompt_n=len(toks) - n_cached, cache_n=n_cached,
                          predicted_n=len(out), id_slot=id_slot,
                          prompt_ms=self.latency * 1000 * 0.5,
                          predicted_ms=self.latency * 1000 * 0.5)

    async def save(self, id_slot, filename):
        self._guard(id_slot, "save")
        self.log.append(("save", id_slot, filename))
        self.blobs[filename] = list(self.cached[id_slot])

    async def restore(self, id_slot, filename):
        self._guard(id_slot, "restore")
        if filename not in self.blobs:
            raise KeyError(f"no such blob: {filename}")
        self.log.append(("restore", id_slot, filename))
        self.cached[id_slot] = list(self.blobs[filename])

    async def erase(self, id_slot):
        self._guard(id_slot, "erase")
        self.log.append(("erase", id_slot))
        self.cached[id_slot] = []

    async def slots(self):
        return [{"id": i, "is_processing": i in self.busy,
                 "n_cached": len(self.cached[i])} for i in range(self.n_slots)]

    # --- inspection --------------------------------------------------------

    def ops(self, kind: str) -> list[tuple]:
        return [e for e in self.log if e[0] == kind]

    def timeline(self, id_slot: int) -> list[str]:
        """Op kinds applied to one slot, in order -- e.g. ['erase', 'complete', 'save']."""
        return [e[0] for e in self.log if e[1] == id_slot]
