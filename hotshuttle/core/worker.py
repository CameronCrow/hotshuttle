"""Worker / Role / Task -- orchestrator-side bookkeeping.

A worker is mostly a pointer to a saved state blob plus the bytes needed to reconstruct
its prompt. The model state itself lives in a slot (hot) or a file (warm).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Role:
    name: str
    system: str
    ctx_budget: int                # tokens; must be <= profile.ctx_per_slot. No default --
                                   # the right value is a property of the model, not of core.
    tools: tuple[str, ...] = ()
    max_out: int = 1024            # n_predict cap per turn
    compact_at: float = 0.8        # fraction of ctx_budget that triggers reset-with-summary
    sampling: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.compact_at < 1:
            raise ValueError(f"{self.name}: compact_at must be in (0, 1), got {self.compact_at}")
        if self.max_out >= self.ctx_budget:
            raise ValueError(f"{self.name}: max_out {self.max_out} >= ctx_budget {self.ctx_budget}")


@dataclass
class Task:
    """What the orchestrator enqueues: one turn for one worker, plus the context it chose
    for that turn (I5 -- the worker never sees the orchestrator's own history)."""
    worker_id: str
    instruction: str
    attach: tuple[str, ...] = ()


@dataclass
class Worker:
    """COLD  kv_file is None and slot is None -- next dispatch prefills the seed
    HOT   slot is not None            -- state resident in VRAM
    WARM  kv_file set, slot is None   -- state saved to disk, restore skips re-prefill

    Compaction returns a worker to COLD with a new seed (I4).
    """
    id: str
    role: Role
    seed: str                       # stable prefix (I3)
    transcript: str = ""            # append-only body
    kv_file: str | None = None
    slot: int | None = None
    n_ctx_used: int = 0
    last_used: float = 0.0
    compactions: int = 0

    @property
    def prompt(self) -> str:
        return self.seed + self.transcript

    @property
    def state(self) -> str:
        if self.slot is not None:
            return "HOT"
        return "WARM" if self.kv_file else "COLD"

    @property
    def needs_compaction(self) -> bool:
        return self.n_ctx_used > self.role.ctx_budget * self.role.compact_at

    def touch(self) -> None:
        self.last_used = time.monotonic()
