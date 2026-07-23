"""M2: two workers, one slot. The paging path, and the invariants that keep it honest."""
from __future__ import annotations

import asyncio
import random

import pytest

from hotshuttle.core.orchestrator import Orchestrator
from hotshuttle.core.pool import SlotPool
from hotshuttle.core.worker import Role, Task
from hotshuttle.profiles.bonsai import BonsaiProfile
from tests.fake import FakeLlama, SlotBusy, tokenize


def build(n_slots=1, latency=0.0, reply=None):
    fake = FakeLlama(n_slots=n_slots, latency=latency, reply=reply)
    profile = BonsaiProfile(n_slots=n_slots)
    pool = SlotPool(fake, n_slots=n_slots)
    orc = Orchestrator(fake, pool, profile)
    return fake, pool, orc


def role(name="r", ctx_budget=12288, **kw):
    return Role(name=name, system=f"You are {name}.", ctx_budget=ctx_budget, **kw)


# --- the M2 acceptance case --------------------------------------------------

@pytest.mark.asyncio
async def test_pingpong_two_workers_one_slot_only_evaluates_new_suffix():
    """A1, B1, A2, B2 on a single slot: every turn after the first for a given worker
    must evaluate only its new suffix, not its whole context."""
    fake, pool, orc = build(n_slots=1, reply=lambda p: "acknowledged " * 5)
    orc.spawn("A", role("A"))
    orc.spawn("B", role("B"))

    a1 = await orc.dispatch(Task("A", "First question for A." + " padding" * 200))
    b1 = await orc.dispatch(Task("B", "First question for B." + " padding" * 200))
    a2 = await orc.dispatch(Task("A", "Second question for A."))
    b2 = await orc.dispatch(Task("B", "Second question for B."))

    # Cold turns pay full prefill; there is nothing cached to reuse.
    assert a1.cache_n == 0 and a1.prompt_n == a1.prompt_tokens
    assert b1.cache_n == 0

    # Warm turns must reuse everything but the new turn. Slack of 8 tokens covers
    # template-boundary effects, mirroring M1's suffix+64 real-tokenizer budget.
    a2_suffix = len(tokenize("Second question for A."))
    assert a2.prompt_n <= a2_suffix + 8, f"A re-prefilled {a2.prompt_n} tokens"
    assert a2.cache_n > 200, "A's restored context was not reused"
    assert b2.prompt_n <= len(tokenize("Second question for B.")) + 8
    assert b2.cache_n > 200

    # And the paging actually happened -- B1 evicted A, A2 evicted B, B2 evicted A.
    assert pool.stats["evictions"] == 3
    assert pool.stats["restores"] == 2       # A2 and B2 restored from blobs
    assert pool.stats["cold_fills"] == 2     # A1 and B1 started clean


@pytest.mark.asyncio
async def test_workers_do_not_contaminate_each_other():
    """Each worker sees only its own transcript -- the classic failure of getting the
    slot bookkeeping wrong is B answering with A's context."""
    fake, pool, orc = build(n_slots=1, reply=lambda p: "SEEN:" + p.split("SECRET-")[-1][:4]
                            if "SECRET-" in p else "SEEN:none")
    orc.spawn("A", role("A"))
    orc.spawn("B", role("B"))
    await orc.dispatch(Task("A", "Remember SECRET-AAAA."))
    await orc.dispatch(Task("B", "You have no secret."))
    a2 = await orc.dispatch(Task("A", "Repeat it."))
    b2 = await orc.dispatch(Task("B", "Repeat it."))
    assert "AAAA" in a2.content
    assert "AAAA" not in b2.content


# --- invariants --------------------------------------------------------------

@pytest.mark.asyncio
async def test_i2_every_eviction_saves_first():
    fake, pool, orc = build(n_slots=1)
    orc.spawn("A", role("A"))
    orc.spawn("B", role("B"))
    await orc.dispatch(Task("A", "one"))
    await orc.dispatch(Task("B", "two"))
    await orc.dispatch(Task("A", "three"))

    assert len(fake.ops("save")) == pool.stats["evictions"] > 0
    # On the slot's timeline, the op immediately before every restore is a save: the
    # outgoing worker's state was captured before the incoming worker's landed on top.
    tl = fake.timeline(0)
    for i, op in enumerate(tl):
        if op == "restore":
            assert tl[i - 1] == "save", f"restore at {i} not preceded by a save: {tl}"


@pytest.mark.asyncio
async def test_hot_worker_repeat_turn_pages_nothing():
    """Consecutive turns for the same worker must not save/restore at all."""
    fake, pool, orc = build(n_slots=1)
    orc.spawn("A", role("A"))
    await orc.dispatch(Task("A", "one"))
    await orc.dispatch(Task("A", "two"))
    await orc.dispatch(Task("A", "three"))
    assert fake.ops("save") == []
    assert fake.ops("restore") == []
    assert pool.stats["hits"] == 2


@pytest.mark.asyncio
async def test_i1_no_dispatch_to_a_busy_slot():
    """FakeLlama raises SlotBusy on any concurrent op; the pool's locking must prevent
    it from ever being reached."""
    fake, pool, orc = build(n_slots=1, latency=0.01)
    for wid in "ABCD":
        orc.spawn(wid, role(wid))
    await asyncio.gather(*[orc.dispatch(Task(wid, "go")) for wid in "ABCD"])
    assert fake.busy == set()


@pytest.mark.asyncio
async def test_two_slots_run_concurrently_but_stay_isolated():
    fake, pool, orc = build(n_slots=2, latency=0.01)
    for wid in "ABC":
        orc.spawn(wid, role(wid))
    await asyncio.gather(*[orc.dispatch(Task(wid, "go")) for wid in "ABC"])
    assert len(pool.resident) <= 2
    assert set(pool.resident) <= {0, 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(8))
async def test_fuzz_random_interleavings_hold_invariants(seed):
    """Randomized task orders against the fake. Any I1 violation raises SlotBusy; any
    lost state shows up as a worker re-prefilling a context it should have had cached."""
    rng = random.Random(seed)
    n_slots = rng.choice([1, 2])
    fake, pool, orc = build(n_slots=n_slots, latency=0.001)
    ids = [f"w{i}" for i in range(rng.randint(2, 5))]
    for wid in ids:
        orc.spawn(wid, role(wid))

    tasks = [Task(rng.choice(ids), f"turn {i}") for i in range(20)]
    batches = [tasks[i:i + 3] for i in range(0, len(tasks), 3)]
    try:
        for batch in batches:
            # Dedupe within a batch: two concurrent turns for the SAME worker would
            # interleave that worker's own transcript, which the append-only contract
            # forbids by construction (the orchestrator is the serialization point).
            seen, uniq = set(), []
            for t in batch:
                if t.worker_id not in seen:
                    seen.add(t.worker_id)
                    uniq.append(t)
            await asyncio.gather(*[orc.dispatch(t) for t in uniq])
    except SlotBusy as e:
        pytest.fail(f"I1 violated: {e}")

    assert fake.busy == set()
    assert len(pool.resident) <= n_slots
    assert set(pool.resident) | set(pool.free) == set(range(n_slots))
    assert len(fake.ops("save")) == pool.stats["evictions"]


# --- retire (compaction support) --------------------------------------------

@pytest.mark.asyncio
async def test_retire_returns_worker_to_cold_and_frees_the_slot():
    fake, pool, orc = build(n_slots=1)
    w = orc.spawn("A", role("A"))
    await orc.dispatch(Task("A", "one"))
    assert w.state == "HOT"
    await pool.retire(w)
    assert w.state == "COLD"
    assert w.slot is None and w.kv_file is None
    assert pool.free == [0] and pool.resident == {}
