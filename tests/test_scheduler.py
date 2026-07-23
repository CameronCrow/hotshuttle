"""M4: overlap orchestrator-side work with generation, without losing the invariants."""
from __future__ import annotations

import asyncio

import pytest

from hotshuttle.core.orchestrator import Orchestrator
from hotshuttle.core.pool import SlotPool
from hotshuttle.core.scheduler import Scheduler, run_serial
from hotshuttle.core.worker import Role, Task
from hotshuttle.profiles.bonsai import BonsaiProfile
from tests.fake import FakeLlama, SlotBusy


def build(n_slots=1, gen=0.02):
    fake = FakeLlama(n_slots=n_slots, latency=gen)
    profile = BonsaiProfile(n_slots=n_slots)
    pool = SlotPool(fake, n_slots=n_slots)
    return fake, pool, Orchestrator(fake, pool, profile)


def role(name):
    return Role(name=name, system=f"You are {name}.", ctx_budget=100000, max_out=64)


def handler(seconds):
    async def handle(task, resp):
        await asyncio.sleep(seconds)
    return handle


@pytest.mark.asyncio
async def test_scheduler_processes_every_task():
    fake, pool, orc = build()
    for wid in "AB":
        orc.spawn(wid, role(wid))
    s = Scheduler(orc, handle=handler(0.005))
    s.submit(*[Task("A" if i % 2 == 0 else "B", f"turn {i}") for i in range(10)])
    stats = await s.run()
    assert stats.dispatched == 10
    assert stats.errors == []


@pytest.mark.asyncio
async def test_overlap_beats_the_serial_baseline():
    """With orchestrator-side work comparable to generation, scheduling should recover
    most of what serial execution wastes."""
    tasks = [Task("A" if i % 2 == 0 else "B", f"turn {i}") for i in range(10)]

    _, _, serial_orc = build(gen=0.02)
    for wid in "AB":
        serial_orc.spawn(wid, role(wid))
    serial = await run_serial(serial_orc, tasks, handle=handler(0.02))

    _, _, sched_orc = build(gen=0.02)
    for wid in "AB":
        sched_orc.spawn(wid, role(wid))
    s = Scheduler(sched_orc, handle=handler(0.02))
    s.submit(*tasks)
    overlapped = await s.run()

    assert overlapped.dispatched == serial.dispatched == 10
    # Theoretical best is hiding all of the handle time behind generation.
    best = serial.wall_ms - serial.handle_ms
    achieved = serial.wall_ms - overlapped.wall_ms
    possible = serial.wall_ms - best
    assert achieved > 0.5 * possible, (
        f"overlap recovered {achieved:.0f}ms of a possible {possible:.0f}ms "
        f"(serial {serial.wall_ms:.0f}ms, scheduled {overlapped.wall_ms:.0f}ms)")


@pytest.mark.asyncio
async def test_invariants_hold_under_the_scheduler():
    fake, pool, orc = build(n_slots=1, gen=0.005)
    for wid in "ABCD":
        orc.spawn(wid, role(wid))
    s = Scheduler(orc, handle=handler(0.002))
    s.submit(*[Task("ABCD"[i % 4], f"turn {i}") for i in range(24)])
    stats = await s.run()

    assert stats.errors == [], stats.errors
    assert not any(isinstance(e, SlotBusy) for e in stats.errors)
    assert fake.busy == set()
    assert len(pool.resident) <= 1
    assert len(fake.ops("save")) == pool.stats["evictions"]


@pytest.mark.asyncio
async def test_same_worker_turns_are_serialized_not_interleaved():
    """Two queued turns for one worker must not interleave into its transcript -- the
    append-only contract has no meaning if they do."""
    fake, pool, orc = build(n_slots=2, gen=0.02)
    orc.spawn("A", role("A"))
    prompts = []
    original = fake.complete

    async def spy(prompt, id_slot, n_predict, cache_prompt=True, **kw):
        prompts.append(prompt)
        return await original(prompt, id_slot, n_predict, cache_prompt, **kw)

    fake.complete = spy
    s = Scheduler(orc, handle=handler(0))
    s.submit(Task("A", "one"), Task("A", "two"), Task("A", "three"))
    await s.run()

    assert len(prompts) == 3
    for earlier, later in zip(prompts, prompts[1:]):
        assert later.startswith(earlier), "same-worker turns interleaved"


@pytest.mark.asyncio
async def test_a_task_arriving_during_compaction_waits_for_it():
    """M4's 'graceful behavior when a task targets a worker mid-compaction'. The turn
    lock covers compaction too, so the queued turn sees the new seed, not a half-reset
    worker."""
    async def slow_summary(w):
        await asyncio.sleep(0.02)
        return "SUMMARY-MARKER"

    fake = FakeLlama(n_slots=1, latency=0.005)
    profile = BonsaiProfile(n_slots=1)
    orc = Orchestrator(fake, SlotPool(fake, 1), profile, summarize=slow_summary)
    w = orc.spawn("A", Role(name="A", system="You are A.", ctx_budget=120, max_out=32))

    s = Scheduler(orc, handle=handler(0))
    s.submit(Task("A", " ".join(f"word{i}" for i in range(100))),   # triggers compaction
             Task("A", "after"))
    stats = await s.run()

    assert stats.errors == [], stats.errors
    assert w.compactions == 1
    assert "SUMMARY-MARKER" in w.seed
    assert "after" in w.transcript          # the second turn landed on the fresh seed
    assert "word50" not in w.transcript     # and not on the old one


@pytest.mark.asyncio
async def test_a_failing_task_does_not_stall_the_queue():
    fake, pool, orc = build()
    orc.spawn("A", role("A"))
    s = Scheduler(orc, handle=handler(0))
    s.submit(Task("A", "ok"), Task("missing", "boom"), Task("A", "ok again"))
    stats = await s.run()
    assert stats.dispatched == 2
    assert len(stats.errors) == 1 and "missing" in stats.errors[0][0]
