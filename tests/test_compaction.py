"""M3: compaction is a reset, not an edit (I4).

The failure this guards against is subtle: an implementation that "compacts" by rewriting
the worker's transcript in place produces perfectly good output and silently re-prefills
the entire context on the next turn, because the prefix cache matches only up to the
first rewritten byte.
"""
from __future__ import annotations

import pytest

from hotshuttle.core.orchestrator import Orchestrator
from hotshuttle.core.pool import SlotPool
from hotshuttle.core.worker import Role, Task
from hotshuttle.profiles.bonsai import BonsaiProfile
from tests.fake import FakeLlama, tokenize


def build(summarize=None, ctx_budget=400, compact_at=0.8, reply=None):
    fake = FakeLlama(n_slots=1, reply=reply or (lambda p: "acknowledged"))
    profile = BonsaiProfile(n_slots=1)
    pool = SlotPool(fake, n_slots=1)
    orc = Orchestrator(fake, pool, profile, summarize=summarize)
    role = Role(name="w", system="You are a worker.", ctx_budget=ctx_budget,
                max_out=64, compact_at=compact_at)
    return fake, pool, orc, role


async def fixed_summary(w):
    return "PLANTED-FACT-8891 was established early."


@pytest.mark.asyncio
async def test_compaction_fires_at_the_threshold_and_shrinks_the_context():
    fake, pool, orc, role = build(summarize=fixed_summary, ctx_budget=200)
    w = orc.spawn("A", role)
    big = " ".join(f"word{i}" for i in range(150))

    await orc.dispatch(Task("A", "first"))
    assert w.compactions == 0
    before = w.n_ctx_used

    await orc.dispatch(Task("A", big))          # pushes past 0.8 * 200
    assert w.compactions == 1, f"expected compaction, n_ctx_used was {before}"
    assert w.transcript == ""
    assert w.n_ctx_used == 0
    assert w.state == "COLD"


@pytest.mark.asyncio
async def test_the_summary_rides_into_the_new_stable_prefix():
    fake, pool, orc, role = build(summarize=fixed_summary, ctx_budget=200)
    w = orc.spawn("A", role)
    await orc.dispatch(Task("A", " ".join(f"word{i}" for i in range(200))))
    assert w.compactions == 1
    assert "PLANTED-FACT-8891" in w.seed
    assert "Prior progress:" in w.seed
    assert "You are a worker." in w.seed          # the role survives the reset


@pytest.mark.asyncio
async def test_after_compaction_the_worker_prefills_once_then_is_suffix_only_again():
    fake, pool, orc, role = build(summarize=fixed_summary, ctx_budget=200)
    w = orc.spawn("A", role)
    await orc.dispatch(Task("A", " ".join(f"word{i}" for i in range(200))))
    assert w.compactions == 1

    # First turn on the new seed: cold, so it pays a prefill -- but a small one.
    fresh = await orc.dispatch(Task("A", "continue"))
    assert fresh.cache_n == 0
    assert fresh.prompt_tokens < 60, "the new seed should be far smaller than the old context"

    # And the turn after that is suffix-only again.
    nxt = await orc.dispatch(Task("A", "and again"))
    assert nxt.prompt_n <= len(tokenize("and again")) + 8
    assert nxt.cache_n > 0


@pytest.mark.asyncio
async def test_compaction_deletes_the_warm_blob(tmp_path):
    fake = FakeLlama(n_slots=1)
    profile = BonsaiProfile(n_slots=1)
    pool = SlotPool(fake, n_slots=1, slot_dir=str(tmp_path))
    orc = Orchestrator(fake, pool, profile, summarize=fixed_summary)
    role = Role(name="w", system="You are a worker.", ctx_budget=200, max_out=64)
    w = orc.spawn("A", role)
    orc.spawn("B", role)

    await orc.dispatch(Task("A", "hello"))
    await orc.dispatch(Task("B", "hello"))       # evicts A -> A is WARM, blob on disk
    assert w.kv_file
    blob = tmp_path / w.kv_file
    blob.write_bytes(b"x")                       # the fake does not touch the filesystem

    await orc.compact(w)
    assert not blob.exists(), "the superseded blob must not be left behind"
    assert w.kv_file is None


@pytest.mark.asyncio
async def test_prefix_guard_permits_the_new_seed():
    """Without guard.reset() in compact(), the next dispatch would trip the append-only
    assertion -- the whole point is that a compaction legitimately breaks the prefix."""
    fake, pool, orc, role = build(summarize=fixed_summary, ctx_budget=200)
    orc.spawn("A", role)
    await orc.dispatch(Task("A", " ".join(f"word{i}" for i in range(200))))
    await orc.dispatch(Task("A", "should not raise"))


@pytest.mark.asyncio
async def test_local_summarizer_runs_through_the_pool_and_cleans_up():
    """The fallback summarizer must not leave an ephemeral worker behind, and must not
    bypass the pool (a raw completion would displace a resident worker unsaved)."""
    fake, pool, orc, role = build(summarize=None, ctx_budget=200,
                                  reply=lambda p: "notes: alpha did the thing")
    w = orc.spawn("A", role)
    await orc.dispatch(Task("A", " ".join(f"word{i}" for i in range(200))))

    assert w.compactions == 1
    assert "notes: alpha did the thing" in w.seed
    assert [k for k in orc.workers if k.startswith("__summarize__")] == []
    assert len(pool.resident) <= 1
    assert set(pool.resident) | set(pool.free) == {0}


@pytest.mark.asyncio
async def test_summarizer_never_sees_raw_template_bytes():
    """Regression, M3: the transcript is rendered ChatML. Passing it to the summarizer
    verbatim nested a conversation inside a conversation, and Bonsai replied
    'Acknowledged.' to the embedded turn instead of summarizing anything -- producing an
    empty summary and silently losing everything the worker knew."""
    fake, pool, orc, role = build(summarize=None, ctx_budget=200, reply=lambda p: "notes")
    orc.spawn("A", role)

    summarizer_prompts = []
    original = fake.complete

    async def spy(prompt, id_slot, n_predict, cache_prompt=True, **kw):
        if "progress notes" in prompt or "compress a worker" in prompt:
            summarizer_prompts.append(prompt)
        return await original(prompt, id_slot, n_predict, cache_prompt, **kw)

    fake.complete = spy
    await orc.dispatch(Task("A", " ".join(f"word{i}" for i in range(200))))

    assert summarizer_prompts, "the local summarizer did not run"
    body = summarizer_prompts[0].split("<context>")[1].split("</context>")[0]
    assert "<|im_start|>" not in body and "<|im_end|>" not in body, \
        "the worker's log reached the summarizer as raw template bytes"
    assert "user:" in body or "assistant:" in body, "turn structure was lost entirely"


def test_to_plain_round_trips_a_rendered_transcript():
    t = BonsaiProfile().template
    rendered = (t.system("You are A.") + t.user("remember XR-3390")
                + t.assistant_open() + "Noted." + t.assistant_close())
    plain = t.to_plain(rendered)
    assert "<|im_start|>" not in plain and "<think>" not in plain
    assert "XR-3390" in plain and "Noted." in plain
    assert plain.startswith("system: You are A.")


@pytest.mark.asyncio
async def test_no_compaction_when_auto_compact_is_off():
    fake, pool, orc, role = build(summarize=fixed_summary, ctx_budget=200)
    w = orc.spawn("A", role)
    await orc.dispatch(Task("A", " ".join(f"word{i}" for i in range(200))),
                       auto_compact=False)
    assert w.compactions == 0
    assert w.transcript != ""
